#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>
#include <stdatomic.h>
#include <execinfo.h>

#include <cuda_runtime.h>
#include <cuda.h>

/*
 * CUDA VRAM IPC hook for sharing one selected cudaMalloc allocation
 * across multiple independent processes.
 *
 * Build:
 *   gcc -shared -fPIC -O2 \
 *     -I/usr/local/cuda/include \
 *     cuda_vram_ipc_hook_fixed_v3.c \
 *     -o libcuda_vram_ipc_hook.so \
 *     -ldl
 *
 * Usage example:
 *   export MODEL_SIZE=49392123904
 *   export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_qwen3_30b_gpu0"
 *   LD_PRELOAD=/path/libcuda_vram_ipc_hook.so ./llama-server ...
 *
 * Optional:
 *   export MODEL_SIZE_TOLERANCE=1048576
 *   export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=10
 *   export CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=1
 *
 * Notes:
 * - This intentionally hooks only cudaMalloc/cudaFree.
 * - It resolves libcudart symbols lazily, which is important when llama.cpp
 *   is built with GGML_BACKEND_DL=ON and CUDA backend appears later via dlopen().
 * - The master process owns the real cudaMalloc allocation.
 * - Worker processes map it using cudaIpcOpenMemHandle().
 * - If master exits while workers are alive, workers may keep their IPC mapping
 *   on some driver/runtime combinations. This code does not try to touch them.
 */

#define DEFAULT_SHM_NAME "/cuda_vram_ipc_auto"
#define SHM_NAME_MAX 256
#define WAIT_USEC 100000

typedef struct {
    atomic_int is_ready;
    pid_t master_pid;
    size_t allocation_size;
    cudaIpcMemHandle_t handle;
} ipc_data_t;

/* Real CUDA runtime functions */
static cudaError_t (*real_cudaMalloc)(void **, size_t) = NULL;
static cudaError_t (*real_cudaFree)(void *) = NULL;
static cudaError_t (*real_cudaIpcGetMemHandle)(cudaIpcMemHandle_t *, void *) = NULL;
static cudaError_t (*real_cudaIpcOpenMemHandle)(void **, cudaIpcMemHandle_t, unsigned int) = NULL;
static cudaError_t (*real_cudaIpcCloseMemHandle)(void *) = NULL;

/* dlopen handle for libcudart when RTLD_NEXT cannot see it */
static void *cudart_handle = NULL;

/* Config */
static size_t target_shared_size = 0;
static size_t target_size_tolerance = 0;
static char shm_name[SHM_NAME_MAX] = DEFAULT_SHM_NAME;
static int shm_size_wait_sec = 10;
static bool suppress_master_free = false;
static bool trace_callers = false;
static bool trace_normal_allocs = false;
static int trace_depth = 8;

/* Per-process state */
static bool shared_allocation_done = false;
static bool is_master_process = false;
static void *master_alloc_ptr = NULL;
static void *worker_mapped_ptr = NULL;

/* Avoid accidental resolver recursion */
static __thread int in_hook = 0;

static bool env_is_true(const char *v) {
    if (!v || !v[0]) {
        return false;
    }

    return strcmp(v, "1") == 0 ||
           strcasecmp(v, "true") == 0 ||
           strcasecmp(v, "yes") == 0 ||
           strcasecmp(v, "on") == 0;
}

static void read_config_from_env(void) {
    const char *env_size = getenv("MODEL_SIZE");
    if (env_size && env_size[0]) {
        errno = 0;
        unsigned long long parsed = strtoull(env_size, NULL, 10);
        if (errno == 0) {
            target_shared_size = (size_t)parsed;
        }
    }

    const char *env_tol = getenv("MODEL_SIZE_TOLERANCE");
    if (env_tol && env_tol[0]) {
        errno = 0;
        unsigned long long parsed = strtoull(env_tol, NULL, 10);
        if (errno == 0) {
            target_size_tolerance = (size_t)parsed;
        }
    }

    const char *env_shm = getenv("CUDA_VRAM_IPC_NAME");
    if (env_shm && env_shm[0] == '/') {
        snprintf(shm_name, sizeof(shm_name), "%s", env_shm);
    }

    const char *env_wait = getenv("CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC");
    if (env_wait && env_wait[0]) {
        errno = 0;
        long parsed = strtol(env_wait, NULL, 10);
        if (errno == 0 && parsed > 0 && parsed < 3600) {
            shm_size_wait_sec = (int)parsed;
        }
    }

    suppress_master_free = env_is_true(getenv("CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE"));

    trace_callers = env_is_true(getenv("CUDA_VRAM_IPC_TRACE_CALLERS"));
    trace_normal_allocs = env_is_true(getenv("CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS"));

    const char *env_trace_depth = getenv("CUDA_VRAM_IPC_TRACE_DEPTH");
    if (env_trace_depth && env_trace_depth[0]) {
        errno = 0;
        long parsed = strtol(env_trace_depth, NULL, 10);
        if (errno == 0 && parsed > 0 && parsed <= 64) {
            trace_depth = (int)parsed;
        }
    }
}

__attribute__((constructor))
static void setup_hooks(void) {
    read_config_from_env();

    fprintf(stderr,
            "[VRAM_HOOK] loaded. MODEL_SIZE=%zu, TOLERANCE=%zu, SHM=%s, "
            "SHM_SIZE_WAIT_SEC=%d, SUPPRESS_MASTER_FREE=%d, "
            "TRACE_CALLERS=%d, TRACE_NORMAL_ALLOCS=%d, TRACE_DEPTH=%d\n",
            target_shared_size,
            target_size_tolerance,
            shm_name,
            shm_size_wait_sec,
            suppress_master_free ? 1 : 0,
            trace_callers ? 1 : 0,
            trace_normal_allocs ? 1 : 0,
            trace_depth);
}


static void print_symbol_line(const char *prefix, void *addr) {
    Dl_info info;
    memset(&info, 0, sizeof(info));

    if (dladdr(addr, &info) && info.dli_fname) {
        const char *sym = info.dli_sname ? info.dli_sname : "?";
        void *base = info.dli_fbase;
        unsigned long offset = 0;

        if (base && addr >= base) {
            offset = (unsigned long)((char *)addr - (char *)base);
        }

        fprintf(stderr,
                "[VRAM_HOOK][TRACE] %s %p %s %s +0x%lx\n",
                prefix,
                addr,
                info.dli_fname,
                sym,
                offset);
    } else {
        fprintf(stderr,
                "[VRAM_HOOK][TRACE] %s %p <unknown>\n",
                prefix,
                addr);
    }
}

static void trace_cuda_caller(const char *api_name, size_t size, void *ptr, cudaError_t result) {
    if (!trace_callers) {
        return;
    }

    fprintf(stderr,
            "[VRAM_HOOK][TRACE] %s(size=%zu, ptr=%p) -> code=%d\n",
            api_name,
            size,
            ptr,
            (int)result);

#if defined(__GNUC__)
    void *ret = __builtin_return_address(0);
    print_symbol_line("direct caller:", ret);
#endif

    if (trace_depth <= 0) {
        return;
    }

    void *frames[64];
    int max_depth = trace_depth;
    if (max_depth > 64) {
        max_depth = 64;
    }

    int n = backtrace(frames, max_depth);

    for (int i = 0; i < n; i++) {
        char label[64];
        snprintf(label, sizeof(label), "bt#%02d:", i);
        print_symbol_line(label, frames[i]);
    }
}

static void trace_cuda_free_caller(const char *api_name, void *ptr, cudaError_t result) {
    if (!trace_callers) {
        return;
    }

    fprintf(stderr,
            "[VRAM_HOOK][TRACE] %s(ptr=%p) -> code=%d\n",
            api_name,
            ptr,
            (int)result);

#if defined(__GNUC__)
    void *ret = __builtin_return_address(0);
    print_symbol_line("direct caller:", ret);
#endif

    if (trace_depth <= 0) {
        return;
    }

    void *frames[64];
    int max_depth = trace_depth;
    if (max_depth > 64) {
        max_depth = 64;
    }

    int n = backtrace(frames, max_depth);

    for (int i = 0; i < n; i++) {
        char label[64];
        snprintf(label, sizeof(label), "bt#%02d:", i);
        print_symbol_line(label, frames[i]);
    }
}

static bool size_matches_target(size_t size) {
    if (target_shared_size == 0) {
        return false;
    }

    if (size == target_shared_size) {
        return true;
    }

    if (target_size_tolerance == 0) {
        return false;
    }

    size_t lo = target_shared_size > target_size_tolerance
                    ? target_shared_size - target_size_tolerance
                    : 0;
    size_t hi = target_shared_size + target_size_tolerance;

    return size >= lo && size <= hi;
}

static int resolve_from_handle(void *handle) {
    if (!handle) {
        return -1;
    }

    if (!real_cudaMalloc) {
        real_cudaMalloc = (cudaError_t (*)(void **, size_t))dlsym(handle, "cudaMalloc");
    }
    if (!real_cudaFree) {
        real_cudaFree = (cudaError_t (*)(void *))dlsym(handle, "cudaFree");
    }
    if (!real_cudaIpcGetMemHandle) {
        real_cudaIpcGetMemHandle =
            (cudaError_t (*)(cudaIpcMemHandle_t *, void *))dlsym(handle, "cudaIpcGetMemHandle");
    }
    if (!real_cudaIpcOpenMemHandle) {
        real_cudaIpcOpenMemHandle =
            (cudaError_t (*)(void **, cudaIpcMemHandle_t, unsigned int))dlsym(handle, "cudaIpcOpenMemHandle");
    }
    if (!real_cudaIpcCloseMemHandle) {
        real_cudaIpcCloseMemHandle =
            (cudaError_t (*)(void *))dlsym(handle, "cudaIpcCloseMemHandle");
    }

    return 0;
}

static int resolve_cuda_symbols(void) {
    if (real_cudaMalloc &&
        real_cudaFree &&
        real_cudaIpcGetMemHandle &&
        real_cudaIpcOpenMemHandle &&
        real_cudaIpcCloseMemHandle) {
        return 0;
    }

    /*
     * First try RTLD_NEXT. This works when libcudart is already globally visible.
     */
    resolve_from_handle(RTLD_NEXT);

    if (real_cudaMalloc &&
        real_cudaFree &&
        real_cudaIpcGetMemHandle &&
        real_cudaIpcOpenMemHandle &&
        real_cudaIpcCloseMemHandle) {
        return 0;
    }

    /*
     * With GGML_BACKEND_DL=ON, CUDA backend/libcudart may appear later via dlopen().
     * Try existing loaded libcudart first, then load it if needed.
     */
    const char *libs[] = {
        "libcudart.so",
        "libcudart.so.13",
        "libcudart.so.12",
        NULL
    };

    for (int i = 0; libs[i] && !cudart_handle; i++) {
#ifdef RTLD_NOLOAD
        cudart_handle = dlopen(libs[i], RTLD_NOW | RTLD_NOLOAD | RTLD_GLOBAL);
#endif
        if (!cudart_handle) {
            cudart_handle = dlopen(libs[i], RTLD_NOW | RTLD_GLOBAL);
        }
    }

    if (!cudart_handle) {
        fprintf(stderr, "[VRAM_HOOK] ERROR: failed to open libcudart: %s\n", dlerror());
        return -1;
    }

    resolve_from_handle(cudart_handle);

    if (!real_cudaMalloc ||
        !real_cudaFree ||
        !real_cudaIpcGetMemHandle ||
        !real_cudaIpcOpenMemHandle ||
        !real_cudaIpcCloseMemHandle) {
        fprintf(stderr,
                "[VRAM_HOOK] ERROR: failed to resolve CUDA symbols: "
                "cudaMalloc=%p cudaFree=%p cudaIpcGetMemHandle=%p "
                "cudaIpcOpenMemHandle=%p cudaIpcCloseMemHandle=%p\n",
                (void *)real_cudaMalloc,
                (void *)real_cudaFree,
                (void *)real_cudaIpcGetMemHandle,
                (void *)real_cudaIpcOpenMemHandle,
                (void *)real_cudaIpcCloseMemHandle);
        return -1;
    }

    fprintf(stderr, "[VRAM_HOOK] CUDA symbols resolved via libcudart handle\n");
    return 0;
}

/*
 * Return:
 *   0  - shm size is OK
 *  -1  - OS error
 *  -2  - stale/broken shm: file exists but never reached expected size
 */
static int wait_for_shm_size(int shm_fd, size_t expected_size) {
    struct stat st;
    int loops = 0;
    int max_loops = (shm_size_wait_sec * 1000000) / WAIT_USEC;

    if (max_loops <= 0) {
        max_loops = 1;
    }

    for (;;) {
        if (fstat(shm_fd, &st) != 0) {
            perror("[VRAM_HOOK] fstat");
            return -1;
        }

        if ((size_t)st.st_size >= expected_size) {
            return 0;
        }

        if (loops == 0 || loops % 10 == 0) {
            fprintf(stderr,
                    "[VRAM_HOOK] Existing shm %s is not initialized yet: "
                    "size=%lld expected=%zu\n",
                    shm_name,
                    (long long)st.st_size,
                    expected_size);
        }

        if (loops >= max_loops) {
            fprintf(stderr,
                    "[VRAM_HOOK] Existing shm %s did not reach expected size within %d sec. "
                    "Treating as stale.\n",
                    shm_name,
                    shm_size_wait_sec);
            return -2;
        }

        usleep(WAIT_USEC);
        loops++;
    }
}

static ipc_data_t *map_ipc_data(int shm_fd) {
    ipc_data_t *data = (ipc_data_t *)mmap(NULL,
                                          sizeof(ipc_data_t),
                                          PROT_READ | PROT_WRITE,
                                          MAP_SHARED,
                                          shm_fd,
                                          0);
    if (data == MAP_FAILED) {
        perror("[VRAM_HOOK] mmap");
        return NULL;
    }

    return data;
}

static bool process_is_dead(pid_t pid) {
    if (pid <= 0) {
        return true;
    }

    if (kill(pid, 0) == -1 && errno == ESRCH) {
        return true;
    }

    return false;
}

static cudaError_t fallback_real_cudaMalloc(void **devPtr, size_t size) {
    if (resolve_cuda_symbols() != 0 || !real_cudaMalloc) {
        *devPtr = NULL;
        return cudaErrorUnknown;
    }

    return real_cudaMalloc(devPtr, size);
}

cudaError_t cudaFree(void *devPtr) {
    if (!devPtr) {
        return cudaSuccess;
    }

    if (in_hook) {
        /*
         * If recursion happens, avoid an infinite loop.
         * This should be rare; normally libcudart calls should go through real_*.
         */
        return cudaErrorUnknown;
    }

    in_hook = 1;

    if (resolve_cuda_symbols() != 0) {
        in_hook = 0;
        return cudaErrorUnknown;
    }

    if (worker_mapped_ptr && devPtr == worker_mapped_ptr) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaFree for worker IPC pointer %p -> cudaIpcCloseMemHandle\n",
                devPtr);

        cudaError_t res = real_cudaIpcCloseMemHandle(devPtr);
        fprintf(stderr,
                "[VRAM_HOOK] worker cudaIpcCloseMemHandle returned code=%d\n",
                (int)res);
        trace_cuda_free_caller("cudaFree(worker-ipc-close)", devPtr, res);

        worker_mapped_ptr = NULL;
        shared_allocation_done = false;

        in_hook = 0;
        return res;
    }

    if (is_master_process && master_alloc_ptr && devPtr == master_alloc_ptr) {
        fprintf(stderr,
                "[VRAM_HOOK] MASTER cudaFree for exported shared allocation %p, unlinking shm %s. "
                "Existing workers may keep IPC mapping until cudaIpcCloseMemHandle.\n",
                devPtr,
                shm_name);

        shm_unlink(shm_name);
        master_alloc_ptr = NULL;
        is_master_process = false;
        shared_allocation_done = false;

        if (suppress_master_free) {
            fprintf(stderr,
                    "[VRAM_HOOK] SUPPRESS_MASTER_FREE=1: not calling real_cudaFree for exported allocation %p\n",
                    devPtr);

            trace_cuda_free_caller("cudaFree(master-suppressed)", devPtr, cudaSuccess);

            in_hook = 0;
            return cudaSuccess;
        }

        cudaError_t res = real_cudaFree(devPtr);

        fprintf(stderr,
                "[VRAM_HOOK] MASTER real_cudaFree returned code=%d\n",
                (int)res);

        trace_cuda_free_caller("cudaFree(master-exported)", devPtr, res);

        in_hook = 0;
        return res;
    }

    cudaError_t res = real_cudaFree(devPtr);
    if (trace_normal_allocs) {
        trace_cuda_free_caller("cudaFree(normal)", devPtr, res);
    }

    in_hook = 0;
    return res;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!devPtr) {
        return cudaErrorInvalidValue;
    }

    *devPtr = NULL;

    if (in_hook) {
        return fallback_real_cudaMalloc(devPtr, size);
    }

    in_hook = 1;

    if (resolve_cuda_symbols() != 0) {
        in_hook = 0;
        return cudaErrorUnknown;
    }

    /*
     * Only one allocation per process should be replaced by IPC.
     * Everything else remains normal: KV cache, scratch buffers, CUDA graphs, etc.
     */
    if (shared_allocation_done || !size_matches_target(size)) {
        cudaError_t res = real_cudaMalloc(devPtr, size);

        if (res == cudaSuccess) {
            fprintf(stderr,
                    "[VRAM_HOOK] cudaMalloc normal: %.2f MB (%zu bytes) at %p\n",
                    (double)size / (1024.0 * 1024.0),
                    size,
                    *devPtr);
        } else {
            fprintf(stderr,
                    "[VRAM_HOOK] cudaMalloc normal failed: size=%zu code=%d\n",
                    size,
                    (int)res);
        }

        if (trace_normal_allocs) {
            trace_cuda_caller("cudaMalloc(normal)", size, *devPtr, res);
        }

        in_hook = 0;
        return res;
    }

    fprintf(stderr, "\n[VRAM_HOOK] ===================================================\n");
    fprintf(stderr,
            "[VRAM_HOOK] Target weights allocation intercepted: %.2f MB (%zu bytes)\n",
            (double)size / (1024.0 * 1024.0),
            size);

    for (;;) {
        int shm_fd = shm_open(shm_name, O_CREAT | O_EXCL | O_RDWR, 0666);

        if (shm_fd >= 0) {
            /*
             * MASTER role
             */
            fprintf(stderr, "[VRAM_HOOK] No active MASTER found. Assuming MASTER role.\n");

            if (ftruncate(shm_fd, sizeof(ipc_data_t)) != 0) {
                perror("[VRAM_HOOK] ftruncate");
                close(shm_fd);
                shm_unlink(shm_name);

                cudaError_t res = real_cudaMalloc(devPtr, size);
                in_hook = 0;
                return res;
            }

            ipc_data_t *data = map_ipc_data(shm_fd);
            if (!data) {
                close(shm_fd);
                shm_unlink(shm_name);

                cudaError_t res = real_cudaMalloc(devPtr, size);
                in_hook = 0;
                return res;
            }

            memset(data, 0, sizeof(*data));
            atomic_store_explicit(&data->is_ready, 0, memory_order_release);
            data->master_pid = getpid();
            data->allocation_size = size;

            cudaError_t res = real_cudaMalloc(devPtr, size);
            if (res != cudaSuccess) {
                fprintf(stderr,
                        "[VRAM_HOOK] MASTER cudaMalloc failed: code=%d\n",
                        (int)res);

                munmap(data, sizeof(ipc_data_t));
                close(shm_fd);
                shm_unlink(shm_name);

                in_hook = 0;
                return res;
            }

            cudaError_t ipc_res = real_cudaIpcGetMemHandle(&data->handle, *devPtr);
            if (ipc_res != cudaSuccess) {
                fprintf(stderr,
                        "[VRAM_HOOK] MASTER cudaIpcGetMemHandle failed: code=%d\n",
                        (int)ipc_res);

                real_cudaFree(*devPtr);
                *devPtr = NULL;

                munmap(data, sizeof(ipc_data_t));
                close(shm_fd);
                shm_unlink(shm_name);

                in_hook = 0;
                return ipc_res;
            }

            master_alloc_ptr = *devPtr;
            is_master_process = true;
            shared_allocation_done = true;

            atomic_store_explicit(&data->is_ready, 1, memory_order_release);

            munmap(data, sizeof(ipc_data_t));
            close(shm_fd);

            fprintf(stderr,
                    "[VRAM_HOOK] MASTER: allocated at %p and published through %s\n",
                    *devPtr,
                    shm_name);

            trace_cuda_caller("cudaMalloc(master-exported)", size, *devPtr, res);

            fprintf(stderr, "[VRAM_HOOK] ===================================================\n\n");

            in_hook = 0;
            return res;
        }

        if (errno != EEXIST) {
            /*
             * Fatal shm error. Fall back to normal allocation.
             */
            perror("[VRAM_HOOK] shm_open fatal error");

            cudaError_t res = real_cudaMalloc(devPtr, size);
            in_hook = 0;
            return res;
        }

        /*
         * WORKER role
         */
        fprintf(stderr,
                "[VRAM_HOOK] shm %s already exists. Trying WORKER path.\n",
                shm_name);

        shm_fd = shm_open(shm_name, O_RDWR, 0666);
        if (shm_fd < 0) {
            /*
             * Race: shm existed, then disappeared. Retry and maybe become master.
             */
            usleep(WAIT_USEC);
            continue;
        }

        int size_wait_res = wait_for_shm_size(shm_fd, sizeof(ipc_data_t));
        if (size_wait_res != 0) {
            close(shm_fd);

            if (size_wait_res == -2) {
                fprintf(stderr,
                        "[VRAM_HOOK] Removing stale shm %s and retrying MASTER election.\n",
                        shm_name);
                shm_unlink(shm_name);
            }

            usleep(WAIT_USEC);
            continue;
        }

        ipc_data_t *data = map_ipc_data(shm_fd);
        if (!data) {
            close(shm_fd);
            cudaError_t res = real_cudaMalloc(devPtr, size);
            in_hook = 0;
            return res;
        }

        pid_t master_pid = data->master_pid;

        if (process_is_dead(master_pid)) {
            fprintf(stderr,
                    "[VRAM_HOOK] WARNING: stale IPC found. MASTER pid=%d is dead. Taking over.\n",
                    master_pid);

            munmap(data, sizeof(ipc_data_t));
            close(shm_fd);
            shm_unlink(shm_name);
            continue;
        }

        fprintf(stderr,
                "[VRAM_HOOK] Found active MASTER pid=%d. Assuming WORKER role.\n",
                master_pid);

        int wait_sec = 0;

        while (atomic_load_explicit(&data->is_ready, memory_order_acquire) == 0) {
            if (wait_sec % 5 == 0) {
                fprintf(stderr,
                        "[VRAM_HOOK] Waiting for MASTER to finish loading weights to VRAM...\n");
            }

            sleep(1);
            wait_sec++;

            if (process_is_dead(master_pid)) {
                break;
            }
        }

        if (atomic_load_explicit(&data->is_ready, memory_order_acquire) == 0) {
            fprintf(stderr,
                    "[VRAM_HOOK] MASTER died before ready. Removing stale shm and retrying.\n");

            munmap(data, sizeof(ipc_data_t));
            close(shm_fd);
            shm_unlink(shm_name);
            continue;
        }

        if (data->allocation_size != size) {
            fprintf(stderr,
                    "[VRAM_HOOK] ERROR: shm allocation size mismatch: shm=%zu requested=%zu. "
                    "Falling back to normal cudaMalloc.\n",
                    data->allocation_size,
                    size);

            munmap(data, sizeof(ipc_data_t));
            close(shm_fd);

            cudaError_t res = real_cudaMalloc(devPtr, size);
            in_hook = 0;
            return res;
        }

        cudaError_t res = real_cudaIpcOpenMemHandle(devPtr,
                                                    data->handle,
                                                    cudaIpcMemLazyEnablePeerAccess);

        if (res == cudaSuccess) {
            worker_mapped_ptr = *devPtr;
            shared_allocation_done = true;

            fprintf(stderr,
                    "[VRAM_HOOK] WORKER: IPC handle mapped at %p. No duplicate weights allocation.\n",
                    *devPtr);

            trace_cuda_caller("cudaMalloc(worker-ipc-open)", size, *devPtr, res);
        } else {
            fprintf(stderr,
                    "[VRAM_HOOK] ERROR: cudaIpcOpenMemHandle failed: code=%d. "
                    "Falling back to normal cudaMalloc.\n",
                    (int)res);

            /*
             * If IPC open fails, do not crash llama.cpp. Allocate normally.
             * This can happen with stale handles, incompatible devices, or permission issues.
             */
            cudaError_t fallback_res = real_cudaMalloc(devPtr, size);
            if (fallback_res == cudaSuccess) {
                shared_allocation_done = true;
                fprintf(stderr,
                        "[VRAM_HOOK] fallback cudaMalloc for weights succeeded at %p\n",
                        *devPtr);
                trace_cuda_caller("cudaMalloc(weights-fallback)", size, *devPtr, fallback_res);
                res = fallback_res;
            }
        }

        munmap(data, sizeof(ipc_data_t));
        close(shm_fd);

        fprintf(stderr, "[VRAM_HOOK] ===================================================\n\n");

        in_hook = 0;
        return res;
    }
}

__attribute__((destructor))
static void cleanup_hook(void) {
    if (is_master_process) {
        fprintf(stderr, "[VRAM_HOOK] destructor: unlinking master shm %s\n", shm_name);
        shm_unlink(shm_name);
    }

    /*
     * Do not call cudaIpcCloseMemHandle here blindly:
     * llama.cpp/CUDA runtime shutdown ordering can make destructor-time CUDA calls unsafe.
     * Normal cudaFree path above closes worker IPC handles.
     */
}

