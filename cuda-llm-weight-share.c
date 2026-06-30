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
#include <stdint.h>
#include <time.h>

#include <cuda_runtime.h>

/*
 * CUDA VRAM IPC hook for sharing one selected cudaMalloc allocation
 * across multiple independent processes.
 *
 * Clean runtime-only version:
 * - hooks cudaMalloc/cudaFree
 * - hooks runtime cudaMemcpy/cudaMemcpyAsync and cudaMemset/cudaMemsetAsync
 * - hooks cudaDeviceSynchronize/cudaStreamSynchronize/cudaEventSynchronize with timing
 * - does not include cuda.h
 * - does not hook CUDA Driver API cuMemcpy/cuMemset
 *
 * Build:
 *   gcc -shared -fPIC -O2 \
 *     -I/usr/local/cuda/include \
 *     cuda-llm-weight-share-clean.c \
 *     -o cuda-llm-weight-share-clean.so \
 *     -ldl
 *
 * Usage example:
 *   export MODEL_SIZE=49392123904
 *   export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_qwen3_30b_gpu0"
 *   LD_PRELOAD=/path/cuda-llm-weight-share-clean.so ./llama-server ...
 *
 * Optional:
 *   export MODEL_SIZE_TOLERANCE=1048576
 *   export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=10
 *   export CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=1
 *   export CUDA_VRAM_IPC_SKIP_WORKER_WEIGHT_UPLOADS=1
 *   export CUDA_VRAM_IPC_LOG_SKIPPED_UPLOADS=0
 *   export CUDA_VRAM_IPC_LOG_MEMCPY_OPS=0
 *   export CUDA_VRAM_IPC_LOG_WEIGHT_COPY_SUMMARY=1
 *   export CUDA_VRAM_IPC_LOG_SYNC_OPS=0
 *   export CUDA_VRAM_IPC_LOG_SYNC_MIN_MS=1
 *   export CUDA_VRAM_IPC_SUMMARY_ON_SYNC=0
 *   export CUDA_VRAM_IPC_SUMMARY_DEACTIVATE_ON_NEXT_MALLOC=1
 *   export CUDA_VRAM_IPC_LOG_EVENT_GAPS=1
 *   export CUDA_VRAM_IPC_LOG_EVENT_GAP_THRESHOLD_MS=1000
 *   export CUDA_VRAM_IPC_TRACE_CALLERS=0
 *   export CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS=0
 *   export CUDA_VRAM_IPC_TRACE_DEPTH=8
 *
 * Notes:
 * - It resolves libcudart symbols lazily, which is important when llama.cpp
 *   is built with GGML_BACKEND_DL=ON and CUDA backend appears later via dlopen().
 * - The master process owns the real cudaMalloc allocation.
 * - Worker processes map it using cudaIpcOpenMemHandle().
 * - Worker weight uploads through runtime cudaMemcpy/cudaMemset can be skipped.
 * - Driver API upload paths are intentionally not intercepted in this clean build.
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
static cudaError_t (*real_cudaMemcpy)(void *, const void *, size_t, enum cudaMemcpyKind) = NULL;
static cudaError_t (*real_cudaMemcpyAsync)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t) = NULL;
static cudaError_t (*real_cudaMemset)(void *, int, size_t) = NULL;
static cudaError_t (*real_cudaMemsetAsync)(void *, int, size_t, cudaStream_t) = NULL;
static cudaError_t (*real_cudaDeviceSynchronize)(void) = NULL;
static cudaError_t (*real_cudaStreamSynchronize)(cudaStream_t) = NULL;
static cudaError_t (*real_cudaEventSynchronize)(cudaEvent_t) = NULL;

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
static bool skip_worker_weight_uploads = true;
static bool log_memcpy_ops = false;
static bool log_skipped_weight_uploads = false;
static bool log_weight_copy_summary = true;
static bool log_sync_ops = false;
static bool summary_on_sync = false;
static bool summary_deactivate_on_next_malloc = true;
static long long log_sync_min_ms = 1;
static int trace_depth = 8;

/* Aggregated copy/memset statistics for shared weights range */
static unsigned long long master_weight_copy_ops = 0;
static unsigned long long master_weight_copy_bytes = 0;
static unsigned long long master_weight_copy_api_elapsed_ms = 0;
static unsigned long long master_weight_sync_ops = 0;
static unsigned long long master_weight_sync_elapsed_ms = 0;
static long long master_weight_upload_phase_start_ms = 0;
static long long master_weight_upload_phase_end_ms = 0;
static bool master_weight_upload_phase_active = false;
static unsigned long long worker_skipped_weight_ops = 0;
static unsigned long long worker_skipped_weight_bytes = 0;
static unsigned long long last_printed_master_weight_copy_ops = 0;
static unsigned long long last_printed_worker_skipped_weight_ops = 0;

/* Wall-clock gap logging: useful when time is spent outside CUDA APIs, e.g. disk IO / mmap page faults / CPU parsing. */
static bool log_event_gaps = true;
static long long last_hook_event_ms = 0;
static long long log_event_gap_threshold_ms = 1000;

/* Per-process state */
static bool shared_allocation_done = false;
static bool is_master_process = false;
static void *master_alloc_ptr = NULL;
static size_t master_alloc_size = 0;
static void *worker_mapped_ptr = NULL;
static size_t worker_mapped_size = 0;

/* Avoid accidental resolver recursion */
static __thread int in_hook = 0;

static long long now_ms(void);
static void note_hook_event(const char *event_name);

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

    const char *env_skip_uploads = getenv("CUDA_VRAM_IPC_SKIP_WORKER_WEIGHT_UPLOADS");
    if (env_skip_uploads && env_skip_uploads[0]) {
        skip_worker_weight_uploads = env_is_true(env_skip_uploads);
    }

    log_memcpy_ops = env_is_true(getenv("CUDA_VRAM_IPC_LOG_MEMCPY_OPS"));
    log_skipped_weight_uploads = env_is_true(getenv("CUDA_VRAM_IPC_LOG_SKIPPED_UPLOADS"));

    const char *env_summary = getenv("CUDA_VRAM_IPC_LOG_WEIGHT_COPY_SUMMARY");
    if (env_summary && env_summary[0]) {
        log_weight_copy_summary = env_is_true(env_summary);
    }

    const char *env_sync = getenv("CUDA_VRAM_IPC_LOG_SYNC_OPS");
    if (env_sync && env_sync[0]) {
        log_sync_ops = env_is_true(env_sync);
    }

    const char *env_sync_min_ms = getenv("CUDA_VRAM_IPC_LOG_SYNC_MIN_MS");
    if (env_sync_min_ms && env_sync_min_ms[0]) {
        errno = 0;
        long parsed = strtol(env_sync_min_ms, NULL, 10);
        if (errno == 0 && parsed >= 0 && parsed < 3600000) {
            log_sync_min_ms = parsed;
        }
    }

    const char *env_summary_on_sync = getenv("CUDA_VRAM_IPC_SUMMARY_ON_SYNC");
    if (env_summary_on_sync && env_summary_on_sync[0]) {
        summary_on_sync = env_is_true(env_summary_on_sync);
    }

    const char *env_summary_deactivate = getenv("CUDA_VRAM_IPC_SUMMARY_DEACTIVATE_ON_NEXT_MALLOC");
    if (env_summary_deactivate && env_summary_deactivate[0]) {
        summary_deactivate_on_next_malloc = env_is_true(env_summary_deactivate);
    }

    const char *env_gaps = getenv("CUDA_VRAM_IPC_LOG_EVENT_GAPS");
    if (env_gaps && env_gaps[0]) {
        log_event_gaps = env_is_true(env_gaps);
    }

    const char *env_gap_ms = getenv("CUDA_VRAM_IPC_LOG_EVENT_GAP_THRESHOLD_MS");
    if (env_gap_ms && env_gap_ms[0]) {
        errno = 0;
        long parsed = strtol(env_gap_ms, NULL, 10);
        if (errno == 0 && parsed >= 0 && parsed < 3600000) {
            log_event_gap_threshold_ms = parsed;
        }
    }

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
            "TRACE_CALLERS=%d, TRACE_NORMAL_ALLOCS=%d, TRACE_DEPTH=%d, "
            "SKIP_WORKER_WEIGHT_UPLOADS=%d, LOG_MEMCPY_OPS=%d, LOG_SKIPPED_UPLOADS=%d, "
            "LOG_WEIGHT_COPY_SUMMARY=%d, LOG_SYNC_OPS=%d, LOG_SYNC_MIN_MS=%lld, "
            "SUMMARY_ON_SYNC=%d, SUMMARY_DEACTIVATE_ON_NEXT_MALLOC=%d, "
            "LOG_EVENT_GAPS=%d, GAP_THRESHOLD_MS=%lld\n",
            target_shared_size,
            target_size_tolerance,
            shm_name,
            shm_size_wait_sec,
            suppress_master_free ? 1 : 0,
            trace_callers ? 1 : 0,
            trace_normal_allocs ? 1 : 0,
            trace_depth,
            skip_worker_weight_uploads ? 1 : 0,
            log_memcpy_ops ? 1 : 0,
            log_skipped_weight_uploads ? 1 : 0,
            log_weight_copy_summary ? 1 : 0,
            log_sync_ops ? 1 : 0,
            log_sync_min_ms,
            summary_on_sync ? 1 : 0,
            summary_deactivate_on_next_malloc ? 1 : 0,
            log_event_gaps ? 1 : 0,
            log_event_gap_threshold_ms);

    last_hook_event_ms = now_ms();
}

static void print_symbol_line(const char *prefix, void *addr) {
    Dl_info info;
    memset(&info, 0, sizeof(info));

    if (dladdr(addr, &info) && info.dli_fname) {
        const char *sym = info.dli_sname ? info.dli_sname : "?";
        uintptr_t a = (uintptr_t)addr;
        uintptr_t base = (uintptr_t)info.dli_fbase;
        unsigned long offset = 0;

        if (base && a >= base) {
            offset = (unsigned long)(a - base);
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

static long long now_ms(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return 0;
    }

    return (long long)ts.tv_sec * 1000LL + ts.tv_nsec / 1000000LL;
}

static void note_hook_event(const char *event_name) {
    long long now = now_ms();

    if (last_hook_event_ms != 0) {
        long long delta = now - last_hook_event_ms;

        if (log_event_gaps && delta >= log_event_gap_threshold_ms) {
            fprintf(stderr,
                    "[VRAM_HOOK] GAP before %s: %lld ms since previous VRAM_HOOK event\n",
                    event_name,
                    delta);
        }
    }

    last_hook_event_ms = now;
}

static bool ptr_range_in_range(const void *ptr, size_t count, const void *base_ptr, size_t range_size) {
    if (!ptr || !base_ptr || range_size == 0) {
        return false;
    }

    uintptr_t p = (uintptr_t)ptr;
    uintptr_t base = (uintptr_t)base_ptr;
    uintptr_t end = base + range_size;

    if (end < base) {
        return false;
    }

    if (p < base || p >= end) {
        return false;
    }

    if (count == 0) {
        return true;
    }

    uintptr_t p_end = p + count;
    if (p_end < p) {
        return false;
    }

    return p_end <= end;
}

static bool ptr_in_worker_shared_range(const void *ptr, size_t count) {
    return ptr_range_in_range(ptr, count, worker_mapped_ptr, worker_mapped_size);
}

static bool ptr_in_master_shared_range(const void *ptr, size_t count) {
    return ptr_range_in_range(ptr, count, master_alloc_ptr, master_alloc_size);
}

static const char *cuda_memcpy_kind_name(enum cudaMemcpyKind kind) {
    switch (kind) {
        case cudaMemcpyHostToHost: return "HostToHost";
        case cudaMemcpyHostToDevice: return "HostToDevice";
        case cudaMemcpyDeviceToHost: return "DeviceToHost";
        case cudaMemcpyDeviceToDevice: return "DeviceToDevice";
        case cudaMemcpyDefault: return "Default";
        default: return "Unknown";
    }
}

static void note_master_weight_copy(size_t count, long long api_elapsed_ms) {
    master_weight_copy_ops++;
    master_weight_copy_bytes += (unsigned long long)count;

    if (api_elapsed_ms > 0) {
        master_weight_copy_api_elapsed_ms += (unsigned long long)api_elapsed_ms;
    }
}

static void note_master_weight_sync_elapsed(long long elapsed_ms) {
    if (!is_master_process || !master_alloc_ptr || !master_weight_upload_phase_active) {
        return;
    }

    master_weight_sync_ops++;

    if (elapsed_ms > 0) {
        master_weight_sync_elapsed_ms += (unsigned long long)elapsed_ms;
    }
}

static void note_worker_skipped_weight_upload(size_t count) {
    worker_skipped_weight_ops++;
    worker_skipped_weight_bytes += (unsigned long long)count;
}

static void print_weight_copy_summary(const char *reason) {
    if (!log_weight_copy_summary) {
        return;
    }

    if (master_weight_copy_ops != last_printed_master_weight_copy_ops) {
        long long now = now_ms();
        long long wall_elapsed_ms = 0;

        if (master_weight_upload_phase_start_ms > 0) {
            long long phase_end = master_weight_upload_phase_end_ms > 0
                                      ? master_weight_upload_phase_end_ms
                                      : now;
            wall_elapsed_ms = phase_end - master_weight_upload_phase_start_ms;
            if (wall_elapsed_ms < 0) {
                wall_elapsed_ms = 0;
            }
        }

        fprintf(stderr,
                "[VRAM_HOOK] SUMMARY[%s]: MASTER shared weights writes/copies so far: "
                "%llu ops, %.2f MB (%llu bytes), copy_api_elapsed=%llu ms, "
                "sync_elapsed=%llu ms (%llu sync ops), wall_elapsed=%lld ms\n",
                reason,
                master_weight_copy_ops,
                (double)master_weight_copy_bytes / (1024.0 * 1024.0),
                master_weight_copy_bytes,
                master_weight_copy_api_elapsed_ms,
                master_weight_sync_elapsed_ms,
                master_weight_sync_ops,
                wall_elapsed_ms);
        last_printed_master_weight_copy_ops = master_weight_copy_ops;
    }

    if (summary_deactivate_on_next_malloc &&
        strcmp(reason, "before-next-cudaMalloc") == 0 &&
        master_weight_upload_phase_active &&
        master_alloc_size > 0 &&
        master_weight_copy_bytes >= (unsigned long long)master_alloc_size) {
        master_weight_upload_phase_active = false;
        master_weight_upload_phase_end_ms = now_ms();
    }

    if (worker_skipped_weight_ops != last_printed_worker_skipped_weight_ops) {
        fprintf(stderr,
                "[VRAM_HOOK] SUMMARY[%s]: WORKER skipped shared weights uploads so far: "
                "%llu ops, %.2f MB (%llu bytes)\n",
                reason,
                worker_skipped_weight_ops,
                (double)worker_skipped_weight_bytes / (1024.0 * 1024.0),
                worker_skipped_weight_bytes);
        last_printed_worker_skipped_weight_ops = worker_skipped_weight_ops;
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

    if (hi < target_shared_size) {
        hi = (size_t)-1;
    }

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
    if (!real_cudaMemcpy) {
        real_cudaMemcpy =
            (cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind))dlsym(handle, "cudaMemcpy");
    }
    if (!real_cudaMemcpyAsync) {
        real_cudaMemcpyAsync =
            (cudaError_t (*)(void *, const void *, size_t, enum cudaMemcpyKind, cudaStream_t))dlsym(handle, "cudaMemcpyAsync");
    }
    if (!real_cudaMemset) {
        real_cudaMemset =
            (cudaError_t (*)(void *, int, size_t))dlsym(handle, "cudaMemset");
    }
    if (!real_cudaMemsetAsync) {
        real_cudaMemsetAsync =
            (cudaError_t (*)(void *, int, size_t, cudaStream_t))dlsym(handle, "cudaMemsetAsync");
    }
    if (!real_cudaDeviceSynchronize) {
        real_cudaDeviceSynchronize =
            (cudaError_t (*)(void))dlsym(handle, "cudaDeviceSynchronize");
    }
    if (!real_cudaStreamSynchronize) {
        real_cudaStreamSynchronize =
            (cudaError_t (*)(cudaStream_t))dlsym(handle, "cudaStreamSynchronize");
    }
    if (!real_cudaEventSynchronize) {
        real_cudaEventSynchronize =
            (cudaError_t (*)(cudaEvent_t))dlsym(handle, "cudaEventSynchronize");
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
                "cudaIpcOpenMemHandle=%p cudaIpcCloseMemHandle=%p "
                "cudaMemcpy=%p cudaMemcpyAsync=%p cudaMemset=%p cudaMemsetAsync=%p "
                "cudaDeviceSynchronize=%p cudaStreamSynchronize=%p cudaEventSynchronize=%p\n",
                (void *)real_cudaMalloc,
                (void *)real_cudaFree,
                (void *)real_cudaIpcGetMemHandle,
                (void *)real_cudaIpcOpenMemHandle,
                (void *)real_cudaIpcCloseMemHandle,
                (void *)real_cudaMemcpy,
                (void *)real_cudaMemcpyAsync,
                (void *)real_cudaMemset,
                (void *)real_cudaMemsetAsync,
                (void *)real_cudaDeviceSynchronize,
                (void *)real_cudaStreamSynchronize,
                (void *)real_cudaEventSynchronize);
        return -1;
    }

    fprintf(stderr, "[VRAM_HOOK] CUDA runtime symbols resolved via libcudart handle\n");
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
    note_hook_event("cudaFree");

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

        long long t0 = now_ms();
        cudaError_t res = real_cudaIpcCloseMemHandle(devPtr);
        long long t1 = now_ms();
        last_hook_event_ms = t1;

        fprintf(stderr,
                "[VRAM_HOOK] WORKER cudaIpcCloseMemHandle done for shared weights ptr=%p code=%d elapsed=%lld ms\n",
                devPtr,
                (int)res,
                t1 - t0);
        trace_cuda_free_caller("cudaFree(worker-ipc-close)", devPtr, res);

        worker_mapped_ptr = NULL;
        worker_mapped_size = 0;
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
        master_alloc_size = 0;
        is_master_process = false;
        shared_allocation_done = false;
        master_weight_upload_phase_active = false;
        if (master_weight_upload_phase_end_ms == 0) {
            master_weight_upload_phase_end_ms = now_ms();
        }

        if (suppress_master_free) {
            fprintf(stderr,
                    "[VRAM_HOOK] SUPPRESS_MASTER_FREE=1: not calling real_cudaFree for exported allocation %p\n",
                    devPtr);

            trace_cuda_free_caller("cudaFree(master-suppressed)", devPtr, cudaSuccess);

            in_hook = 0;
            return cudaSuccess;
        }

        long long t0 = now_ms();
        cudaError_t res = real_cudaFree(devPtr);
        long long t1 = now_ms();
        last_hook_event_ms = t1;

        fprintf(stderr,
                "[VRAM_HOOK] MASTER real_cudaFree done for shared weights ptr=%p code=%d elapsed=%lld ms\n",
                devPtr,
                (int)res,
                t1 - t0);

        trace_cuda_free_caller("cudaFree(master-exported)", devPtr, res);

        in_hook = 0;
        return res;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaFree(devPtr);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    if (trace_normal_allocs) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaFree normal ptr=%p code=%d elapsed=%lld ms\n",
                devPtr,
                (int)res,
                t1 - t0);
        trace_cuda_free_caller("cudaFree(normal)", devPtr, res);
    }

    in_hook = 0;
    return res;
}

cudaError_t cudaMalloc(void **devPtr, size_t size) {
    note_hook_event("cudaMalloc");

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
        print_weight_copy_summary("before-next-cudaMalloc");

        fprintf(stderr,
                "[VRAM_HOOK] cudaMalloc normal request: %.2f MB (%zu bytes)\n",
                (double)size / (1024.0 * 1024.0),
                size);

        long long t0 = now_ms();
        cudaError_t res = real_cudaMalloc(devPtr, size);
        long long t1 = now_ms();
        last_hook_event_ms = t1;

        if (res == cudaSuccess) {
            fprintf(stderr,
                    "[VRAM_HOOK] cudaMalloc normal done: %.2f MB (%zu bytes) at %p, elapsed=%lld ms\n",
                    (double)size / (1024.0 * 1024.0),
                    size,
                    *devPtr,
                    t1 - t0);
        } else {
            fprintf(stderr,
                    "[VRAM_HOOK] cudaMalloc normal failed: size=%zu code=%d elapsed=%lld ms\n",
                    size,
                    (int)res,
                    t1 - t0);
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

            fprintf(stderr,
                    "[VRAM_HOOK] MASTER cudaMalloc request for shared weights: %.2f MB (%zu bytes)\n",
                    (double)size / (1024.0 * 1024.0),
                    size);

            long long master_malloc_t0 = now_ms();
            cudaError_t res = real_cudaMalloc(devPtr, size);
            long long master_malloc_t1 = now_ms();
            last_hook_event_ms = master_malloc_t1;

            if (res != cudaSuccess) {
                fprintf(stderr,
                        "[VRAM_HOOK] MASTER cudaMalloc failed for shared weights: code=%d elapsed=%lld ms\n",
                        (int)res,
                        master_malloc_t1 - master_malloc_t0);

                munmap(data, sizeof(ipc_data_t));
                close(shm_fd);
                shm_unlink(shm_name);

                in_hook = 0;
                return res;
            }

            fprintf(stderr,
                    "[VRAM_HOOK] MASTER cudaMalloc done for shared weights: %.2f MB (%zu bytes) at %p, elapsed=%lld ms\n",
                    (double)size / (1024.0 * 1024.0),
                    size,
                    *devPtr,
                    master_malloc_t1 - master_malloc_t0);

            long long get_handle_t0 = now_ms();
            cudaError_t ipc_res = real_cudaIpcGetMemHandle(&data->handle, *devPtr);
            long long get_handle_t1 = now_ms();
            last_hook_event_ms = get_handle_t1;

            if (ipc_res != cudaSuccess) {
                fprintf(stderr,
                        "[VRAM_HOOK] MASTER cudaIpcGetMemHandle failed: code=%d elapsed=%lld ms\n",
                        (int)ipc_res,
                        get_handle_t1 - get_handle_t0);

                real_cudaFree(*devPtr);
                *devPtr = NULL;

                munmap(data, sizeof(ipc_data_t));
                close(shm_fd);
                shm_unlink(shm_name);

                in_hook = 0;
                return ipc_res;
            }

            fprintf(stderr,
                    "[VRAM_HOOK] MASTER cudaIpcGetMemHandle done for shared weights: code=%d elapsed=%lld ms\n",
                    (int)ipc_res,
                    get_handle_t1 - get_handle_t0);

            master_alloc_ptr = *devPtr;
            master_alloc_size = size;
            is_master_process = true;
            shared_allocation_done = true;
            master_weight_upload_phase_active = true;
            master_weight_upload_phase_start_ms = now_ms();
            master_weight_upload_phase_end_ms = 0;

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

        fprintf(stderr,
                "[VRAM_HOOK] WORKER cudaIpcOpenMemHandle request for shared weights: %.2f MB (%zu bytes)\n",
                (double)size / (1024.0 * 1024.0),
                size);

        long long ipc_open_t0 = now_ms();
        cudaError_t res = real_cudaIpcOpenMemHandle(devPtr,
                                                    data->handle,
                                                    cudaIpcMemLazyEnablePeerAccess);
        long long ipc_open_t1 = now_ms();
        last_hook_event_ms = ipc_open_t1;

        if (res == cudaSuccess) {
            worker_mapped_ptr = *devPtr;
            worker_mapped_size = size;
            shared_allocation_done = true;

            fprintf(stderr,
                    "[VRAM_HOOK] WORKER cudaIpcOpenMemHandle done for shared weights: %.2f MB (%zu bytes) at %p, code=%d elapsed=%lld ms. No duplicate weights allocation.\n",
                    (double)size / (1024.0 * 1024.0),
                    size,
                    *devPtr,
                    (int)res,
                    ipc_open_t1 - ipc_open_t0);

            trace_cuda_caller("cudaMalloc(worker-ipc-open)", size, *devPtr, res);
        } else {
            fprintf(stderr,
                    "[VRAM_HOOK] ERROR: cudaIpcOpenMemHandle failed for shared weights: code=%d elapsed=%lld ms. "
                    "Falling back to normal cudaMalloc.\n",
                    (int)res,
                    ipc_open_t1 - ipc_open_t0);

            /*
             * If IPC open fails, do not crash llama.cpp. Allocate normally.
             * This can happen with stale handles, incompatible devices, or permission issues.
             */
            fprintf(stderr,
                    "[VRAM_HOOK] fallback cudaMalloc request for shared weights: %.2f MB (%zu bytes)\n",
                    (double)size / (1024.0 * 1024.0),
                    size);

            long long fallback_t0 = now_ms();
            cudaError_t fallback_res = real_cudaMalloc(devPtr, size);
            long long fallback_t1 = now_ms();
            last_hook_event_ms = fallback_t1;

            if (fallback_res == cudaSuccess) {
                shared_allocation_done = true;
                fprintf(stderr,
                        "[VRAM_HOOK] fallback cudaMalloc done for weights: %.2f MB (%zu bytes) at %p, elapsed=%lld ms\n",
                        (double)size / (1024.0 * 1024.0),
                        size,
                        *devPtr,
                        fallback_t1 - fallback_t0);
                trace_cuda_caller("cudaMalloc(weights-fallback)", size, *devPtr, fallback_res);
                res = fallback_res;
            } else {
                fprintf(stderr,
                        "[VRAM_HOOK] fallback cudaMalloc failed for weights: code=%d elapsed=%lld ms\n",
                        (int)fallback_res,
                        fallback_t1 - fallback_t0);
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

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, enum cudaMemcpyKind kind) {
    note_hook_event("cudaMemcpy");
    if (resolve_cuda_symbols() != 0 || !real_cudaMemcpy) {
        return cudaErrorUnknown;
    }

    if (skip_worker_weight_uploads && ptr_in_worker_shared_range(dst, count)) {
        note_worker_skipped_weight_upload(count);
        if (log_skipped_weight_uploads) {
            fprintf(stderr,
                    "[VRAM_HOOK] WORKER: skipping cudaMemcpy to shared weights dst=%p size=%zu kind=%s(%d)\n",
                    dst,
                    count,
                    cuda_memcpy_kind_name(kind),
                    (int)kind);
        }
        return cudaSuccess;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaMemcpy(dst, src, count, kind);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    if (res == cudaSuccess && ptr_in_master_shared_range(dst, count)) {
        note_master_weight_copy(count, t1 - t0);
    }

    if (log_memcpy_ops || trace_normal_allocs) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaMemcpy dst=%p src=%p size=%zu kind=%s(%d) code=%d elapsed=%lld ms\n",
                dst,
                src,
                count,
                cuda_memcpy_kind_name(kind),
                (int)kind,
                (int)res,
                t1 - t0);
    }

    return res;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count,
                            enum cudaMemcpyKind kind, cudaStream_t stream) {
    note_hook_event("cudaMemcpyAsync");
    if (resolve_cuda_symbols() != 0 || !real_cudaMemcpyAsync) {
        return cudaErrorUnknown;
    }

    if (skip_worker_weight_uploads && ptr_in_worker_shared_range(dst, count)) {
        note_worker_skipped_weight_upload(count);
        if (log_skipped_weight_uploads) {
            fprintf(stderr,
                    "[VRAM_HOOK] WORKER: skipping cudaMemcpyAsync to shared weights dst=%p size=%zu kind=%s(%d) stream=%p\n",
                    dst,
                    count,
                    cuda_memcpy_kind_name(kind),
                    (int)kind,
                    (void *)stream);
        }
        return cudaSuccess;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaMemcpyAsync(dst, src, count, kind, stream);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    if (res == cudaSuccess && ptr_in_master_shared_range(dst, count)) {
        note_master_weight_copy(count, t1 - t0);
    }

    if (log_memcpy_ops || trace_normal_allocs) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaMemcpyAsync dst=%p src=%p size=%zu kind=%s(%d) stream=%p code=%d elapsed=%lld ms\n",
                dst,
                src,
                count,
                cuda_memcpy_kind_name(kind),
                (int)kind,
                (void *)stream,
                (int)res,
                t1 - t0);
    }

    return res;
}

cudaError_t cudaMemset(void *devPtr, int value, size_t count) {
    note_hook_event("cudaMemset");
    if (resolve_cuda_symbols() != 0 || !real_cudaMemset) {
        return cudaErrorUnknown;
    }

    if (skip_worker_weight_uploads && ptr_in_worker_shared_range(devPtr, count)) {
        note_worker_skipped_weight_upload(count);
        if (log_skipped_weight_uploads) {
            fprintf(stderr,
                    "[VRAM_HOOK] WORKER: skipping cudaMemset on shared weights ptr=%p size=%zu value=%d\n",
                    devPtr,
                    count,
                    value);
        }
        return cudaSuccess;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaMemset(devPtr, value, count);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    if (res == cudaSuccess && ptr_in_master_shared_range(devPtr, count)) {
        note_master_weight_copy(count, t1 - t0);
    }

    if (log_memcpy_ops || trace_normal_allocs) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaMemset ptr=%p size=%zu value=%d code=%d elapsed=%lld ms\n",
                devPtr,
                count,
                value,
                (int)res,
                t1 - t0);
    }

    return res;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream) {
    note_hook_event("cudaMemsetAsync");
    if (resolve_cuda_symbols() != 0 || !real_cudaMemsetAsync) {
        return cudaErrorUnknown;
    }

    if (skip_worker_weight_uploads && ptr_in_worker_shared_range(devPtr, count)) {
        note_worker_skipped_weight_upload(count);
        if (log_skipped_weight_uploads) {
            fprintf(stderr,
                    "[VRAM_HOOK] WORKER: skipping cudaMemsetAsync on shared weights ptr=%p size=%zu value=%d stream=%p\n",
                    devPtr,
                    count,
                    value,
                    (void *)stream);
        }
        return cudaSuccess;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaMemsetAsync(devPtr, value, count, stream);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    if (res == cudaSuccess && ptr_in_master_shared_range(devPtr, count)) {
        note_master_weight_copy(count, t1 - t0);
    }

    if (log_memcpy_ops || trace_normal_allocs) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaMemsetAsync ptr=%p size=%zu value=%d stream=%p code=%d elapsed=%lld ms\n",
                devPtr,
                count,
                value,
                (void *)stream,
                (int)res,
                t1 - t0);
    }

    return res;
}

cudaError_t cudaDeviceSynchronize(void) {
    note_hook_event("cudaDeviceSynchronize");
    if (resolve_cuda_symbols() != 0 || !real_cudaDeviceSynchronize) {
        return cudaErrorUnknown;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaDeviceSynchronize();
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    long long elapsed_ms = t1 - t0;
    note_master_weight_sync_elapsed(elapsed_ms);
    if (log_sync_ops && elapsed_ms >= log_sync_min_ms) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaDeviceSynchronize code=%d elapsed=%lld ms\n",
                (int)res,
                elapsed_ms);
    }

    if (summary_on_sync) {
        print_weight_copy_summary("after-cudaDeviceSynchronize");
    }
    return res;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream) {
    note_hook_event("cudaStreamSynchronize");
    if (resolve_cuda_symbols() != 0 || !real_cudaStreamSynchronize) {
        return cudaErrorUnknown;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaStreamSynchronize(stream);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    long long elapsed_ms = t1 - t0;
    note_master_weight_sync_elapsed(elapsed_ms);
    if (log_sync_ops && elapsed_ms >= log_sync_min_ms) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaStreamSynchronize stream=%p code=%d elapsed=%lld ms\n",
                (void *)stream,
                (int)res,
                elapsed_ms);
    }

    if (summary_on_sync) {
        print_weight_copy_summary("after-cudaStreamSynchronize");
    }
    return res;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event) {
    note_hook_event("cudaEventSynchronize");
    if (resolve_cuda_symbols() != 0 || !real_cudaEventSynchronize) {
        return cudaErrorUnknown;
    }

    long long t0 = now_ms();
    cudaError_t res = real_cudaEventSynchronize(event);
    long long t1 = now_ms();
    last_hook_event_ms = t1;

    long long elapsed_ms = t1 - t0;
    note_master_weight_sync_elapsed(elapsed_ms);
    if (log_sync_ops && elapsed_ms >= log_sync_min_ms) {
        fprintf(stderr,
                "[VRAM_HOOK] cudaEventSynchronize event=%p code=%d elapsed=%lld ms\n",
                (void *)event,
                (int)res,
                elapsed_ms);
    }

    if (summary_on_sync) {
        print_weight_copy_summary("after-cudaEventSynchronize");
    }
    return res;
}

__attribute__((destructor))
static void cleanup_hook(void) {
    print_weight_copy_summary("destructor");

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

