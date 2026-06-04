#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdbool.h>
#include <errno.h>
#include <signal.h>

// Universal shared memory name
#define SHM_NAME "/cuda_vram_ipc_auto"

// Structure for inter-process communication
typedef struct {
    volatile int is_ready;     // Flag indicating weights are fully loaded
    pid_t master_pid;          // PID of the master process
    cudaIpcMemHandle_t handle; // The IPC memory handle itself
} ipc_data_t;

// Pointers to the original CUDA functions
static cudaError_t (*real_cudaMalloc)(void**, size_t) = NULL;
static cudaError_t (*real_cudaFree)(void*) = NULL;

// Global variables for configuration and state tracking
static size_t target_shared_size = 0;
static void* worker_mapped_ptr = NULL; // Keeps track of the IPC mapped pointer

// Initialization routine executed when the library is loaded
__attribute__((constructor)) static void setup_hooks() {
    if (!real_cudaMalloc) {
        real_cudaMalloc = dlsym(RTLD_NEXT, "cudaMalloc");
    }
    if (!real_cudaFree) {
        real_cudaFree = dlsym(RTLD_NEXT, "cudaFree");
    }
    
    // Read the target weights size from the environment variable
    const char* env_size = getenv("MODEL_SIZE");
    if (env_size) {
        target_shared_size = (size_t)strtoull(env_size, NULL, 10);
    }
}

// ========================================================================
// Intercepting cudaFree to prevent crash on exit
// ========================================================================
cudaError_t cudaFree(void* devPtr) {
    if (!real_cudaFree) setup_hooks();

    // If this pointer is the one we mapped via IPC, close the handle properly!
    if (worker_mapped_ptr && devPtr == worker_mapped_ptr) {
        fprintf(stderr, "[VRAM_HOOK] Intercepted cudaFree for IPC weights at %p. Closing handle safely.\n", devPtr);
        cudaError_t res = cudaIpcCloseMemHandle(devPtr);
        worker_mapped_ptr = NULL; // Reset
        return res;
    }

    // For all other regular allocations (like KV Cache), use normal cudaFree
    return real_cudaFree(devPtr);
}

// ========================================================================
// Intercepting cudaMalloc for sharing weights
// ========================================================================
cudaError_t cudaMalloc(void **devPtr, size_t size) {
    if (!real_cudaMalloc) setup_hooks();

    // If the requested size doesn't match the model weights, or if MODEL_SIZE is not set (Reconnaissance mode)
    if (size != target_shared_size || target_shared_size == 0) {
        cudaError_t res = real_cudaMalloc(devPtr, size);
        
        // Logging for personal KV cache allocations and recon mode
        if (res == cudaSuccess) {
            fprintf(stderr, "[VRAM_HOOK] cudaMalloc: Allocated %.2f MB (%zu bytes) at %p\n", 
                    (double)size / (1024 * 1024), size, *devPtr);
        }
        return res;
    }

    fprintf(stderr, "\n[VRAM_HOOK] ===================================================\n");
    fprintf(stderr, "[VRAM_HOOK] Target weights size intercepted: %.2f MB\n", (double)size / (1024 * 1024));

    // Infinite loop for attempting to acquire leadership (Master role)
    while (1) { 
        
        // 1. Attempt to ATOMICALLY create the file. Only one process will succeed.
        int shm_fd = shm_open(SHM_NAME, O_CREAT | O_EXCL | O_RDWR, 0666);
        
        if (shm_fd >= 0) {
            // =================================================================
            // ROLE: MASTER
            // =================================================================
            fprintf(stderr, "[VRAM_HOOK] No active MASTER found. Assuming MASTER role.\n");
            
            ftruncate(shm_fd, sizeof(ipc_data_t));
            ipc_data_t* data = mmap(0, sizeof(ipc_data_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            
            data->is_ready = 0;
            data->master_pid = getpid();

            // Allocate actual VRAM on the GPU
            cudaError_t res = real_cudaMalloc(devPtr, size);
            if (res != cudaSuccess) {
                shm_unlink(SHM_NAME); // Clean up the shared memory file on error
                return res;
            }

            // Get the IPC handle and signal WORKERs that they can connect
            cudaIpcGetMemHandle(&data->handle, *devPtr);
            __sync_synchronize(); // Memory barrier to ensure visibility across processes
            data->is_ready = 1;

            munmap(data, sizeof(ipc_data_t));
            close(shm_fd);

            fprintf(stderr, "[VRAM_HOOK] MASTER: Memory allocated at %p and published.\n", *devPtr);
            fprintf(stderr, "[VRAM_HOOK] ===================================================\n\n");
            return res;

        } else if (errno == EEXIST) {
            // =================================================================
            // ROLE: WORKER (File was already created by another process)
            // =================================================================
            shm_fd = shm_open(SHM_NAME, O_RDWR, 0666);
            if (shm_fd < 0) continue; // File was deleted just now? Retry to become MASTER.

            // Wait until the file reaches the required size to avoid reading garbage data
            struct stat st;
            fstat(shm_fd, &st);
            while (st.st_size < sizeof(ipc_data_t)) {
                usleep(100000); // Wait 100ms
                fstat(shm_fd, &st);
            }

            ipc_data_t* data = mmap(0, sizeof(ipc_data_t), PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
            
            // HEARTBEAT CHECK: Is the MASTER process still alive?
            if (kill(data->master_pid, 0) == -1 && errno == ESRCH) {
                fprintf(stderr, "[VRAM_HOOK] WARNING: Stale IPC found. Master (PID %d) is DEAD.\n", data->master_pid);
                fprintf(stderr, "[VRAM_HOOK] Taking over and becoming the new MASTER...\n");
                munmap(data, sizeof(ipc_data_t));
                close(shm_fd);
                shm_unlink(SHM_NAME); // Delete the zombie file
                continue; // Restart the loop (go back to O_CREAT)
            }

            fprintf(stderr, "[VRAM_HOOK] Found active MASTER (PID %d). Assuming WORKER role.\n", data->master_pid);
            
            // Wait for the MASTER to finish loading the heavy weights into VRAM
            int wait_sec = 0;
            while (data->is_ready == 0) {
                if (wait_sec % 5 == 0) {
                    fprintf(stderr, "[VRAM_HOOK] Waiting for MASTER to finish loading weights to VRAM...\n");
                }
                sleep(1);
                wait_sec++;
                
                // Break if the MASTER suddenly dies during the loading process
                if (kill(data->master_pid, 0) == -1 && errno == ESRCH) break; 
            }

            if (data->is_ready == 0) {
                // Fell out of the loop because the MASTER died before finishing.
                // Clean up and take over leadership.
                munmap(data, sizeof(ipc_data_t));
                close(shm_fd);
                shm_unlink(SHM_NAME);
                continue; 
            }

            // MASTER is ready. Map the shared VRAM into the current process!
            cudaError_t res = cudaIpcOpenMemHandle(devPtr, data->handle, cudaIpcMemLazyEnablePeerAccess);
            if (res == cudaSuccess) {
                // REMEMBER the pointer so we can intercept cudaFree later!
                worker_mapped_ptr = *devPtr;
                fprintf(stderr, "[VRAM_HOOK] WORKER: IPC handle mapped successfully at %p. ZERO VRAM spent!\n", *devPtr);
            } else {
                fprintf(stderr, "[VRAM_HOOK] ERROR: cudaIpcOpenMemHandle failed (code %d).\n", res);
            }

            munmap(data, sizeof(ipc_data_t));
            close(shm_fd);
            fprintf(stderr, "[VRAM_HOOK] ===================================================\n\n");
            return res;

        } else {
            // Critical OS error (e.g., out of file descriptors, permission denied)
            perror("[VRAM_HOOK] shm_open fatal error");
            return real_cudaMalloc(devPtr, size);
        }
    }
}