# CUDA LLM Weight Share

A lightweight `LD_PRELOAD` library that allows multiple independent Linux processes to share a single selected CUDA allocation, typically a large machine learning model's weights, in CUDA VRAM.

This tool is designed to drastically reduce VRAM usage when running multiple instances of LLM servers, such as `llama.cpp`, on the same GPU.

> Tested primarily with `llama.cpp` / `ggml-cuda`. Other CUDA-based engines may work if their model weights are allocated through a large `cudaMalloc()` allocation.

---

## Why do you need this?

Normally, if you run 5 instances of a 30 GB model on a single GPU, CUDA may allocate:

```text
5 × 30 GB = 150 GB VRAM
```

This is inefficient when the model weights are effectively read-only.

This library intercepts `cudaMalloc()` and uses CUDA IPC, Inter-Process Communication, to share the VRAM allocation containing the model weights across multiple Linux processes.

```text
Master process:
  Performs the real cudaMalloc()
  Exports the allocation through cudaIpcGetMemHandle()

Worker processes:
  Intercept the same cudaMalloc() request
  Open the master's IPC handle through cudaIpcOpenMemHandle()
  Map the same VRAM allocation into their own address space

Private allocations:
  KV cache, scratch buffers, CUDA graph buffers, and other allocations stay private
```

Example result for 2 `llama.cpp` processes using the same model:

```text
Process A: weights + KV/cache/private buffers
Process B: shared weights mapping + its own KV/cache/private buffers
```

---

## Features

* **Automatic Master/Worker election**
  The first process becomes the Master. Later processes become Workers automatically.

* **CUDA IPC based sharing**
  Workers map the Master's CUDA allocation instead of allocating another copy of the model weights.

* **Private KV cache per process**
  Only the selected model weight allocation is shared. Other CUDA allocations remain private.

* **Lazy CUDA symbol resolving**
  CUDA runtime symbols are resolved lazily with `dlsym()` / `dlopen()`. This works better with `llama.cpp` builds using `GGML_BACKEND_DL=ON`, where CUDA backends may be loaded later through `dlopen()`.

* **Stale IPC recovery**
  If a stale `/dev/shm` entry exists from a crashed or killed process, the next process can remove it and elect itself as the new Master.

* **Broken zero-size shm recovery**
  If a process dies after `shm_open()` but before `ftruncate()`, the shm file may remain with size `0`. The library now detects this and removes it after a configurable timeout.

* **Optional allocation size tolerance**
  Useful when the target allocation size slightly changes between builds or backend configurations.

* **Optional caller tracing**
  The library can print call stacks for intercepted `cudaMalloc()` / `cudaFree()` calls to help identify which component performs the allocation.

* **Fallback mode**
  If `cudaIpcOpenMemHandle()` fails, the process can fall back to a normal `cudaMalloc()` instead of crashing.

---

## Building

### Prerequisites

* Linux
* CUDA Toolkit headers
* GCC
* `libdl`

### Build command

```bash
gcc -shared -fPIC -O2 -g -Wall -Wextra \
  -I/usr/local/cuda/include \
  cuda-llm-weight-share.c \
  -o cuda-llm-weight-share.so \
  -ldl
```

`-lrt` is usually not required on modern Linux distributions, but it is harmless if your system needs it:

```bash
gcc -shared -fPIC -O2 -g -Wall -Wextra \
  -I/usr/local/cuda/include \
  cuda-llm-weight-share.c \
  -o cuda-llm-weight-share.so \
  -ldl -lrt
```

The library should not need to link directly against `libcudart.so` because CUDA runtime functions are resolved dynamically at runtime.

Check exported hooks:

```bash
nm -D ./cuda-llm-weight-share.so | grep -E ' cudaMalloc| cudaFree'
```

Expected:

```text
T cudaMalloc
T cudaFree
```

Check that the library does not have a hard dependency on `libcudart`:

```bash
ldd ./cuda-llm-weight-share.so
```

---

## Usage

### Step 1: Reconnaissance mode

First, find the exact CUDA allocation size used for model weights.

Run your server with `LD_PRELOAD`, but without `MODEL_SIZE`:

```bash
LD_PRELOAD=./cuda-llm-weight-share.so \
./llama-server -m your_model.gguf
```

Example output:

```text
[VRAM_HOOK] loaded. MODEL_SIZE=0, TOLERANCE=0, SHM=/cuda_vram_ipc_auto
[VRAM_HOOK] cudaMalloc normal: 860.98 MB (902804096 bytes) at 0xea33e0000000
[VRAM_HOOK] cudaMalloc normal: 248.10 MB (260149504 bytes) at 0x32ee00000
[VRAM_HOOK] cudaMalloc normal: 30575.16 MB (32060375552 bytes) at 0xee0880000000
[VRAM_HOOK] cudaMalloc normal: 5120.00 MB (5368709120 bytes) at 0xee0740000000
```

Find the large allocation that corresponds to model weights.

Example:

```text
32060375552
```

---

### Step 2: Production mode

Set `MODEL_SIZE` for all processes that should share the same model weights.

#### Terminal 1: Master

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_SIZE=32060375552
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_auto"

LD_PRELOAD=./cuda-llm-weight-share.so \
./llama-server -m your_model.gguf --port 8000
```

Example output:

```text
[VRAM_HOOK] loaded. MODEL_SIZE=32060375552, TOLERANCE=0, SHM=/cuda_vram_ipc_auto

[VRAM_HOOK] ===================================================
[VRAM_HOOK] Target weights allocation intercepted: 30575.16 MB (32060375552 bytes)
[VRAM_HOOK] No active MASTER found. Assuming MASTER role.
[VRAM_HOOK] MASTER: allocated at 0xfe8f00000000 and published through /cuda_vram_ipc_auto
[VRAM_HOOK] ===================================================
```

#### Terminal 2: Worker

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_SIZE=32060375552
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_auto"

LD_PRELOAD=./cuda-llm-weight-share.so \
./llama-server -m your_model.gguf --port 8001
```

Example output:

```text
[VRAM_HOOK] loaded. MODEL_SIZE=32060375552, TOLERANCE=0, SHM=/cuda_vram_ipc_auto

[VRAM_HOOK] ===================================================
[VRAM_HOOK] Target weights allocation intercepted: 30575.16 MB (32060375552 bytes)
[VRAM_HOOK] Found active MASTER pid=154167. Assuming WORKER role.
[VRAM_HOOK] WORKER: IPC handle mapped at 0xfa5720000000. No duplicate weights allocation.
[VRAM_HOOK] ===================================================
```

Example `nvidia-smi` result:

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           60765    C+G   .../llama-server                     37392MiB |
|    0   N/A  N/A           60829    C+G   .../llama-server                      6816MiB |
+-----------------------------------------------------------------------------------------+
```

You can run as many worker processes as your remaining VRAM allows for private KV cache and other per-process buffers.

---

## Environment variables

### Required

#### `MODEL_SIZE`

Exact target allocation size in bytes.

```bash
export MODEL_SIZE=32060375552
```

When unset or set to `0`, the library works in reconnaissance mode and only logs allocations.

---

### Optional

#### `MODEL_SIZE_TOLERANCE`

Allows matching allocation sizes within a byte range around `MODEL_SIZE`.

Default:

```bash
export MODEL_SIZE_TOLERANCE=0
```

Example: allow ±16 MiB:

```bash
export MODEL_SIZE_TOLERANCE=16777216
```

This is useful if the target allocation changes slightly between builds, CUDA backend modes, alignment, or engine settings.

---

#### `CUDA_VRAM_IPC_NAME`

Shared memory name used for CUDA IPC metadata.

Default:

```bash
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_auto"
```

Recommended when running different models or GPUs:

```bash
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_qwen3_30b_gpu0"
```

Use a unique name per:

```text
model
GPU
allocation size
runtime configuration
```

---

#### `CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC`

Timeout for detecting broken shm files that exist but were never initialized.

Default:

```bash
export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=10
```

Example:

```bash
export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=3
```

This handles the case where a process dies after `shm_open()` but before `ftruncate()`, leaving a zero-size file in `/dev/shm`.

---

#### `CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE`

Optional mode to suppress the real `cudaFree()` call for the exported master allocation.

Default:

```bash
export CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=0
```

Enable:

```bash
export CUDA_VRAM_IPC_SUPPRESS_MASTER_FREE=1
```

This can be useful if you want existing workers to keep their IPC mapping after the Master exits.

Important: this may leak memory if the Master process unloads/reloads a model without fully exiting. Use only if you understand the lifecycle.

---

#### `CUDA_VRAM_IPC_TRACE_CALLERS`

Enable caller tracing for intercepted target allocations and IPC close/free paths.

Default:

```bash
export CUDA_VRAM_IPC_TRACE_CALLERS=0
```

Enable:

```bash
export CUDA_VRAM_IPC_TRACE_CALLERS=1
```

---

#### `CUDA_VRAM_IPC_TRACE_DEPTH`

Backtrace depth for caller tracing.

Default:

```bash
export CUDA_VRAM_IPC_TRACE_DEPTH=8
```

Example:

```bash
export CUDA_VRAM_IPC_TRACE_DEPTH=32
```

---

#### `CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS`

Trace ordinary non-shared `cudaMalloc()` / `cudaFree()` calls too.

Default:

```bash
export CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS=0
```

Enable:

```bash
export CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS=1
```

This can produce very large logs.

---

## Caller tracing

To see where allocations come from:

```bash
export CUDA_VRAM_IPC_TRACE_CALLERS=1
export CUDA_VRAM_IPC_TRACE_DEPTH=32
```

Run with demangling:

```bash
LD_PRELOAD=./cuda-llm-weight-share.so \
./llama-server ... 2> >(c++filt | tee hook.log)
```

To trace all CUDA allocations:

```bash
export CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS=1
```

For better symbol names, build the target application with debug symbols and exported symbols when possible:

```bash
-g -fno-omit-frame-pointer -rdynamic
```

For CUDA device/kernel-level analysis, use CUDA tools such as:

```bash
cuobjdump
nsys
ncu
```

`LD_PRELOAD` can trace host-side runtime calls such as `cudaMalloc()` and `cudaFree()`, but it cannot trace every internal CUDA device function.

---

## Stale shm cleanup

If a process crashes or is killed, the shm metadata file may remain in `/dev/shm`.

Manual cleanup:

```bash
rm -f /dev/shm/cuda_vram_ipc_auto
```

Or for custom names:

```bash
rm -f /dev/shm/cuda_vram_ipc_*
```

The library can automatically handle:

```text
stale shm with dead master PID
broken shm with size 0
master died before setting is_ready=1
```

Example log:

```text
[VRAM_HOOK] shm /cuda_vram_ipc_auto already exists. Trying WORKER path.
[VRAM_HOOK] Existing shm /cuda_vram_ipc_auto is not initialized yet: size=0 expected=...
[VRAM_HOOK] Existing shm /cuda_vram_ipc_auto did not reach expected size within 3 sec. Treating as stale.
[VRAM_HOOK] Removing stale shm /cuda_vram_ipc_auto and retrying MASTER election.
```

---

## Master and Worker lifecycle

The Master owns the real CUDA allocation.

Workers open the allocation through CUDA IPC.

If the Master exits while Workers are still running, existing Workers may continue to hold and use their IPC mapping on tested setups. In that case, the VRAM may become accounted to the Worker process.

When a new process starts after the old Master exits, it cannot use the dead Master's IPC metadata. The library removes the stale shm entry and promotes the new process to Master, creating a new copy of the weights.

This means the following lifecycle is supported in practice:

```text
Master A starts
Worker B maps Master A weights
Master A exits
Worker B keeps running
Master C starts
Master C allocates a new weights copy
Worker B exits later and closes its IPC mapping
```

This may temporarily duplicate VRAM usage, which is expected.

---

## Important considerations

### Same physical GPU

All processes sharing memory must run on the same physical GPU.

Use:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Do not mix different physical GPUs under the same `CUDA_VRAM_IPC_NAME`.

---

### Same model and runtime configuration

Workers should use the same model and compatible runtime settings so that the target allocation size corresponds to the same weights buffer.

Use unique shm names for different models:

```bash
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_model_a_gpu0"
```

---

### Only one selected allocation is shared

The library intentionally shares only one target `cudaMalloc()` allocation per process.

Other allocations remain normal CUDA allocations.

---

### Exact allocation matching

By default, matching is exact:

```bash
MODEL_SIZE_TOLERANCE=0
```

If the target allocation is not intercepted, check the logs and optionally set a small tolerance:

```bash
export MODEL_SIZE_TOLERANCE=16777216
```

---

### `GGML_BACKEND_DL=ON`

The library resolves CUDA symbols lazily to support runtimes where CUDA backend libraries are loaded later with `dlopen()`.

This is important for `llama.cpp` builds using:

```bash
-DGGML_BACKEND_DL=ON
```

---

## Example full launch

```bash
export CUDA_VISIBLE_DEVICES=0

export MODEL_SIZE=32060375552
export MODEL_SIZE_TOLERANCE=0
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_qwen3_30b_gpu0"
export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=3

LD_PRELOAD=./cuda-llm-weight-share.so \
./llama-server \
  -m /models/your_model.gguf \
  --port 8000
```

Worker:

```bash
export CUDA_VISIBLE_DEVICES=0

export MODEL_SIZE=32060375552
export MODEL_SIZE_TOLERANCE=0
export CUDA_VRAM_IPC_NAME="/cuda_vram_ipc_qwen3_30b_gpu0"
export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=3

LD_PRELOAD=./cuda-llm-weight-share.so \
./llama-server \
  -m /models/your_model.gguf \
  --port 8001
```

---

## Troubleshooting

### The process hangs after "Target weights allocation intercepted"

Check for a stale zero-size shm file:

```bash
ls -lh /dev/shm/cuda_vram_ipc_auto
```

Remove it:

```bash
rm -f /dev/shm/cuda_vram_ipc_auto
```

Or set a shorter timeout:

```bash
export CUDA_VRAM_IPC_SHM_SIZE_WAIT_SEC=3
```

---

### Worker becomes a normal allocation instead of shared

Possible reasons:

```text
MODEL_SIZE does not match
MODEL_SIZE_TOLERANCE is too small
CUDA_VRAM_IPC_NAME differs between processes
processes are on different GPUs
master exited before worker opened the handle
cudaIpcOpenMemHandle failed and fallback cudaMalloc was used
```

Enable tracing:

```bash
export CUDA_VRAM_IPC_TRACE_CALLERS=1
export CUDA_VRAM_IPC_TRACE_DEPTH=32
```

---

### Logs are too noisy

Disable normal allocation tracing:

```bash
unset CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS
```

or:

```bash
export CUDA_VRAM_IPC_TRACE_NORMAL_ALLOCS=0
```

---

## License

This project is licensed under the MIT License.

