# cuda-llm-weight-share

A lightweight `LD_PRELOAD` library that allows multiple independent Linux processes to share a single copy of a large machine learning model's weights in CUDA VRAM. 

This tool is designed to drastically reduce VRAM usage when running multiple instances of LLM servers (like `llama.cpp`, `vLLM`, or custom TensorRT-LLM deployments) on the same GPU.

## 🚀 Why do you need this?

Normally, if you run 5 instances of a 30GB model on a single GPU, CUDA will allocate 5 × 30 = 150 GB of VRAM. This is highly inefficient because model weights are **Read-Only**.

This library intercepts `cudaMalloc` and uses **CUDA IPC (Inter-Process Communication)** to share the VRAM containing the weights across multiple processes. 

* **Master Process:** Allocates the actual memory for weights and exports an IPC handle.
* **Worker Processes:** Intercept the allocation request, read the IPC handle, and map the Master's memory into their own virtual address space—spending **0 extra VRAM** for weights.
* **Private KV Cache:** Smaller, private memory allocations (like KV Cache) are safely isolated for each process.

### ✨ Features
- **Auto-Discovery:** No need to hardcode Master/Worker roles. The first process automatically becomes the Master; subsequent processes become Workers.
- **Self-Healing:** If the Master process dies, the next Worker automatically cleans up the stale IPC and promotes itself to Master.
- **Engine Agnostic:** Works blindly at the CUDA runtime level. Tested primarily with `llama.cpp` (`ggml`), but should work with any framework that loads weights in a single large chunk.

---

## 🛠️ Building

### Prerequisites
- Linux OS
- CUDA Toolkit installed (nvcc, headers)
- GCC compiler

### Compilation
Compile the C code into a shared library:

```bash
gcc -shared -fPIC -o cuda-llm-weight-share.so cuda-llm-weight-share.c -ldl -lrt -I/usr/local/cuda/include
```

## 📖 Usage

### Step 1: Reconnaissance (Find your model size)

First, you need to know the exact size (in bytes) of the memory chunk your engine allocates for the model weights.

Run your server normally with the interceptor loaded, but **without** setting the `MODEL_SIZE` variable:

```bash
LD_PRELOAD=./cuda-llm-weight-share.so ./llama-server -m your_model.gguf
```

Look at the stderr output. You will see several allocations. Find the largest one (e.g., 30+ GB):

```
[VRAM_HOOK] cudaMalloc: Allocated 62.81 MB (65863680 bytes) at 0xee1036200000
[VRAM_HOOK] cudaMalloc: Allocated 30575.16 MB (32060375552 bytes) at 0xee0880000000
[VRAM_HOOK] cudaMalloc: Allocated 5120.00 MB (5368709120 bytes) at 0xee0740000000
[VRAM_HOOK] cudaMalloc: Allocated 62.81 MB (65863680 bytes) at 0xee0ff7200000
[VRAM_HOOK] cudaMalloc: Allocated 324.02 MB (339755264 bytes) at 0xee1020400000

```

Copy that exact byte size (`32060375552`).

### Step 2: Running in Production Mode

Now, set the `MODEL_SIZE` environment variable for all your instances.

**Terminal 1 (Instance A):**

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_SIZE=32060375552
LD_PRELOAD=./cuda-llm-weight-share.so ./llama-server -m your_model.gguf --port 8000
```

```
[VRAM_HOOK] cudaMalloc: Allocated 248.10 MB (260149504 bytes) at 0x32ee00000
[VRAM_HOOK] cudaMalloc: Allocated 62.81 MB (65863680 bytes) at 0xea1036200000

[VRAM_HOOK] ===================================================
[VRAM_HOOK] Target weights size intercepted: 30575.16 MB
[VRAM_HOOK] WARNING: Stale IPC found. Master (PID 35709) is DEAD.
[VRAM_HOOK] Taking over and becoming the new MASTER...
[VRAM_HOOK] No active MASTER found. Assuming MASTER role.
[VRAM_HOOK] MASTER: Memory allocated at 0xea0880000000 and published.
[VRAM_HOOK] ===================================================

[VRAM_HOOK] cudaMalloc: Allocated 5120.00 MB (5368709120 bytes) at 0xea0740000000
[VRAM_HOOK] cudaMalloc: Allocated 62.81 MB (65863680 bytes) at 0xea0ff7200000
```

**Terminal 2 (Instance B):**

```bash
export CUDA_VISIBLE_DEVICES=0
export MODEL_SIZE=32060375552
LD_PRELOAD=./cuda-llm-weight-share.so ./llama-server -m your_model.gguf --port 8001
```

```
[VRAM_HOOK] cudaMalloc: Allocated 248.10 MB (260149504 bytes) at 0x32ee00000
[VRAM_HOOK] cudaMalloc: Allocated 62.81 MB (65863680 bytes) at 0xfc9bf6200000

[VRAM_HOOK] ===================================================
[VRAM_HOOK] Target weights size intercepted: 30575.16 MB
[VRAM_HOOK] Found active MASTER (PID 59906). Assuming WORKER role.
[VRAM_HOOK] WORKER: IPC handle mapped successfully at 0xfc9440000000. ZERO VRAM spent!
[VRAM_HOOK] ===================================================

[VRAM_HOOK] cudaMalloc: Allocated 5120.00 MB (5368709120 bytes) at 0xfc9300000000
[VRAM_HOOK] cudaMalloc: Allocated 62.81 MB (65863680 bytes) at 0xfc9bb7200000
```


GPU Memory usage for 2 llama.cpp processes with the same model.
```
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|    0   N/A  N/A           60765    C+G   ...ma.cpp/build/bin/llama-server      37392MiB |
|    0   N/A  N/A           60829    C+G   ...ma.cpp/build/bin/llama-server       6816MiB |
+-----------------------------------------------------------------------------------------+

```

You can run as many instances as your remaining VRAM (for private KV caches) allows!



## ⚠️ Important Considerations

**CUDA Context Alignment:** All processes sharing the memory must run on the same physical GPU. Always use `export CUDA_VISIBLE_DEVICES=0` (or whichever specific GPU ID you target) to ensure alignment.

**Termination Sequence:** If the Master process dies, the shared VRAM persists as long as at least one Worker is alive (CUDA reference counting). However, new workers cannot connect to a dead Master's handle. The library handles this gracefully by taking over the Master role upon detecting a dead PID.

## 📄 License

This project is licensed under the MIT License.