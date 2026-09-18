## GPU Optimization for LLM inference

GPU optimization means getting maximum tokens/sec and minimum latency while keeping GPU memory and compute efficiently utilized.

```
                 LLM Inference
                      │
       ┌──────────────┼──────────────┐
       ↓              ↓              ↓
   GPU Memory      GPU Compute    Data Transfer
       │              │              │
   KV Cache        GEMM/MFMA       CPU ↔ GPU
   Weights         Attention       PCIe/NVLink
   Activations     Kernels         Host Memory

```

For AMD important characteristics are :
```
ROCm
 ├── HIP
 ├── rocBLAS / hipBLASLt
 ├── MIOpen
 └── AITER
 ```

1. GPT-OSS 120B pre-loaded
2. Optimized vLLM configuration for MI300X
3. All AITER environment variables pre-set. Optimize attention.
| Variable | Purpose |
|----------|---------|
| `VLLM_ROCM_USE_AITER=1` | Master switch — forces vLLM to use AITER's optimized C++ kernels instead of standard Python attention layers |
| `VLLM_ROCM_USE_AITER_MHA=0` | Disables AITER's multi-head attention (still evolving, kept off for stability) |
| `VLLM_ROCM_QUICK_REDUCE_QUANTIZATION=INT4` | Uses 4-bit integers to maximize memory bandwidth |
| `HIP_FORCE_DEV_KERNARG=1` | Micro-optimization for faster kernel argument passing |

Start model deployment
```
vllm serve openai/gpt-oss-120b \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --no-enable-chunked-prefill \
    --max-num-seqs 512 \
    --port 8080
```

Retrive the available models:
```
response = requests.get(f"{BASE_URL}/models")
models = response.json()

MODEL_NAME = models["data"][0]["id"]
print(f"Using model: {MODEL_NAME}")
```
Test Chat Completion:
```
response = requests.post(
    f"{BASE_URL}/chat/completions",
    json={
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How many r's are in the phrase Strawberry Milkshaker?"}
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

Running benchmarks:
```
!cd GPU-Optimization-for-LLM-Inference && python summary_benchmark.py --port 8080 --model "openai/gpt-oss-120b"
```

Configure AITER Environment Variables

Set up ROCm and AITER optimizations for best performance on MI300X.

### Distributed Communication Variables

| Variable | Purpose |
|----------|---------|
| `GLOO_SOCKET_IFNAME=eth0` | PyTorch distributed backend for process communication |
| `NCCL_SOCKET_IFNAME=eth0` | RCCL (AMD's NCCL equivalent) for heavy data movement between processes |

### vLLM ROCm/AITER Variables

| Variable | Purpose |
|----------|---------|
| `VLLM_ROCM_USE_AITER=1` | Master switch — uses AITER's optimized C++ kernels |
| `VLLM_ROCM_USE_AITER_TRITON_FUSED_ROPE_ZEROS_KV_CACHE=1` | Fuses KV cache dequantization and RoPE calculation into one step |
| `VLLM_ROCM_USE_AITER_LINEAR=1` | Routes matrix multiplications through optimized AMD kernels |
| `TORCH_BLAS_PREFER_HIPBLASLT=1` | Prefer hipBLASLt for matrix ops |


### Launch DeepSeek-R1 with Optimized Settings
```
%%bash --bg --out vllm_log --err vllm_err
python -m vllm.entrypoints.openai.api_server \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-70B \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --trust-remote-code \
    --dtype bfloat16 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.92 \
    --no-enable-chunked-prefill \
    --max-num-seqs 512 \
    --port 8080
```

Run benchmark:
```
!python /workspace/GPU-Optimization-for-LLM-Inference/summary_benchmark.py --port 8080 --model "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
```


`AITER `  : AI Tensor Engine for ROCm

`TORCH_BLAS_PREFER_HIPBLASLT=1`       :       Prefers the hipBLASLt backend for optimized matrix multiplication on AMD GPUs

`VLLM_ROCM_USE_AITER=1 `              :       Forces VLLM to use AITER's AMD C++ kernels instead of standard Python attention layers

`--device /dev/kfd `      :   Kernel Fusion Driver for GPU compute

`MHA` : Multi head Attention

`--ipc=host`      :       It allows multiple worker processes to share memory for tensor exchange

important metrics:
| Metric                     | Meaning               |
| -------------------------- | --------------------- |
| **TTFT**                   | Time To First Token   |
| **ITL**                    | Inter-Token Latency   |
| **Tokens/sec**             | Generation throughput |
| **Requests/sec**           | Server throughput     |
| **GPU utilization**        | GPU activity          |
| **GPU memory utilization** | VRAM usage            |
| **HBM bandwidth**          | Memory pressure       |
| **KV cache usage**         | Cache consumption     |
| **Batch size**             | Concurrent workload   |
| **Power/Watt**             | Energy efficiency     |
