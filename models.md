# Qwen3-30B-A3B Model
docker-compose.yaml
```
version: "3.9"
services:
  vllm-qwen35:
    image: rocm/vllm-dev:nightly_main_20260211
    container_name: vllm-qwen3B-A3B
    devices:
      - /dev/kfd
      - /dev/dri
    group_add:
      - video
      - render
    ipc: host
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    ports:
      - "xxxx:8000"
    volumes:
      - /home/xxxx/docker-files/vllm/Qwen3-30B-A3B/models:/models:ro
      - /home/xxxx/.cache/huggingface:/root/.cache/huggingface
    environment:
      HSA_OVERRIDE_GFX_VERSION: "9.4.2"
      VLLM_ROCM_USE_AITER: "1"
      # Only HIP_VISIBLE_DEVICES is set — combining it with ROCR_VISIBLE_DEVICES
      # double-filters devices and can leave fewer GPUs visible than expected.
      HIP_VISIBLE_DEVICES: "6,7"
    command: >
      bash -c "
      pip install -q --upgrade 'transformers>=4.56.0' &&
      python3 -c 'import torch; print(\"Visible GPUs:\", torch.cuda.device_count())' &&
      vllm serve /models --served-model-name Qwen3-30B-A3B --dtype bfloat16 --tensor-parallel-size 2 --gpu-memory-utilization 0.70 --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder --host 0.0.0.0 --port 8000
      "
    restart: unless-stopped
```
### Troubleshooting:
- In `rocm-smi` command 3,4,5 Number GPUs VRAM is showing free. 0% utilized.
- But when the model is start deplying it is showing an error of Memory not available. only 12G Available.
- Use below command to check which GPU have how much amount of VRAM is avaialble.
```
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -e HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  rocm/vllm-dev:nightly_main_20260211 \
  python3 -c "
import torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print(f'index {i}: free={free/1e9:.1f}GB / total={total/1e9:.1f}GB')
"
```
- Then according to use the GPU ID into docker-compose as the output of this command is showing. here GPU id is INDEX id of command.

Sample Curl:
```
curl -X POST "http://xx.xxx.xx.xx:xxxx/v1/completions"    -H "Content-Type: application/json"     --data '{
                "model": "Qwen3-30B-A3B",
                "prompt": "capital of france ?",
                "max_tokens": 512,
                "temperature": 0.5
        }'
```

### Want to set Authorization token ?
Add below parameter into docker-compose.yaml
```
--api-key my-secret-token
```
How to test?
```
curl -X POST "http://xx.xxx.xxx.xx:11430/v1/completions"    -H "Content-Type:H "Authorization: Bearer my-secret-token"   - application/json"     --data '{
                "model": "Qwen3-30B-A3B",
                "prompt": "capital of france ?",
                "max_tokens": 512,
                "temperature": 0.5
        }'
```
