# llama3.3-70B Model
docker-compose.yaml

> Note: Download model from the Huggingface and run.

```YAML
services:
  vllm-llama:
    image: vllm/vllm-openai:latest
    container_name: vllm-llama3.3-70b
    runtime: nvidia
    entrypoint: [""]
    environment:
      - NVIDIA_VISIBLE_DEVICES=1,2
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
      - HUGGING_FACE_HUB_TOKEN=${HF_TOKEN}
    ipc: host
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    ports:
      - "11436:8000"
    volumes:
      - /Data-1/docker/volumes/vllm_data/models:/models
      - /Data-1/docker/volumes/vllm_data/vllm-cache:/root/.cache/huggingface
    command: >
      bash -c "pip install mistral_common -q &&
      vllm serve meta-llama/Llama-3.3-70B-Instruct
      --served-model-name llama3.3-70b
      --tensor-parallel-size 2
      --gpu-memory-utilization 0.9
      --max-model-len 8192
      --host 0.0.0.0
      --port 8000"

    restart: unless-stopped

```
Start:
```
docker-compose up -d
```
> Note : Model is downloaded at **/Data-1/docker/volumes/vllm_data/models** location.

### Test:
```
curl -X POST "http://10.120.130.62:11436/v1/chat/completions"    -H "Content-Type: application/json"     --data '{
                "model": "llama3.3-70b",
                "messages": [
                        {
                                "role": "user",
                                "content": "What is the capital of France?"
                        }
                ]
        }'

```
# Qwen3-3-B model
docker-compose.yaml

> Note:  Load the already downloaded model.

```YAML
services:
  vllm-qwen3:
    image: vllm/vllm-openai:latest
    container_name: vllm-Qwen3-30B-A3B-Instruct-2507
    runtime: nvidia
    entrypoint: [""]
    environment:
      - NVIDIA_VISIBLE_DEVICES=6,7
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    ipc: host
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    ports:
      - "11438:8000"
    volumes:
      - /Data-1/docker/volumes/vllm_data/models/qwen3-30b:/models
      - ${VLLM_CACHE}:/root/.cache/vllm
    command: >
      bash -c "pip install mistral_common -q &&
      vllm serve /models
      --served-model-name Qwen3-30B-A3B-Instruct-2507
      --tensor-parallel-size 2
      --host 0.0.0.0
      --port 8000"
    restart: unless-stopped
```
start:
```
docker-compose up -d
```

# Custom model
docker-compose.yaml

> Note: Load the customized model. model is stored at - /Data-1/docker/volumes/vllm_data/models/xxxx-model

```
services:
  vllm-qwen3:
    image: vllm/vllm-openai:latest
    container_name: vllm-xxxx-model
    runtime: nvidia
    entrypoint: [""]
    environment:
      - NVIDIA_VISIBLE_DEVICES=5
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
      - VLLM_WORKER_MULTIPROC_METHOD=spawn
    ipc: host
    cap_add:
      - SYS_PTRACE
    security_opt:
      - seccomp=unconfined
    ports:
      - "11438:8000"
    volumes:
      - /Data-1/docker/volumes/vllm_data/models/xxxx-model:/models
    command: >
      bash -c "pip install mistral_common -q &&
      vllm serve /models
      --served-model-name xxxxx-model
      --tensor-parallel-size 1
      --gpu-memory-utilization 0.9
      --max-model-len 8192
      --host 0.0.0.0
      --port 8000"
    restart: unless-stopped
```
start: `docker-compose.yaml`

