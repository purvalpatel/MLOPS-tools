### Setup k8sgpt package:

#### Download K8sgpt for ubuntu:
```
curl -LO https://github.com/k8sgpt-ai/k8sgpt/releases/download/v0.4.33/k8sgpt_amd64.deb
```

#### Install:
```
dpkg -i k8sgpt_amd64.deb
```

> Kubernetes sends the cluster diagnostics to OpenAI APIs. but it is paid so we can use local inference server ollama.

### Setup ollama if not exists.

#### Pull model in ollama if not available:
```
ollama pull qwen3:8b
```

#### Verify:
```
ollama list
```

#### Test:
```
curl http://localhost:11434/api/generate -d '{
  "model":"qwen3:8b",
  "prompt":"What is CrashLoopBackOff?"
}'
```
### Configure k8sgpt:

#### Add ollama:
```
k8sgpt auth add \
  --backend localai \
  --baseurl http://10.10.110.25:11434/v1 \
  --model qwen3:8b
```
#### Set default provider:
```
k8sgpt auth default --provider localai
```

#### Check available providers:
```
k8sgpt auth list
```

#### Analyze cluster:
```
k8sgpt analyze
```

#### Explain with AI:
```
k8sgpt analyze --explain
```
<img width="2280" height="559" alt="image" src="https://github.com/user-attachments/assets/1c23fd83-bf81-44a9-8d69-5b18f10abd5a" />

