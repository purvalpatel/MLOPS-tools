## Without KServe

You manually start vLLM:
```
docker run \
  --gpus all \
  -p 8000:8000 \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B
```
Now users access:
```
http://server:8000/v1/chat/completions
```
You manage:
- Pod creation
- Scaling
- Updates
- Restarts

yourself.

## With KServe

You tell Kubernetes:
```
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: qwen3
spec:
  predictor:
    containers:
    - image: vllm/vllm-openai:latest
      args:
      - --model
      - Qwen/Qwen3-8B
      resources:
        limits:
          amd.com/gpu: 1
```

Deploy:
```
kubectl apply -f qwen3.yaml
```
KServe automatically:
- Create Deployment
- Create Service
- Create Endpoint
- Manage Health Checks
- Restart Failed Pods
- Expose URL

Check Status:
```
kubectl get inferenceservices
```
Output:
```
NAME    URL
qwen3   http://qwen3.default.example.com
```
Call the Model:
```
curl http://qwen3.default.example.com/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
  "model":"Qwen/Qwen3-8B",
  "messages":[
    {
      "role":"user",
      "content":"What is Kubernetes?"
    }
  ]
}'
```
Response:
```
{
  "choices": [
    {
      "message": {
        "content": "Kubernetes is..."
      }
    }
  ]
}
```

KServe provides:

- Model deployment
- Autoscaling
- Rolling updates
- Canary releases
- Traffic splitting
- Monitoring integration

while vLLM provides:
- Model loading
- Inference
- Token generation
- Batching
- KV cache
