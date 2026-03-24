## LLM Prompt tracing with Phoenix:

> Currentlt the tracing mechanism is working only for network traffic, Not LLM logic.

So it shows,

- URL
- Status code
- Request/Response size
- Request/Response time
- body etc.

This data comes from the `envoy proxy`.

- But it will not show the Promt, token, this lived inside the request body.

> Istio does not parse the request body. <br>
This information is only visible to the application that actually calls the model.

```
client -> Istio ingressgateway -> Envoy proxy -> OpenTelemetry Collector -> Phoenix
```

To get this type of details, something must run code that knows the prompt and response. <br>

So we need one application that runs between the ingress gateway and vLLM. which we can call as  API Service.

```
client -> Istio ingressgateway -> API Service -> vLLM service -> vLLM Pod -> GPU
```

> Istio : networking, routing, security <br>
API Services: Prompt processing, logging, tracing <br>
vLLM : GPU inference.

Flow:
```
Istio ingress gateway
    |
FastAPI Wrapper (API Service)  <-- This needs to added.
    |
vLLM service (ClusterIP)
    |
vLLM Pod
```
Docker Registry : docker.merai.app/devops/llama-llm-proxy:0.7

### API-Proxy image create:
app.py
```
from fastapi import FastAPI, Request, HTTPException
import requests
import time
import os
import json

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# OpenInference
from openinference.semconv.trace import (
    SpanAttributes,
)

# ---------------------------------------------------
# Phoenix setup
# ---------------------------------------------------
resource = Resource(attributes={
    "openinference.project.name": "vllm-observability"
})

trace.set_tracer_provider(TracerProvider(resource=resource))

otlp_exporter = OTLPSpanExporter(
    endpoint=os.getenv("PHOENIX_ENDPOINT", "phoenix.observability.svc.cluster.local:4317"),
    insecure=True,
    headers={"x-phoenix-project-name": "vllm-observability"}
)

trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(otlp_exporter)
)

tracer = trace.get_tracer(__name__)

# ---------------------------------------------------
# Model routing
# ---------------------------------------------------
PATH_ROUTES = {
    "gemma3-27b": os.getenv("GEMMA3_URL", "http://gemma3-27b-service:8000"),
    "kimi": os.getenv("KIMI_URL", "http://kimi-service:8000"),
    "llama": os.getenv("LLAMA_URL", "http://llama-service:8000"),
    "nucurate": os.getenv("NUCURATE_URL", "http://nucurate-model-service:8000"),
    "qwen3.5-27b": os.getenv("QWEN_URL", "http://qwen3-5-svc:8000"),
}

DEFAULT_VLLM_URL = os.getenv("DEFAULT_VLLM_URL", "http://llama-service:8000")

# ---------------------------------------------------
# FastAPI
# ---------------------------------------------------
app = FastAPI(title="Multi-Model Proxy")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


# ---------------------------------------------------
# Safe JSON parsing
# ---------------------------------------------------
async def safe_json(request: Request):
    try:
        return await request.json()
    except Exception:
        raw = await request.body()
        print("❌ INVALID JSON:", raw.decode())
        raise HTTPException(status_code=400, detail="Invalid JSON payload")


# ---------------------------------------------------
# Model resolver
# ---------------------------------------------------
def resolve_by_model(model_name: str):
    if not model_name:
        return DEFAULT_VLLM_URL

    lower = model_name.lower()
    for key, url in PATH_ROUTES.items():
        if key in lower:
            return url

    return DEFAULT_VLLM_URL


# ---------------------------------------------------
# Token estimation fallback
# ---------------------------------------------------
def estimate_tokens(prompt, response):
    pt = max(1, len(prompt) // 4)
    ct = max(1, len(response) // 4)
    return pt, ct, pt + ct


# ---------------------------------------------------
# Core forward logic
# ---------------------------------------------------
async def _forward(data, backend_url, route_label, endpoint="/v1/chat/completions"):

    messages = data.get("messages", [])
    model_name = data.get("model", route_label)

    full_url = f"{backend_url.rstrip('/')}{endpoint}"

    print(f"[PROXY] route={route_label} → {full_url}")

    # Convert payload for completion models
    if endpoint == "/v1/completions":
        prompt = data.get("prompt")
        if not prompt and messages:
            prompt = messages[0].get("content", "")
        data = {"model": model_name, "prompt": prompt}

    with tracer.start_as_current_span(f"vllm-{route_label}-call", kind=SpanKind.CLIENT) as span:

        # Mark LLM span
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")

        # ---------------- INPUT ----------------
        span.set_attribute(
            SpanAttributes.INPUT_VALUE,
            json.dumps(messages)
        )

        for i, msg in enumerate(messages):
            span.set_attribute(f"llm.input_messages.{i}.message.role", msg.get("role", "user"))
            span.set_attribute(f"llm.input_messages.{i}.message.content", str(msg.get("content", "")))

        start = time.time()

        try:
            response = requests.post(full_url, json=data, timeout=120)
            latency = time.time() - start
            response.raise_for_status()
            output = response.json()

        except Exception as e:
            latency = time.time() - start
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.set_attribute("llm.latency", latency)
            raise HTTPException(status_code=500, detail=str(e))

        # ---------------- OUTPUT ----------------
        span.set_attribute(
            SpanAttributes.OUTPUT_VALUE,
            json.dumps(output)[:2000]
        )

        response_text = ""

        for i, choice in enumerate(output.get("choices", [])):
            msg = choice.get("message", {})
            text = msg.get("content", "")
            response_text = text

            span.set_attribute(f"llm.output_messages.{i}.message.role", msg.get("role", "assistant"))
            span.set_attribute(f"llm.output_messages.{i}.message.content", text)

        # ---------------- TOKENS ----------------
        usage = output.get("usage")

        if usage:
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            tt = usage.get("total_tokens", 0)
        else:
            prompt_text = " ".join([str(m.get("content", "")) for m in messages])
            pt, ct, tt = estimate_tokens(prompt_text, response_text)

        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT, pt)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION, ct)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL, tt)

        span.set_attribute("llm.latency", latency)

    return output


# ---------------------------------------------------
# Path routing
# ---------------------------------------------------
@app.post("/{service}/v1/chat/completions")
async def proxy_by_path(service: str, request: Request):
    backend_url = PATH_ROUTES.get(service)
    if not backend_url:
        raise HTTPException(status_code=404, detail=f"Unknown service {service}")

    data = await safe_json(request)
    return await _forward(data, backend_url, service)


# ---------------------------------------------------
# Chat fallback
# ---------------------------------------------------
@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    data = await safe_json(request)
    backend = resolve_by_model(data.get("model", ""))
    return await _forward(data, backend, data.get("model", "default"))


# ---------------------------------------------------
# Completion endpoint (nucurate)
# ---------------------------------------------------
@app.post("/v1/completions")
async def proxy_completion(request: Request):
    data = await safe_json(request)
    backend = resolve_by_model(data.get("model", ""))
    return await _forward(data, backend, data.get("model", "default"), endpoint="/v1/completions")
```

Dockerfile:
```
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
requirements.txt
```
fastapi
uvicorn
requests
arize-phoenix
openinference-semantic-conventions
```
Build and push the image to your registry:
```
sudo docker build -t docker.merai.app/devops/llama-llm-proxy:0.5 .
sudo docker push docker.merai.app/devops/llama-llm-proxy:0.5
```
#### Deployment - Fast-api-wrapper.yaml 
```YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llama-llm-proxy
  namespace: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llama-llm-proxy
  template:
    metadata:
      labels:
        app: llama-llm-proxy
    spec:
      containers:
      - name: llama-llm-proxy
        image: docker.merai.app/devops/llama-llm-proxy:0.7
        imagePullPolicy: Always
        ports:
        - containerPort: 8000
        env:
        - name: LLAMA_URL
          value: "http://llama-service:8000"
        - name: GEMMA3_URL
          value: "http://gemma3-27b-service:8000"
        - name: NUCURATE_URL
          value: "http://nucurate-model-service:8000"
        - name: KIMI_URL
          value: "http://kimi-service:8000"
        - name: QWEN_URL
          value: "http://qwen3-5-svc:8000"
---
apiVersion: v1
kind: Service
metadata:
  name: llama-llm-proxy
  namespace: vllm
spec:
  selector:
    app: llama-llm-proxy
  ports:
    - protocol: TCP
      port: 8000
      targetPort: 8000
```

Make below changes in virtualservice.yaml
```YAML
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: vllm-llama-70b-vs
  namespace: vllm
spec:
  hosts:
  - "*"
  gateways:
  - nucurate-infer-gateway   # your existing Istio gateway
  http:
  - match:
    - uri:
        prefix: /llama3.3-70b/
    rewrite:
      uri: /
    route:
    - destination:
#        host: llama-service # <-- replace this
        host: llama-llm-proxy  ## Call proxy before calling Model to collect prompts
        port:
          number: 8000
```
> Note: <br>
For **google/gemma-3-27b-it** model rewrite to / will give error so comment it. because this model will get work with resolve_by_model function. and for that we need full path.


Now call the model as it is, and check in pheonix:
<img width="1905" height="1080" alt="image" src="https://github.com/user-attachments/assets/f7b33a58-079b-4530-95c3-cd7fa6df2694" />

