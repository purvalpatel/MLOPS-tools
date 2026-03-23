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
    OpenInferenceMimeTypeValues,
)

# ---------------------------------------------------
# Configure Tracing (Phoenix)
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
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

PATH_ROUTES: dict[str, str] = {
    "gemma3-27b": os.getenv("GEMMA3_URL",   "http://gemma3-27b-service:8000"),
    "kimi":       os.getenv("KIMI_URL",      "http://kimi-service:8000"),
    "llama":      os.getenv("LLAMA_URL",     "http://llama-service:8000"),
    "nucurate":   os.getenv("NUCURATE_URL",  "http://nucurate-model-service:8000"),
    "qwen3-5":    os.getenv("QWEN_URL",      "http://qwen3-5-svc:8000"),
    "vllm-qwen":  os.getenv("VLLM_QWEN_URL", "http://vllm-qwen3-5-svc:8000"),
}

DEFAULT_VLLM_URL = os.getenv("DEFAULT_VLLM_URL", "http://llama-service:8000")

print("Loaded path routes:")
for prefix, url in PATH_ROUTES.items():
    print(f"  /{prefix}/*  →  {url}")
print(f"  (default)   →  {DEFAULT_VLLM_URL}")


def resolve_by_model(model_name: str) -> str:
    """
    Fallback resolution when client calls /v1/chat/completions directly.
    Matches the model name string against PATH_ROUTES keys (substring).
    e.g. "google/gemma-3-27b-it"  →  contains "gemma"  →  gemma3-27b-service
    """
    if not model_name:
        return DEFAULT_VLLM_URL
    lower = model_name.lower()
    for key, url in PATH_ROUTES.items():
        # normalise key: "gemma3-27b" → "gemma" also matches
        if key.lower() in lower or lower in key.lower():
            return url
    print(f"[WARN] No route matched for model='{model_name}', falling back to default.")
    return DEFAULT_VLLM_URL


# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(title="Multi-Model vLLM Proxy")


@app.get("/healthz")
async def health():
    return {"status": "ok"}


@app.get("/routes")
async def list_routes():
    """Returns the active routing table — useful for debugging."""
    return {"path_routes": PATH_ROUTES, "default": DEFAULT_VLLM_URL}


# ---------------------------------------------------
# Core tracing + forwarding logic (shared)
# ---------------------------------------------------
async def _forward(
    data: dict,
    backend_url: str,
    route_label: str,
):
    """
    Instruments an OpenInference LLM span and forwards the request to
    the resolved backend service.
    """
    messages = data.get("messages", [])
    model_name = data.get("model", "unknown")
    full_url = f"{backend_url.rstrip('/')}/v1/chat/completions"

    print(f"[PROXY] model={model_name!r}  route={route_label!r}  →  {full_url}")

    with tracer.start_as_current_span(f"vllm-{route_label}-call", kind=SpanKind.CLIENT) as span:

        # ── OpenInference metadata ──────────────────────────────
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, model_name)
        span.set_attribute("llm.backend_url", backend_url)
        span.set_attribute("llm.route_label", route_label)

        # ── Input ───────────────────────────────────────────────
        span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(messages))

        for i, msg in enumerate(messages):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Multimodal content arrives as a list — serialise for the span
            if isinstance(content, list):
                content = json.dumps(content)
            span.set_attribute(f"llm.input_messages.{i}.message.role", role)
            span.set_attribute(f"llm.input_messages.{i}.message.content", content)

        # ── Call backend ────────────────────────────────────────
        start = time.time()
        try:
            response = requests.post(
                full_url,
                json=data,
                timeout=120,
                headers={"Content-Type": "application/json"},
            )
            latency = time.time() - start
            response.raise_for_status()
            output = response.json()

        except requests.exceptions.Timeout:
            latency = time.time() - start
            _record_error(span, latency, "timeout")
            raise HTTPException(status_code=504, detail=f"Backend timeout: {full_url}")

        except requests.exceptions.ConnectionError as e:
            latency = time.time() - start
            _record_error(span, latency, str(e))
            raise HTTPException(status_code=502, detail=f"Cannot connect to {full_url}: {e}")

        except requests.exceptions.HTTPError as e:
            latency = time.time() - start
            _record_error(span, latency, str(e))
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Backend returned error: {e}",
            )

        except Exception as e:
            latency = time.time() - start
            _record_error(span, latency, str(e))
            raise HTTPException(status_code=500, detail=str(e))

        # ── Output ──────────────────────────────────────────────
        for i, choice in enumerate(output.get("choices", [])):
            msg_out = choice.get("message", {})
            span.set_attribute(f"llm.output_messages.{i}.message.role",    msg_out.get("role", "assistant"))
            span.set_attribute(f"llm.output_messages.{i}.message.content", msg_out.get("content", ""))

        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(output))
        span.set_attribute(SpanAttributes.OUTPUT_MIME_TYPE, OpenInferenceMimeTypeValues.JSON.value)
        span.set_attribute("llm.latency", latency)

        # ── Token usage ─────────────────────────────────────────
        usage = output.get("usage", {})
        if usage:
            span.set_attribute("llm.token_count.prompt",     usage.get("prompt_tokens", 0))
            span.set_attribute("llm.token_count.completion", usage.get("completion_tokens", 0))
            span.set_attribute("llm.token_count.total",      usage.get("total_tokens", 0))

    return output


def _record_error(span, latency: float, message: str):
    span.set_attribute("error", True)
    span.set_attribute("error.message", message)
    span.set_attribute("llm.latency", latency)


# ---------------------------------------------------
# Route A — Path-prefixed (recommended)
#
#   POST /{service}/v1/chat/completions
#
#   Examples:
#     /gemma3-27b/v1/chat/completions  →  gemma3-27b-service:8000
#     /kimi/v1/chat/completions        →  kimi-service:8000
#     /llama/v1/chat/completions       →  llama-service:8000
# ---------------------------------------------------
@app.post("/{service}/v1/chat/completions")
async def proxy_by_path(service: str, request: Request):
    backend_url = PATH_ROUTES.get(service)
    if backend_url is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unknown service prefix '/{service}'. "
                f"Valid prefixes: {list(PATH_ROUTES.keys())}"
            ),
        )
    data = await request.json()
    return await _forward(data, backend_url=backend_url, route_label=service)


# ---------------------------------------------------
# Route B — Direct / model-name fallback
#
#   POST /v1/chat/completions
#
#   Routes by the "model" field in the request body.
#   Useful for clients that don't support path prefixes
#   (e.g. LiteLLM, OpenAI SDK with base_url set here).
# ---------------------------------------------------
@app.post("/v1/chat/completions")
async def proxy_by_model(request: Request):
    data = await request.json()
    model_name = data.get("model", "")
    backend_url = resolve_by_model(model_name)
    return await _forward(data, backend_url=backend_url, route_label=model_name or "default")
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
![alt text](image-13.png)
