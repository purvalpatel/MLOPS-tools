# Langfuse

LLM Engineering platform focused on:
1. Tracing
2. Observability
3. evaluation
4. Prompt management
5. Metrics
6. debugging

## Tracing:
Distributed tracing for LLM pipelines.

It helps you track:

- prompts + outputs
- latency
- errors
- token usage
- multi-step workflows (retrieval → model → post-processing)

| Area                 | Langfuse                        | Phoenix                           |
| -------------------- | ------------------------------- | --------------------------------- |
| Primary focus        | Production tracing & monitoring | Evaluation & debugging            |
| Best use             | Live systems (APIs, pipelines)  | Offline analysis, experiments     |
| Data model           | Traces, spans, generations      | Datasets, evals, embeddings       |
| Production readiness | Strong                          | Moderate (more research-oriented) |

```
Langfuse = Datadog / Jaeger for LLM pipelines
Phoenix  = Jupyter + evaluation lab for LLMs
```
> Langfuse only becomes useful whenyou integrate it into your application via SDK. <br>
> Not Meant for General microservice tracing. <br>
> Only for LLM/AI Observability. <br>

```
Prompt + Token + Latency + Output + Trace + Spans | Errors + evaluation.
```

# Setup langfuse
### Add helm repo.
```
kubectl create namespace langfuse
helm repo add langfuse https://langfuse.github.io/langfuse-k8s
helm repo update
```

### Values.yaml
```
langfuse:
  nextauth:
    secret:
      value: "nuvoai@123"
  salt:
    value: "d8fK3j9slPz8Jk29sdf8sdf9sdf9sdFsd9f=="
  env:
    NEXTAUTH_URL: "http://langfuse.local"

s3:
  deploy: false                          # ← don't deploy internal MinIO
  storageProvider: "s3"
  bucket: "numol"
  region: "us-east-1"
  endpoint: "http://mns3006.merai.cloud"
  forcePathStyle: true
  accessKeyId:
    value: "0B3AN861NURCZ5Gfd1T61WTxxxxxx"        # ← nested under value:
  secretAccessKey:
    value: "_pUoc4IqCbYTReX2LC87w7KU8KdgfsdffjgTS11gFK83id50A408"   # ← nested under value:

minio:
  enabled: false

clickhouse:
  enabled: true
  auth:
    username: langfuse
    password: "nuvoai@123"
    database: langfuse

redis:
  enabled: true
  auth:
    enabled: false

postgresql:
  enabled: true
  auth:
    username: langfuse
    password: "xxxxx@123"
    database: langfuse
```
Apply:
```
helm install langfuse langfuse/langfuse   -n langfuse   -f values.yaml
```
verify:
```
kubectl get all -n langfuse
```
## Open langfuse web ui:
Edit service to NodePort:
```
kubectl edit svc langfuse-web -n langfuse
```

### Troubleshooting:
upgrade : `helm upgrade langfuse langfuse/langfuse -n langfuse -f values.yaml` <br>
check rollout status: `kubectl rollout status deploy/langfuse-web -n langfuse` <br>
Show helm values: `helm show values langfuse/langfuse | grep -A 30 "^s3:"` <br>
Get env variables : `kubectl exec -n langfuse deploy/langfuse-web -- env | grep -E "S3|AWS"` <br>

## Create Project in langfuse.

New project - test/test
```
Organization - test
project - test

LANGFUSE_SECRET_KEY="sk-lf-fafa98da-25dd-44e9-991b-72dd195ad241"
LANGFUSE_PUBLIC_KEY="pk-lf-138ea7c6-5288-4a02-895b-667cb062e262"
LANGFUSE_BASE_URL="http://10.10.110.53:32217"

```


# Integration in VLLM:

- This integrates 
    - Phoenix
    - Store logs into  file for evidently (/data/kubernetes-nfs-storage/hf-cache/llm_logs.jsonl)
    - Langfuse (LLM tracing)
  
### Fastapi - `app.py`
```
from fastapi import FastAPI, Request, HTTPException
import requests
import time
import os
import json

# Logging
import logging
from logging.handlers import RotatingFileHandler

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.trace import SpanKind
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# OpenInference
from openinference.semconv.trace import SpanAttributes

# Langfuse v2
from langfuse import Langfuse

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
# Langfuse setup (pinned to v2 API)
# ---------------------------------------------------
_langfuse_enabled = os.getenv("LANGFUSE_ENABLED", "true").lower() == "true"

langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
    host=os.getenv("LANGFUSE_HOST", "http://langfuse-web.langfuse.svc.cluster.local:3000"),
) if _langfuse_enabled else None

# Verify credentials at startup — disables langfuse instead of crashing if wrong
if langfuse:
    try:
        langfuse.auth_check()
        print("✅ Langfuse auth OK")
    except Exception as e:
        print(f"❌ Langfuse auth FAILED: {e}")
        print("   Langfuse disabled — check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST")
        langfuse = None

# ---------------------------------------------------
# Logging setup
# ---------------------------------------------------
LOG_FILE = os.getenv("LOG_FILE", "/data/llm_logs.jsonl")

logger = logging.getLogger("llm_logger")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(LOG_FILE, maxBytes=50 * 1024 * 1024, backupCount=5)
handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(handler)

# ---------------------------------------------------
# Model routing
# ---------------------------------------------------
PATH_ROUTES = {
    "gemma3-27b":   os.getenv("GEMMA3_URL",    "http://gemma3-27b-service:8000"),
    "kimi":         os.getenv("KIMI_URL",       "http://kimi-service:8000"),
    "llama3.3-70b": os.getenv("LLAMA_URL",      "http://llama-service:8000"),
    "nucurate":     os.getenv("NUCURATE_URL",   "http://nucurate-model-service:8000"),
    "qwen3.5-27b":  os.getenv("QWEN_URL",       "http://qwen3-5-svc:8000"),
    "gpt-oss":      os.getenv("GPT_OSS_URL",    "http://gpt-oss-svc:8000"),
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
async def _forward(
    data,
    backend_url,
    route_label,
    endpoint="/v1/chat/completions",
    user_id="anonymous",
    session_id=None,
):
    model_name = data.get("model", route_label)
    full_url = f"{backend_url.rstrip('/')}{endpoint}"

    print(f"[PROXY] route={route_label} user={user_id} → {full_url}")

    # ---------------- NORMALIZE INPUT ----------------
    messages = data.get("messages")
    prompt   = data.get("prompt")

    if endpoint == "/v1/completions":
        if not prompt and messages:
            prompt = " ".join([str(m.get("content", "")) for m in messages])
        if isinstance(prompt, list):
            prompt = " ".join(prompt)
        data = {"model": model_name, "prompt": prompt}
        input_payload = [{"role": "user", "content": prompt or ""}]
    else:
        if not messages and prompt:
            messages = [{"role": "user", "content": prompt}]
        input_payload = messages or []

    # ---------------- LANGFUSE TRACE (v2 API) ----------------
    lf_trace = langfuse.trace(
        name=f"vllm-{route_label}",
        user_id=user_id,
        session_id=session_id,
        input=input_payload,
        metadata={
            "route":    route_label,
            "backend":  backend_url,
            "endpoint": endpoint,
            "model":    model_name,
        },
    ) if langfuse else None

    lf_generation = lf_trace.generation(
        name="inference",
        model=model_name,
        input=input_payload,
        metadata={"endpoint": endpoint},
    ) if lf_trace else None

    # ---------------- PHOENIX TRACE ----------------
    with tracer.start_as_current_span(f"vllm-{route_label}-call", kind=SpanKind.CLIENT) as span:

        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, json.dumps(input_payload))

        for i, msg in enumerate(input_payload):
            span.set_attribute(f"llm.input_messages.{i}.message.role",    msg.get("role", "user"))
            span.set_attribute(f"llm.input_messages.{i}.message.content", str(msg.get("content", "")))

        start = time.time()

        # ---------------- CALL vLLM ----------------
        try:
            response = requests.post(full_url, json=data, timeout=120)
            latency  = time.time() - start
            response.raise_for_status()
            output = response.json()

        except Exception as e:
            latency = time.time() - start

            span.set_attribute("error", True)
            span.set_attribute("error.message", str(e))
            span.set_attribute("llm.latency", latency)

            if lf_generation:
                lf_generation.end(
                    level="ERROR",
                    status_message=str(e),
                    usage={"input": 0, "output": 0, "total": 0},
                )
            if lf_trace:
                lf_trace.update(output={"error": str(e)})
            if langfuse:
                langfuse.flush()

            raise HTTPException(status_code=500, detail=str(e))

        # ---------------- OUTPUT ----------------
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, json.dumps(output)[:2000])

        response_text = ""
        for i, choice in enumerate(output.get("choices", [])):
            msg  = choice.get("message", {})
            text = msg.get("content") or choice.get("text", "")
            response_text = text
            span.set_attribute(f"llm.output_messages.{i}.message.role",    msg.get("role", "assistant"))
            span.set_attribute(f"llm.output_messages.{i}.message.content", text)

        # ---------------- TOKENS ----------------
        usage = output.get("usage")
        if usage:
            pt = usage.get("prompt_tokens",     0)
            ct = usage.get("completion_tokens", 0)
            tt = usage.get("total_tokens",      0)
        else:
            prompt_text = input_payload[0]["content"] if input_payload else ""
            pt, ct, tt  = estimate_tokens(prompt_text, response_text)

        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_PROMPT,     pt)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_COMPLETION,  ct)
        span.set_attribute(SpanAttributes.LLM_TOKEN_COUNT_TOTAL,       tt)
        span.set_attribute("llm.latency", latency)

        # ---------------- LANGFUSE END ----------------
        if lf_generation:
            lf_generation.end(
                output=response_text,
                usage={"input": pt, "output": ct, "total": tt},
                metadata={"latency_seconds": round(latency, 4)},
            )
        if lf_trace:
            lf_trace.update(output=response_text)
        if langfuse:
            langfuse.flush()

        # ---------------- FILE LOGGING ----------------
        try:
            log_entry = {
                "timestamp":  time.time(),
                "model":      model_name,
                "route":      route_label,
                "backend":    backend_url,
                "endpoint":   endpoint,
                "user_id":    user_id,
                "session_id": session_id,
                "input":      input_payload,
                "output":     output,
                "latency":    latency,
                "tokens":     {"prompt": pt, "completion": ct, "total": tt},
            }
            logger.info(json.dumps(log_entry))
        except Exception as log_err:
            print("⚠️ Logging failed:", str(log_err))

    return output


# ---------------------------------------------------
# Path routing  — /{service}/v1/chat/completions
# ---------------------------------------------------
@app.post("/{service}/v1/chat/completions")
async def proxy_by_path(service: str, request: Request):
    backend_url = PATH_ROUTES.get(service)
    if not backend_url:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")

    data       = await safe_json(request)
    user_id    = request.headers.get("x-user-id",    "anonymous")
    session_id = request.headers.get("x-session-id", None)

    return await _forward(data, backend_url, service,
                          user_id=user_id, session_id=session_id)


# ---------------------------------------------------
# Chat fallback  — /v1/chat/completions
# ---------------------------------------------------
@app.post("/v1/chat/completions")
async def proxy_chat(request: Request):
    data       = await safe_json(request)
    user_id    = request.headers.get("x-user-id",    "anonymous")
    session_id = request.headers.get("x-session-id", None)
    backend    = resolve_by_model(data.get("model", ""))

    return await _forward(data, backend, data.get("model", "default"),
                          user_id=user_id, session_id=session_id)


# ---------------------------------------------------
# Completions endpoint  — /v1/completions  (nucurate)
# ---------------------------------------------------
@app.post("/v1/completions")
async def proxy_completion(request: Request):
    data       = await safe_json(request)
    user_id    = request.headers.get("x-user-id",    "anonymous")
    session_id = request.headers.get("x-session-id", None)
    backend    = resolve_by_model(data.get("model", ""))

    return await _forward(data, backend, data.get("model", "default"),
                          endpoint="/v1/completions",
                          user_id=user_id, session_id=session_id)

```
### Dockerfile : `Dockerfile`
```
FROM python:3.10-slim

## new added 
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv


WORKDIR /app

COPY requirements.txt .
# below line changed - download from uv
RUN uv pip install --system --no-cache  -r requirements.txt
#RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build image:
```
sudo docker build -t docker.merai.app/devops/llama-llm-proxy:langfuse-1 .
sudo docker push docker.merai.app/devops/llama-llm-proxy:langfuse-1
```

- This image contains the integration of langfuse.


### Deployment YAML - `Fast-api-deployment.yaml`
```
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
        image: docker.merai.app/devops/llama-llm-proxy:langfuse-1
#        image: docker.merai.app/devops/llama-llm-proxy:log-0.2
        imagePullPolicy: Always
        ports:
        - containerPort: 8000

        resources:
          requests:
            cpu: "250m"
            memory: "256Mi"
          limits:
            cpu: "3"
            memory: "10Gi"


        # 🔥 Mount for logs - This is for looging the calls which we will use in evidently
        volumeMounts:
        - name: logs
          mountPath: /data

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
        - name: QWEN3_URL
          value: "http://qwen3-32b-svc:8000"
        - name: GPT_OSS_URL
          value: "http://gpt-oss-svc:8000"

        # Langfuse — v2 SDK reads LANGFUSE_HOST from constructor arg
        # and also auto-reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY from env
        - name: LANGFUSE_HOST
          value: "http://langfuse-web.langfuse.svc.cluster.local:3000"
        - name: LANGFUSE_PUBLIC_KEY
          value: "pk-lf-138ea7c6-5288-4a02-895b-667cb062e262"
        - name: LANGFUSE_SECRET_KEY
          value: "sk-lf-fafa98da-25dd-44e9-991b-72dd195ad241"


      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: hf-cache-pvc
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
![alt text](image-28.png)

- Traces stored on S3: `s3://xxxx/cmo9yoahr0004za076c8u8wz0`

![alt text](vllm-deployment-22Apr.png)
