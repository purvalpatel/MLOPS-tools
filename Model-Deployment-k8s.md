Simple model deployment with kubernetes:
--------------------------------------
Here is the simplest, cleanest possible example of deploying a Hugging Face model on Kubernetes without Triton or complex steps. <br>

We will use: <br>
✅ Model: distilbert-base-uncased <br>
A tiny, fast text classifier model from HuggingFace. <br>

We will run it using:<br>
✅ Framework: FastAPI + Transformers<br>
✅ Container: custom Docker image<br>
✅ Kubernetes Deployment + Service<br>

### Step 1 - install dependencies:
```
# install python3.10-venv package if not installed
apt install python3.10-venv

# create virtual environment
python3 -m venv hf-venv
source hf-venv/bin/activate

# install huggingface-cli - If want to download model manually. (not required in this example. )
pip install --upgrade huggingface_hub
hf download distilbert/distilbert-base-uncased
```

### Step 2 - Create project folder:
```
mkdir hf-k8s-model
cd hf-k8s-model
```

### Step 3 - Create app.py (FastAPI inference server)
```
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Load HF model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(req: Request):
    out = classifier(req.text)
    return {"result": out}

```

### Step 4 - Create requirements.txt
```
fastapi
uvicorn
transformers
torch
```

### Step 5 - Create Dockerfile:
Here the model is downloaded from huggingface. thats why we have not downloaded this in step 1.
```
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Download the model during build (optional)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased')"

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Step 6 - Build and push image on dockerhub.
```
docker build -t <your-dockerhub-user>/hf-demo:latest .
docker push <your-dockerhub-user>/hf-demo:latest
```

### Step 7 - Create kubernetes deployment
Create file : hf-model.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hf-model
  template:
    metadata:
      labels:
        app: hf-model
    spec:
      containers:
      - name: hf-container
        image: <your-dockerhub-user>/hf-demo:latest
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: hf-model-service
spec:
  type: NodePort
  selector:
    app: hf-model
  ports:
    - port: 8000
      targetPort: 8000
      nodePort: 30080
```
### Deploy:
```
kubectl apply -f hf-model.yaml
kubectl get pods -w
```

### Test model:
```
kubectl get nodes -o wide
```

```
curl -X POST http://<NODE-IP>:30080/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"This is amazing!"}'
```

Your HuggingFace model is now running in Kubernetes! <br>
Simple. No Triton. No GPU requirement.


Auto-scaling using CPU (HPA):
-----------------------------
Step 1 - Create : deployment.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hf-model
spec:
  replicas: 1
  selector:
    matchLabels:
      app: hf-model
  template:
    metadata:
      labels:
        app: hf-model
    spec:
      containers:
      - name: hf-model
        image: yourrepo/hf-sentiment:1
        ports:
        - containerPort: 8000
        resources:
          requests:
            cpu: "200m"
            memory: "512Mi"
          limits:
            cpu: "1"
            memory: "1Gi"
---
apiVersion: v1
kind: Service
metadata:
  name: hf-model
spec:
  type: ClusterIP
  selector:
    app: hf-model
  ports:
  - port: 80
    targetPort: 8000
```

Step 2 - Create Horizontal pod autoscaler:
```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: hf-model-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: hf-model
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
```

Apply changes:
```
kubectl apply -f deployment.yaml
kubectl apply -f hpa.yaml
```

Now test by generating load:
```
while true; do curl -X POST http://10.96.11.14/predict  -H "Content-Type: application/json"  -d '{"text":"hello"}'; done
```
Note: Here i am using cluster IP from host machine, it will only work for the single node cluster. because it is in the same network. <br>

check scaling:
```
kubectl get hpa
```
If output is like this, <br>
<img width="929" height="67" alt="image" src="https://github.com/user-attachments/assets/f7164b12-b9d2-4661-9c31-498afac93223" />

Then there is an issue. <br>
⚠️ HPA cannot read CPU metrics <br>
⚠️ Your cluster does NOT have metrics-server installed <br>
⚠️ So HPA will NEVER autoscale <br>


### Install metrics server:
```
kubectl top pod
```
If it is not showing, <br>

install metrics server:
```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
```
Edit:
```
kubectl -n kube-system edit deployment metrics-server
```
Add this under spec.template.spec.containers[0].args:
```
        - --kubelet-insecure-tls
        - --kubelet-preferred-address-types=InternalIP
```
Then restart,
```
kubectl rollout restart deployment metrics-server -n kube-system
```
Now check,
```
kubectl top pod
```
<img width="656" height="73" alt="image" src="https://github.com/user-attachments/assets/cbc298b6-8d70-4616-8273-b2190f9b49ec" />

Now verify Horizontal pod autoscaler: <br>
```
kubectl get hpa
```
<img width="890" height="68" alt="image" src="https://github.com/user-attachments/assets/ebe4075b-6d27-40dc-a30b-844bfbb0bfe7" />


<br>
Now according to this your pods will scale: <br>
<img width="639" height="272" alt="image" src="https://github.com/user-attachments/assets/d5cb1450-7fcb-4274-86db-753e5ec3f58c" />


Auto-scaling with prometheus
-----------------------------

1. Application -> Exposing prometheus metrics from application. app.py, /metrics
2. Prometheus scraps it --> ServiceMonitor  tell Prometheus how to scrape metrics from your application.
3. Prometheus adapter --> HPA Cant reads metrices from Prometheus, You must **install Prometheus adapter** into  prometheus and configure it. it expose metrics under **custom metrics** of kubernetes. its a bridge between **promethes** and **k8s custom metrics** api. 
4. HPA reads metrics --> HPA reads metrics from Prometheus adapter.
5. Scale pods

So the flow will be: <br>
```
[Your app] -> [Prometheus] -> [Prometheus Adapter] -> [Kubernetes HPA] -> [Scaling] 
```

## 1. Building application FastAPI.
app.py
```
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from prometheus_client import Counter, make_asgi_app

app = FastAPI()
requests_total = Counter("requests_total", "Total API Requests")

# Load HF model
classifier = pipeline("sentiment-analysis", model="distilbert-base-uncased")

class Request(BaseModel):
    text: str

@app.post("/predict")
def predict(req: Request):
    requests_total.inc()
    out = classifier(req.text)
    return {"result": out}

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```
requirements.txt
```
fastapi
uvicorn
transformers
torch
prometheus-client
```

Build docker image: <br>
Dockerfile:
```
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app.py .

# Download the model during build (optional)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='distilbert-base-uncased')"

# Expose port
EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```
Build and push dockerimage:
```
docker build -t <your-dockerhub-user>/hf-demo:latest .
docker push <your-dockerhub-user>/hf-demo:latest
```
## 2. Now create deployment: <br>
deployment.yaml
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fastapi-sentiment
  labels:
    app: fastapi-sentiment
spec:
  replicas: 1
  selector:
    matchLabels:
      app: fastapi-sentiment
  template:
    metadata:
      labels:
        app: fastapi-sentiment
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: fastapi-app
        image: docker.linux.app/harshal/hf-model:0.2
        ports:
        - containerPort: 8000
        env:
        - name: PYTHONUNBUFFERED
          value: "1"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /docs
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /docs
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fastapi-sentiment-service
  labels:
    app: fastapi-sentiment
spec:
  selector:
    app: fastapi-sentiment
  ports:
  - name: http
    port: 8000
    targetPort: 8000
    protocol: TCP
  type: ClusterIP
```

Apply changes:
```
kubectl apply -f deployment.yaml
```

## 3. Create servicemonitor to tell prometheus to scrap metrics from application:
```
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: fastapi-sentiment-monitor
  namespace: monitoring
  labels:
    release: prometheus
spec:
  namespaceSelector:
    matchNames:
    - default
  selector:
    matchLabels:
      app: fastapi-sentiment
  endpoints:
    - port: http        # MUST match the service port name
      path: /metrics
      interval: 15s
```
apply:
```
kubectl apply -f servicemonitor.yaml
```

Verify the metrics are exposing in metrics or not:
```
curl http://<clusterip>:8000/metrics/
# / is mandadory at the end.
```

## 4. Create prometheus adapter to get metrics from prometheus, and expose it into custom metrics.
prometheus-adapter-values.yaml
```
prometheus:
  url: http://prometheus-kube-prometheus-prometheus.monitoring.svc
rules:
  custom:
    - seriesQuery: 'requests_total{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace:
            resource: namespace
          pod:
            resource: pod
      name:
        matches: "^requests_total$"
        as: "requests"
      metricsQuery: 'sum(rate(requests_total{<<.LabelMatchers>>}[2m])) by (<<.GroupBy>>)'
```
upgrade helm for prometheus-adapter:
```
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter   -f prometheus-adapter-values.yaml -n monitoring
```

## 5. Create Horizontal pod autoscaler:
hpa-request-total.yaml
```
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: fastapi-sentiment-hpa
  namespace: default
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: fastapi-sentiment   # your deployment name
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Pods
    pods:
      metric:
        name: requests
      target:
        type: AverageValue
        averageValue: "0.100"  ## 1 requests/second per pod.

```
Apply:
```
kubectl apply -f hpa-request-count.yaml
```

Verify hpa:
```
kubectl get hpa
```

## 6. Test:
```
while true; do   curl -s http://10.99.219.66:8000/predict -X POST -H "Content-Type: application/json"      -d '{"text":"hello"}' > /dev/null; done
```

Now check the metrics, in that you can see the request_total is increasing:
```
curl http://clusterip:8000/metrics/
```
Now check the HPA:
```
kubectl get hpa
```

<img width="1020" height="70" alt="image" src="https://github.com/user-attachments/assets/d8086268-6c38-4863-b6f6-04219173e76c" />

Here, `9m/10` means: <br>
**CURRENT_VALUE/TARGET_VALUE** <br>
This is your **HPA** Settings.

`m` means **milli-units** **(1/1000)**
10 = 1 unit of the custom metric, **10 requests per second** <br>
1m = 1/1000 of a unit, means **0.001 requests per seconds**. <br>

“Scale up when each pod has more than 10 requests per second.” <br>

<img width="722" height="133" alt="image" src="https://github.com/user-attachments/assets/2c21a594-09be-4ee3-a81f-458b40c2eace" />

### Troubleshooting:

1. Check metrices are showing in **prometheus** or not:
To check the metrics are showing in prometheues you first need to **forward a port** of prometheus so you can check. <br>
```
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-prometheus 9091:9090
```
then check,
```
curl -s http://localhost:9091/api/v1/query?query=requests_total | jq
curl http://localhost:9091/api/v1/label/__name__/values
```

2. Check the custom metrics which is exposed by prometheus-adapter is showing or not.
```
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 
```
If it is showing [] empty then,
```
{"kind":"APIResourceList","apiVersion":"v1","groupVersion":"custom.metrics.k8s.io/v1beta1","resources":[]}
```

restart prometheus-adapter:
```
kubectl -n monitoring delete pod -l app.kubernetes.io/name=prometheus-adapter
# OR
kubectl -n monitoring delete pod -l app=prometheus-adapter

```

Check which prometheus data is getting fetched by prometheus-adapter:
```
kubectl -n monitoring get deploy prometheus-adapter -o yaml | grep -i prometheus -A3 -B3
```
it should be from `monitoring` namespace.
`http://prometheus-kube-prometheus-prometheus.monitoring.svc:9090` that we have mentioned in `prometheus-adapter-values.yaml`.

Now prometheus adapter is getting the data from custom metrics :

```
kubectl get --raw /apis/custom.metrics.k8s.io/v1beta1 | jq
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1/namespaces/default/pods/*/requests"
```
   
Now check hpa,
```
kubectl get hpa
```

Allow Huggingface models:
-------------------------
1. Loging hugging face and on model page fill below details in terms of accepting terms. <br>
<img width="1013" height="811" alt="image" src="https://github.com/user-attachments/assets/7983d4a7-ffa3-4752-86ed-2b1112649b93" />

2. Create Access tokens if not have.
Login hugging face -> Settings -> Access Token <br>
<img width="1680" height="575" alt="image" src="https://github.com/user-attachments/assets/41396988-8419-448e-8f43-c242f4abf6a1" />

Save the token on safe location. <br>

3. Wait for the request to approve.
Settings -> Gated Repository. <br>
<img width="1468" height="850" alt="image" src="https://github.com/user-attachments/assets/85720a31-b954-414b-8c5e-da5c8060c48d" />



    
