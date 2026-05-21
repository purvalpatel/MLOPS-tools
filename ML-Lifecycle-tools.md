# Complete MLOps Lifecycle With Common Tools

For modern AI/LLM systems, MLOps usually looks like this:
```
Data
  ↓
Training
  ↓
Experiment Tracking
  ↓
Model Registry
  ↓
CI/CD
  ↓
Deployment
  ↓
Inference
  ↓
Observability
  ↓
Monitoring
  ↓
Retraining
```

### 1. Data Collection & Storage
Store:
```
datasets
embeddings
logs
prompts
feedback
evaluation data
```

#### Common Tools
| Purpose        | Tools                                                                                                                                                            | status |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | --- |
| Object storage | MinIO, Amazon S3                                               | ✅ |
| Data lake      | Delta Lake                                                      | ✅ |
| Databases      | PostgreSQL, MongoDB                                       | ✅ |
| Vector DB      | Milvus, Qdrant, Weaviate                                         | ✅ |

### 2. Data Versioning

Track dataset changes.

Example:
```
training dataset v1
cleaned dataset v2
RLHF dataset v3
```

#### Tools:
| Tool                                                                                 | Purpose                      | status |
| ------------------------------------------------------------------------------------ | ---------------------------- | -- |
| DVC                                      | Data version control         | -- |
| LakeFS                                   | Git-like versioning for data | -- |
| Weights & Biases Artifacts | Dataset lineage              | -- |


### 3. Experiment Tracking
Track:
```
hyperparameters
metrics
benchmarks
training runs
```

#### Tools:

| Tool                                                        | Purpose                | status |
| ----------------------------------------------------------- | ---------------------- | -- |
| MLflow         | Experiment tracking    | -- |
| Weights & Biases | Training visualization | -- |
| Neptune AI     | Experiment metadata    | -- |

### 4. Model Training

Train/fine-tune models.

#### Tools
| Area                      | Tools                      | status |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- | -- |
| Distributed training      | PyTorch                                                       | ✅ |
| LLM fine-tuning           | Hugging Face Transformers                 | ✅ |
| Distributed orchestration | Ray, Kubeflow |   ✅ |
| GPU scheduling            | Kubernetes                                                                                                    | ✅ |


### 5. Model Registry

Manage model versions.

Example:
```
llama3-chat
 ├── v1
 ├── v2
 └── production
```

#### Tools:

| Tool                                                                                         | Purpose       | status |
| -------------------------------------------------------------------------------------------- | ------------- | -- |
| MLflow Registry | Versioning    | -- |
| Weights & Biases Registry             | Registry      | -- |
| Hugging Face Hub                        | Model hosting | ✅ |

### 6. CI/CD For ML
Automate:
```
testing
packaging
deployments
rollbacks
```
#### Tools:
| Tool                                                                                                            | Purpose              | status |
| --------------------------------------------------------------------------------------------------------------- | -------------------- |  -- |
| GitHub Actions                                   | CI/CD                |  ✅ |
| GitLab CI/CD | Pipelines            | ✅ |
| Argo CD                                                | GitOps               | ✅ | 
| Tekton                                                         | Kubernetes-native CI | -- |


### 7. Deployment

Serve models in production.

#### Tools:
| Type                | Tools                                                               | status |
| ------------------- | ------------------------------------------------------------------- | -- |
| LLM inference       | vLLM | ✅ |
| Multi-model serving | KServe | ✅ |
| General serving     | Seldon Core       | -- |
| API layer           | FastAPI      | ✅ |


### 8. Traffic Management

Handle:
```
canary rollout
A/B testing
blue/green deployments
```

#### Tools:

| Tool                                                                                           | Purpose           | status |
| ---------------------------------------------------------------------------------------------- | ----------------- | -- |
| Istio                                                                                          | Traffic splitting | ✅ |
| Linkerd                                        | Service mesh      | -- |
| NGINX Ingress Controller | Ingress routing   | ✅ |

### 9. Observability & Tracing
Track:
```
prompts
latency
traces
token usage
failures
```

#### Tools

| Tool                                                             | Purpose             | status |
| ---------------------------------------------------------------- | ------------------- | -- |
| Langfuse          | LLM observability   | ✅ |
| OpenTelemetry | Distributed tracing | ✅ |
| Jaeger   | Trace visualization | ✅ |
| Helicone     | LLM analytics       | -- |


### 10. Infrastructure Monitoring
Monitor:
```
GPU usage
CPU/memory
pod health
token throughput
queue depth
```
#### Tools
| Tool                                                                                   | Purpose           | status |
| -------------------------------------------------------------------------------------- | ----------------- | -- |
| Prometheus                            | Metrics           | ✅ |
| Grafana                               | Dashboards        | ✅ |
| NVIDIA DCGM Exporter | GPU metrics       | ✅ |
| cAdvisor               | Container metrics | ✅ |



### 11. Security & Governance

Control:
```
access
secrets
compliance
audit logs
```

#### Tools:
| Tool                                                                                                  | Purpose            | Status |
| ----------------------------------------------------------------------------------------------------- | ------------------ | -- |
| Vault                                         | Secrets            | ✅ |
| Keycloak                                        | Authentication     | ✅ |
| OPA Gatekeeper <br> ResourceQuota <br> LimitRange | Policy enforcement | ✅ |


12. Retraining & Automation

Automatically:
```
retrain models
evaluate drift
promote new versions
```

#### Tools:
| Tool                                                                                             | Purpose      | status |
| ------------------------------------------------------------------------------------------------ | ------------ | -- |
| Kubeflow Pipelines | ML workflows | -- |
| Apache Airflow                             | Scheduling   | -- |
| Metaflow                                  | ML pipelines | -- |


## LLMOps Stack (Modern AI Inference Platforms)

For your kind of infrastructure:
```
Kubernetes
 ├── Istio
 ├── FastAPI
 ├── vLLM
 ├── Langfuse
 ├── Prometheus
 ├── Grafana
 ├── MLflow
 ├── MinIO
 └── ArgoCD
```
This is effectively a modern LLMOps platform.

Simplified Mapping
| Stage               | Most Popular Tool |
| ------------------- | ----------------- |
| Data storage        | S3 / MinIO        |
| Experiment tracking | MLflow            |
| Training            | PyTorch           |
| Model registry      | MLflow            |
| Deployment          | vLLM              |
| Orchestration       | Kubernetes        |
| Routing             | Istio             |
| Observability       | Langfuse          |
| Monitoring          | Prometheus        |
| Visualization       | Grafana           |
| GitOps              | ArgoCD            |


#### Real-World Division
### DevOps
Handles:
- Kubernetes
- networking
- ingress
- autoscaling
- monitoring

### MLOps
Handles:
- model lifecycle
- experiments
- deployment versions
- retraining
- evaluation

### LLMOps
Handles:
- prompts
- traces
- token analytics
- RAG observability
- inference optimization
