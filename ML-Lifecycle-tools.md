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
| Purpose        | Tools                                                                                                                                                            |
| -------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Object storage | [MinIO](https://min.io?utm_source=chatgpt.com), [Amazon S3](https://aws.amazon.com/s3/?utm_source=chatgpt.com)                                                   |
| Data lake      | [Delta Lake](https://delta.io?utm_source=chatgpt.com)                                                                                                            |
| Databases      | [PostgreSQL](https://www.postgresql.org?utm_source=chatgpt.com), [MongoDB](https://www.mongodb.com?utm_source=chatgpt.com)                                       |
| Vector DB      | [Milvus](https://milvus.io?utm_source=chatgpt.com), [Qdrant](https://qdrant.tech?utm_source=chatgpt.com), [Weaviate](https://weaviate.io?utm_source=chatgpt.com) |

### 2. Data Versioning

Track dataset changes.

Example:
```
training dataset v1
cleaned dataset v2
RLHF dataset v3
```

#### Tools:
| Tool                                                                                 | Purpose                      |
| ------------------------------------------------------------------------------------ | ---------------------------- |
| [DVC](https://dvc.org?utm_source=chatgpt.com)                                        | Data version control         |
| [LakeFS](https://lakefs.io?utm_source=chatgpt.com)                                   | Git-like versioning for data |
| [Weights & Biases Artifacts](https://wandb.ai/site/artifacts?utm_source=chatgpt.com) | Dataset lineage              |


### 3. Experiment Tracking
Track:
```
hyperparameters
metrics
benchmarks
training runs
```

#### Tools:

| Tool                                                        | Purpose                |
| ----------------------------------------------------------- | ---------------------- |
| [MLflow](https://mlflow.org?utm_source=chatgpt.com)         | Experiment tracking    |
| [Weights & Biases](https://wandb.ai?utm_source=chatgpt.com) | Training visualization |
| [Neptune AI](https://neptune.ai?utm_source=chatgpt.com)     | Experiment metadata    |

### 4. Model Training

Train/fine-tune models.

#### Tools
| Area                      | Tools                                                                                                         |
| ------------------------- | ------------------------------------------------------------------------------------------------------------- |
| Distributed training      | [PyTorch](https://pytorch.org?utm_source=chatgpt.com)                                                         |
| LLM fine-tuning           | [Hugging Face Transformers](https://huggingface.co/docs/transformers?utm_source=chatgpt.com)                  |
| Distributed orchestration | [Ray](https://www.ray.io?utm_source=chatgpt.com), [Kubeflow](https://www.kubeflow.org?utm_source=chatgpt.com) |
| GPU scheduling            | Kubernetes                                                                                                    |


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

| Tool                                                                                         | Purpose       |
| -------------------------------------------------------------------------------------------- | ------------- |
| [MLflow Registry](https://mlflow.org/docs/latest/model-registry.html?utm_source=chatgpt.com) | Versioning    |
| [Weights & Biases Registry](https://wandb.ai/site/models?utm_source=chatgpt.com)             | Registry      |
| [Hugging Face Hub](https://huggingface.co?utm_source=chatgpt.com)                            | Model hosting |

### 6. CI/CD For ML
Automate:
```
testing
packaging
deployments
rollbacks
```
#### Tools:
| Tool                                                                                                            | Purpose              |
| --------------------------------------------------------------------------------------------------------------- | -------------------- |
| [GitHub Actions](https://github.com/features/actions?utm_source=chatgpt.com)                                    | CI/CD                |
| [GitLab CI/CD](https://about.gitlab.com/stages-devops-lifecycle/continuous-integration/?utm_source=chatgpt.com) | Pipelines            |
| [Argo CD](https://argo-cd.readthedocs.io?utm_source=chatgpt.com)                                                | GitOps               |
| [Tekton](https://tekton.dev?utm_source=chatgpt.com)                                                             | Kubernetes-native CI |


### 7. Deployment

Serve models in production.

#### Tools:
| Type                | Tools                                                               |
| ------------------- | ------------------------------------------------------------------- |
| LLM inference       | [vLLM](https://github.com/vllm-project/vllm?utm_source=chatgpt.com) |
| Multi-model serving | [KServe](https://kserve.github.io/website?utm_source=chatgpt.com)   |
| General serving     | [Seldon Core](https://www.seldon.io?utm_source=chatgpt.com)         |
| API layer           | [FastAPI](https://fastapi.tiangolo.com?utm_source=chatgpt.com)      |


### 8. Traffic Management

Handle:
```
canary rollout
A/B testing
blue/green deployments
```

#### Tools:

| Tool                                                                                           | Purpose           |
| ---------------------------------------------------------------------------------------------- | ----------------- |
| Istio                                                                                          | Traffic splitting |
| [Linkerd](https://linkerd.io?utm_source=chatgpt.com)                                           | Service mesh      |
| [NGINX Ingress Controller](https://kubernetes.github.io/ingress-nginx/?utm_source=chatgpt.com) | Ingress routing   |

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

| Tool                                                             | Purpose             |
| ---------------------------------------------------------------- | ------------------- |
| [Langfuse](https://langfuse.com?utm_source=chatgpt.com)          | LLM observability   |
| [OpenTelemetry](https://opentelemetry.io?utm_source=chatgpt.com) | Distributed tracing |
| [Jaeger](https://www.jaegertracing.io?utm_source=chatgpt.com)    | Trace visualization |
| [Helicone](https://www.helicone.ai?utm_source=chatgpt.com)       | LLM analytics       |


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
| Tool                                                                                   | Purpose           |
| -------------------------------------------------------------------------------------- | ----------------- |
| [Prometheus](https://prometheus.io?utm_source=chatgpt.com)                             | Metrics           |
| [Grafana](https://grafana.com?utm_source=chatgpt.com)                                  | Dashboards        |
| [NVIDIA DCGM Exporter](https://github.com/NVIDIA/dcgm-exporter?utm_source=chatgpt.com) | GPU metrics       |
| [cAdvisor](https://github.com/google/cadvisor?utm_source=chatgpt.com)                  | Container metrics |



### 11. Security & Governance

Control:
```
access
secrets
compliance
audit logs
```

#### Tools:
| Tool                                                                                                  | Purpose            |
| ----------------------------------------------------------------------------------------------------- | ------------------ |
| [Vault](https://www.vaultproject.io?utm_source=chatgpt.com)                                           | Secrets            |
| [Keycloak](https://www.keycloak.org?utm_source=chatgpt.com)                                           | Authentication     |
| [OPA Gatekeeper](https://open-policy-agent.github.io/gatekeeper/website/docs/?utm_source=chatgpt.com) | Policy enforcement |


12. Retraining & Automation

Automatically:
```
retrain models
evaluate drift
promote new versions
```

#### Tools:
| Tool                                                                                             | Purpose      |
| ------------------------------------------------------------------------------------------------ | ------------ |
| [Kubeflow Pipelines](https://www.kubeflow.org/docs/components/pipelines/?utm_source=chatgpt.com) | ML workflows |
| [Apache Airflow](https://airflow.apache.org?utm_source=chatgpt.com)                              | Scheduling   |
| [Metaflow](https://metaflow.org?utm_source=chatgpt.com)                                          | ML pipelines |


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
