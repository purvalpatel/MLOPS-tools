# MlFlow
```
Github + Experiment tracer + Model registry for ML
```
MLFlow is an opensource platform for managing machine learning lifecycle.

It helps to track experiments, Compare model performance, save models,  manage versions, and support deployment.

Imagine you are building model with predict house prices. You train models with different algorithms and parameters.

**Without MLFlow** you might record everything manually in notebooks or spreadsheets.

**With MLFlow** you can automatically track:
- Which algorithm you used.
- Which parameters you selected.
- How well model performed.
- Which Model Files were generated.
- Which experiment produced the same result.

```
Experiment, Algorithm, Learning rate, Accuracy
                |
            Logistic regression, Random forest, Gradient boosting
```
With MLFlow you can log these values and compare the experiments through  web-based interface.

Components of MLFlow:
- Tracking  - Parameters, Metrics, Artifacts
- Projects
- Models    - standard format for packaging models
- Model Registry    - Manage model versions.

It helps track:
- Experiments
- Parameters
- Versions
- Metrics
- Models
- deployments

Implemented project - https://github.com/purvalpatel/Sample-mlops-project <br>

- Experiment tracking + Model Registry.
- Log metrics
- Log Parameters
- Version models
- Register models 
- Deploy models

```
Try 10 models -> Compare -> Pick best -> Deploy
```
### Production stack:

- MLflow + Kubernetes : scalable deployment
- MLflow + GPUs :	deep learning
- MLflow + DVC	: dataset versioning
- MLflow + ArgoCD	: GitOps ML
- MLflow + vLLM :	LLM serving
- MLflow + LoRA	: fine-tuning tracking


### Prepare ubuntu environment

```
mkdir mlflow-tutorial
cd mlflow-tutorial
```

Create a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```
Upgrade pip:
```
python -m pip install --upgrade pip
```
Install MLflow and the required ML libraries:
```
pip install mlflow scikit-learn pandas numpy
```
Verify the installation:
```
mlflow --version
```

Start the MLTracking server:
```
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```

## Deployment MLFLow

Deployment `mlflow-deployment.yaml`
```YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mlflow
  namespace: mlflow
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mlflow
  template:
    metadata:
      labels:
        app: mlflow
    spec:
      containers:
        - name: mlflow
          image: ghcr.io/mlflow/mlflow:latest
          args:
            [
              "mlflow",
              "server",
              "--host",
              "0.0.0.0",
              "--port",
              "5000",
              "--default-artifact-root",
              "file:/mlflow/artifacts"
            ]
          ports:
            - containerPort: 5000
          volumeMounts:
            - name: mlflow-artifacts
              mountPath: /mlflow/artifacts
      volumes:
        - name: mlflow-artifacts
          persistentVolumeClaim:
            claimName: mlflow-pvc

```

PVC `mlflow-pvc.yaml`
```YAML
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mlflow-pvc
  namespace: mlflow
spec:
  accessModes:
    - ReadWriteMany
  storageClassName: k8s-nfs-storage
  resources:
    requests:
      storage: 1Gi
```

Service `mlflow-service.yaml`
```YAML
apiVersion: v1
kind: Service
metadata:
  name: mlflow-service
  namespace: mlflow
spec:
  type: NodePort
  selector:
    app: mlflow
  ports:
    - port: 5000
      targetPort: 5000
      nodePort: 31555
```
