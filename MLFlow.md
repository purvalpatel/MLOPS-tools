# MlFlow
```
Github + Experiment tracer + Model registry for ML
```
- MLFlow is an opensource platform for managing machine learning lifecycle.
- It helps to track experiments, Compare model performance, save models,  manage versions, and support deployment.
- Imagine you are building model with predict house prices. You train models with different algorithms and parameters.
- **Without MLFlow** you might record everything manually in notebooks or spreadsheets.
- **With MLFlow** you can automatically track:
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

### Components of MLFlow:
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

### In-short MLFlow used for:
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


### Demo Setup of training to deployment using MLFlow:

#### Prepare ubuntu environment
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
#### Install MLflow and the required ML libraries:
```
pip install mlflow scikit-learn pandas numpy
```
Verify the installation:
```
mlflow --version
```

#### Start the mltracking server:
```
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts
```
> Here, mlflow.db database will be created in current location. mlartifacts directory will also created in current location.

Open in browser:
```
http://localhost:5000
```

#### Sample training program: `train.py`
```
import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri("http://127.0.0.1:5000")

EXPERIMENT_NAME = "breast-cancer-classification"
MODEL_NAME = "breast-cancer-classifier"

mlflow.set_experiment(EXPERIMENT_NAME)


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

print("Dataset shape:", X.shape)
print("Number of features:", X.shape[1])


# --------------------------------------------------
# 3. Split dataset
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)


# --------------------------------------------------
# 4. Define model parameters
# --------------------------------------------------

params = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": 42,
    "class_weight": "balanced",
}


# --------------------------------------------------
# 5. Start MLflow run and train model
# --------------------------------------------------

with mlflow.start_run(run_name="random-forest-v1") as run:

    print("MLflow Run ID:", run.info.run_id)

    model = RandomForestClassifier(**params)

    model.fit(X_train, y_train)

    # Predict on test dataset
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------
    # 6. Log parameters
    # --------------------------------------------------

    mlflow.log_params(params)

    mlflow.log_param("dataset", "breast_cancer")
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("algorithm", "RandomForestClassifier")

    # --------------------------------------------------
    # 7. Log metrics
    # --------------------------------------------------

    mlflow.log_metrics({
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    })

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    # --------------------------------------------------
    # 8. Log and register model
    # --------------------------------------------------

    signature = infer_signature(
        X_train,
        model.predict(X_train),
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        input_example=X_train.iloc[:3],
        registered_model_name=MODEL_NAME,
        skops_trusted_types=["sklearn.tree._tree.Tree"],
    )

    print("Model URI:", model_info.model_uri)

    print("Model registered successfully.")

    print("Registered model name:", MODEL_NAME)

    print("Training completed.")
```

Run Training job:
```
python3 train.py
```
#### Open MLFlow WebUI : 

http://0.0.0.0:5000

You can see the Experiment inside MLFlow Webui.

<img width="2390" height="458" alt="image" src="https://github.com/user-attachments/assets/d68aceb4-e447-4121-a833-668d67fc68d8" />

<img width="2393" height="914" alt="image" src="https://github.com/user-attachments/assets/b1a1c556-3fa1-490e-adfc-1f866e688830" />


#### Serve the model.
Now your model is registered with MLFlow, Next step is to serve the registered model as an API.so another application can send input data and received predictions.

For your current breast cancer classification project, I recommend this progression:

1. **Deploy locally**: Serve the registered model through an HTTP REST API.
2. **Test the API**: Send sample data and verify that predictions are returned.
3. **Containerize it**: Package the model-serving application with Docker.
4. **Deploy to Kubernetes**: When you're ready, expose the model through a Kubernetes Service or Ingress.

#### Identify registered model name.
- Open web UI of MLFlow
- Models -> find the registered model.
    - Model Name : breast-cancer-classifier
    - Version: 1

<img width="2398" height="854" alt="image" src="https://github.com/user-attachments/assets/3b8422c3-fb4f-4df7-a5e6-4b186d50fc82" />

#### Start Serving model
```
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
mlflow models serve \
  --model-uri "models:/breast-cancer-classifier/1" \
  --host 127.0.0.1 \
  --port 5001 \
  --env-manager local
```
MLflow will load the registered model and start an inference server. The prediction endpoint will be available at http://127.0.0.1:5001/invocations.

#### Verify the Deployment:
```
curl http://127.0.0.1:5001/health
```
You can also check the version:
```
curl http://127.0.0.1:5001/version
```

#### Send prediction Request:
```PYTHON
import requests
from sklearn.datasets import load_breast_cancer

# Load a sample with the same features used during training
dataset = load_breast_cancer(as_frame=True)
sample = dataset.data.iloc[[0]]

# Send the sample to the deployed model
response = requests.post(
    "http://127.0.0.1:5001/invocations",
    json={
        "dataframe_split": {
            "columns": sample.columns.tolist(),
            "data": sample.values.tolist()
        }
    },
    timeout=30
)

print("HTTP status:", response.status_code)
print("Prediction:", response.text)
```

#### Understand what we have deployed:
```
MLflow Tracking Server
  http://127.0.0.1:5000
          |
          v
MLflow Model Registry
  breast-cancer-classifier
  Version 1
          |
          v
MLflow Model Serving
  http://127.0.0.1:5001
          |
          v
  POST /invocations
          |
          v
      Prediction
```

#### What should you do next?

After confirming that the API works, you can containerize the model-serving application with Docker and then deploy it to Kubernetes.

### Containerize your MLflow model with Docker.
Since you've already registered your breast cancer classification model in MLflow and started serving it on port 5001, the next step is to package the model-serving application into a Docker container.

use the architecture:
```
MLflow Tracking Server
        |
        v
  Registered Model
        |
        v
  Docker Container
  +-----------------------+
  | MLflow Model Serving  |
  |                       |
  | Port 5001             |
  | POST /invocations     |
  +-----------------------+
        |
        v
   Model Predictions
```

Now will create self serving image of the model.

#### Build docker image:
```
mlflow models build-docker \
  --model-uri "models:/breast-cancer-classifier/1" \
  --name "breast-cancer-api" \
  --env-manager local
```

Verify the created image:
```
docker images
```
Run the container:
```
docker run -d \
  --name breast-cancer-serving \
  -p 5001:8080 \
  breast-cancer-api
```
### RBAC in MLFlow:

RBAC Allowing you to control who can view experiments, log runs, modify registered models, and manage permissions.

```
pip install 'mlflow[auth]'
```
Enable Authentication:
```
export MLFLOW_FLASK_SERVER_SECRET_KEY="$(python -c 'import secrets; print(secrets.token_hex(32))')"

export MLFLOW_AUTH_ADMIN_PASSWORD='Replace-With-A-Strong-Password'
```

Then restart MLflow using the same database and artifact settings:
```
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlartifacts \
  --app-name basic-auth
```
The authentication system seperately stored into `basic_auth.db`.

After the administrator account created successfully, remove the initial password  from the environment.
```
unset MLFLOW_AUTH_ADMIN_PASSWORD
```


Create users and assign roles:
```PYTHON
from mlflow.server import get_app_client

tracking_uri = "http://127.0.0.1:5000"

auth_client = get_app_client(
    "basic-auth",
    tracking_uri=tracking_uri,
)

# Create users
auth_client.create_user(
    username="data_scientist",
    password="DataScientist-Strong-Password",
)

auth_client.create_user(
    username="model_approver",
    password="ModelApprover-Strong-Password",
)

print("Users created successfully")
```

Assign roles:
```PYTHON
from mlflow.server import get_app_client

tracking_uri = "http://127.0.0.1:5000"

auth_client = get_app_client(
    "basic-auth",
    tracking_uri=tracking_uri,
)

# Create a role for users who can edit the experiment.
scientist_role = auth_client.create_role(
    name="data-scientist",
    workspace="default",
    description="Can edit the designated ML experiment",
)

auth_client.add_role_permission(
    role_id=scientist_role.id,
    resource_type="experiment",
    resource_pattern="YOUR_EXPERIMENT_ID",
    permission="EDIT",
)

auth_client.assign_role(
    username="data_scientist",
    role_id=scientist_role.id,
)

# Create a role for managing the registered model.
approver_role = auth_client.create_role(
    name="model-approver",
    workspace="default",
    description="Can manage the designated registered model",
)

auth_client.add_role_permission(
    role_id=approver_role.id,
    resource_type="registered_model",
    resource_pattern="breast-cancer-classifier",
    permission="MANAGE",
)

auth_client.assign_role(
    username="model_approver",
    role_id=approver_role.id,
)

print("Roles and permissions configured")
```

#### Test RBAC
Test without credentials:
```
curl -i http://127.0.0.1:5000/
```
Test with Credentials:
```
curl -i \
  -u 'data_scientist:DataScientist-Strong-Password' \
  http://127.0.0.1:5000/
```

> We can assign permissions to specific Experiment ID.

You want three roles:

 | Role | Permission | Responsibility |
|---|---|---|
| Data Scientist | `EDIT` | View experiments and update runs or model resources within the granted scope |
| Model Approver | `MANAGE` | Manage the specific registered model, including its permissions |
| Platform Admin | Administrator privileges | Manage users, roles, and platform-wide access |


## Deployment mlflow with Docker-compose

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
