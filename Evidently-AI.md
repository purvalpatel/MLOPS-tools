## Evidently AI:
- Tool to monitor data quality, model performance and data drift.
- Monitor and evaluate machine learning models after deployment.
- Library that checks wheather your model is still working correctly in real-world data.

## Why ?
- Training data = clean, controlled
- Production data = messy, changing over time.
- So problem happens like:
    - Data drift (distribution changes over time)
    - Concept drift (relationship between features and target changes)
    - Model performance degradation (accuracy drops)

## What it does:
It continuously monitors:
```
Incoming data 
     ↓
Compare with training/reference data
     ↓
Generate reports | alerts
```
### 👉 Evidently is just a tool that tells:

“Hey, your input data changed” <br>
“Your model behavior is changing” <br>

```
User → vLLM → Output
          ↓
      Save logs
          ↓
   Check quality
          ↓
   Detect problems
```
**Evidently can be used to: It just automates this checking.** <br>

**Phoenix** -> what happens in this request? <br>
**Evidently AI** -> Is my model behaviour chnaging over time ?


## setup:
Lets Assume that this is your current architecture.
```
user
  |
istio ingress gateway
  |
fastAPI proxy  -> Logs -> storage (file/db/kafka)
  |
Models (trace-level observability)
  |
Phoenix
```

### Step 1 -- Set logging into  FastAPI Proxy `app.py`
- Logs stored into /data/llm_logs.jsonl
- /data directory is mounted on pvc. our pvc is `hf-cache-pvc`.

### Step 2 -- Create Docker image of evidently runner.
- Note: Below code is for Single Dataset. <br>
- It will generate single .html report.
- Docker image : `docker.xxxx.xxxx/devops/evidently:0.1`

evidently_runner.py 
- This is for single dataset.
```
"""
Evidently LLM Monitor — fixed for Evidently >= 0.7
=====================================================
Install:
    pip install "evidently>=0.7"

Run report only (no UI):
    python evidently_llm_monitor_v2.py

View in UI after running:
    evidently ui --workspace /data/evidently_workspace
    Then open http://localhost:8000
"""

import os
import pandas as pd
import evidently

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset

print(f"Evidently version: {evidently.__version__}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOG_FILE       = "/data/llm_logs.jsonl"
WORKSPACE_PATH = "/data/evidently_workspace"
PROJECT_NAME   = "LLM Monitoring"
OUTPUT_HTML    = "/data/llm_drift_report.html"

# ─────────────────────────────────────────────
# 1. Load data
# ─────────────────────────────────────────────
df = pd.read_json(LOG_FILE, lines=True)
if df.empty:
    raise ValueError("Input log file is empty")

print(f"Loaded {len(df)} rows. Columns: {df.columns.tolist()}")

# ─────────────────────────────────────────────
# 2. Feature engineering
# ─────────────────────────────────────────────
df["prompt_len"]   = df["input"].apply(lambda x: len(str(x)))
df["response_len"] = df["output"].apply(lambda x: len(str(x)))
df["latency"]      = pd.to_numeric(df.get("latency", 0), errors="coerce").fillna(0)

# tokens is a nested dict — extract safely
if "tokens" in df.columns:
    df["total_tokens"] = df["tokens"].apply(
        lambda x: x.get("total", 0) if isinstance(x, dict) else 0
    )
else:
    df["total_tokens"] = 0

features = ["prompt_len", "response_len", "latency", "total_tokens"]
df = df[features].fillna(0)

# Replace zeros to avoid constant-column issues
df = df.replace(0, 1e-6)

# Drop constant columns (Evidently drift needs variance)
valid_cols = [c for c in df.columns if df[c].nunique() > 1]
if not valid_cols:
    raise ValueError("All columns are constant — cannot compute drift")

df = df[valid_cols]
print(f"Using features: {valid_cols}")

# ─────────────────────────────────────────────
# 3. Split reference / current
# ─────────────────────────────────────────────
split = int(len(df) * 0.7)
if split < 1 or split >= len(df):
    raise ValueError("Not enough data to split (need at least 2 rows)")

reference_df = df.iloc[:split].copy()
current_df   = df.iloc[split:].copy()
print(f"Reference rows: {len(reference_df)} | Current rows: {len(current_df)}")

# ─────────────────────────────────────────────
# 4. Wrap in Evidently Dataset  ← NEW in v0.7
#    DataDefinition replaces ColumnMapping
# ─────────────────────────────────────────────
data_def = DataDefinition(
    numerical_columns=valid_cols  # explicitly mark as numeric
)

reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_def)
current_dataset   = Dataset.from_pandas(current_df,   data_definition=data_def)

# ─────────────────────────────────────────────
# 5. Build and run report  ← NEW in v0.7
#    Report([ preset ])  not  Report(metrics=[preset])
# ─────────────────────────────────────────────
report = Report([DataDriftPreset()])

# run() takes Dataset objects, not raw DataFrames
result = report.run(current_dataset, reference_dataset)

# ─────────────────────────────────────────────
# 6. Save HTML  ← use result.save_html() in v0.7
# ─────────────────────────────────────────────
result.save_html(OUTPUT_HTML)
print(f"HTML report saved → {OUTPUT_HTML}")

# ─────────────────────────────────────────────
# 7. Save to local workspace for the UI
#    workspace.add_run()  replaces  workspace.add_report()
# ─────────────────────────────────────────────
try:
    from evidently.ui.workspace import Workspace

    os.makedirs(WORKSPACE_PATH, exist_ok=True)
    workspace = Workspace.create(WORKSPACE_PATH)

    # Find or create project
    project = None
    for p in workspace.list_projects():
        if p.name == PROJECT_NAME:
            project = p
            break
    if project is None:
        project = workspace.create_project(PROJECT_NAME)
        project.description = "LLM log drift monitoring"
        project.save()

    # add_run() is the correct method in v0.7 (not add_report)
    workspace.add_run(project.id, result)
    print(f"Run logged to workspace → {WORKSPACE_PATH}")
    print(f"View in UI: evidently ui --workspace {WORKSPACE_PATH}")

except Exception as e:
    print(f"[WARN] Could not save to workspace: {e}")
    print("The HTML report was still saved successfully.")
```
Dockerfile:
```
FROM python:3.10-slim

WORKDIR /app

RUN pip install evidently pandas

COPY evidently_runner.py .

CMD ["python", "evidently_runner.py"]
```
Build and push docker image:
```
Create repository in docker.xxxx.xxxx
sudo docker build -t docker.xxx.xxx/devops/evidently:0.1 .
sudo docker push docker.xxx.xxxx/devops/evidently:0.1
```
#### evidently_runner.py for multiple datasets(models):
- This will generate reports for each dataset and save them as separate HTML files.
- path - `/data/kubernetes-nfs-storage/hf-cache/reports/`
- Docker image : `docker.merai.app/devops/evidently:0.2`

evidently_runner.py
- This is for multiple models dataset.
```
"""
Evidently Multi-Model LLM Monitor
===================================
Tailored for log format:
{
  "timestamp", "model", "route", "backend", "endpoint",
  "input": [{"role": "user", "content": "..."}],
  "output": { "choices": [...], "usage": {...} },
  "latency", "tokens": {"prompt", "completion", "total"}
}

Output:
  /data/reports/<model_name>_drift_report.html  — one per model
  /data/evidently_workspace/                    — UI workspace

View UI:
  evidently ui --workspace /data/evidently_workspace
  open http://localhost:8000
"""

import os
import json
import pandas as pd
import evidently

from evidently import Dataset, DataDefinition, Report
from evidently.presets import DataDriftPreset
from evidently.ui.workspace import Workspace

print(f"Evidently version: {evidently.__version__}")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
LOG_FILE       = "/data/llm_logs.jsonl"
WORKSPACE_PATH = "/data/evidently_workspace"
REPORTS_DIR    = "/data/reports"

# ─────────────────────────────────────────────
# 1. Load + flatten JSONL
# ─────────────────────────────────────────────
def load_logs(path: str) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping line {i}: {e}")
                continue

            # ── Flatten input ──────────────────────────────────────────────
            input_text = ""
            if isinstance(raw.get("input"), list):
                for msg in reversed(raw["input"]):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        input_text = str(msg.get("content", ""))
                        break
            else:
                input_text = str(raw.get("input", ""))

            # ── Flatten output ─────────────────────────────────────────────
            output_text   = ""
            finish_reason = ""
            out = raw.get("output", {})
            if isinstance(out, dict):
                choices = out.get("choices", [])
                if choices and isinstance(choices, list):
                    msg           = choices[0].get("message", {})
                    output_text   = str(msg.get("content", ""))
                    finish_reason = choices[0].get("finish_reason", "")
            else:
                output_text = str(out)

            # ── Flatten tokens ─────────────────────────────────────────────
            tokens            = raw.get("tokens", {})
            prompt_tokens     = tokens.get("prompt", 0)     if isinstance(tokens, dict) else 0
            completion_tokens = tokens.get("completion", 0) if isinstance(tokens, dict) else 0
            total_tokens      = tokens.get("total", 0)      if isinstance(tokens, dict) else 0

            records.append({
                "timestamp":         raw.get("timestamp"),
                "model":             raw.get("model", "unknown"),
                "route":             raw.get("route", ""),
                "latency":           float(raw.get("latency", 0)),
                "input_text":        input_text,
                "output_text":       output_text,
                "finish_reason":     finish_reason,
                "prompt_tokens":     int(prompt_tokens),
                "completion_tokens": int(completion_tokens),
                "total_tokens":      int(total_tokens),
                "prompt_len":        len(input_text),
                "response_len":      len(output_text),
            })

    df = pd.DataFrame(records)
    print(f"[INFO] Loaded {len(df)} valid rows from {path}")
    print(f"[INFO] Models found: {df['model'].unique().tolist()}")
    return df


# ─────────────────────────────────────────────
# 2. Run report for one model
# ─────────────────────────────────────────────
def run_model_report(model_name: str, df: pd.DataFrame, workspace: Workspace):
    print(f"\n── {model_name} ({len(df)} rows) ──")

    features = ["prompt_len", "response_len", "latency",
                "prompt_tokens", "completion_tokens", "total_tokens"]

    feat_df   = df[features].copy().fillna(0)
    valid_cols = [c for c in feat_df.columns if feat_df[c].nunique() > 1]

    if not valid_cols:
        print(f"   [SKIP] All columns constant — not enough variance")
        return

    feat_df = feat_df[valid_cols]
    print(f"   Features: {valid_cols}")

    split = int(len(feat_df) * 0.7)
    if split < 1 or split >= len(feat_df):
        print(f"   [SKIP] Need more rows (have {len(feat_df)}, need at least 3)")
        return

    reference_df = feat_df.iloc[:split].copy()
    current_df   = feat_df.iloc[split:].copy()
    print(f"   Reference: {len(reference_df)} | Current: {len(current_df)}")

    data_def          = DataDefinition(numerical_columns=valid_cols)
    reference_dataset = Dataset.from_pandas(reference_df, data_definition=data_def)
    current_dataset   = Dataset.from_pandas(current_df,   data_definition=data_def)

    report = Report([DataDriftPreset()])
    result = report.run(current_dataset, reference_dataset)

    # HTML named after model (slashes replaced for safe filename)
    safe_name = model_name.replace("/", "_").replace(" ", "_")
    html_path = os.path.join(REPORTS_DIR, f"{safe_name}_drift_report.html")
    result.save_html(html_path)
    print(f"   HTML saved → {html_path}")

    # One workspace project per model
    project = None
    for p in workspace.list_projects():
        if p.name == model_name:
            project = p
            break
    if project is None:
        project = workspace.create_project(model_name)
        project.description = f"Drift monitoring — {model_name}"
        project.save()

    workspace.add_run(project.id, result)
    print(f"   Workspace updated: {model_name}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(WORKSPACE_PATH, exist_ok=True)

    df        = load_logs(LOG_FILE)
    workspace = Workspace.create(WORKSPACE_PATH)
    models    = df["model"].unique().tolist()

    print(f"\n[INFO] Processing {len(models)} model(s)")

    success, failed = [], []
    for model_name in models:
        model_df = df[df["model"] == model_name].reset_index(drop=True)
        try:
            run_model_report(model_name, model_df, workspace)
            success.append(model_name)
        except Exception as e:
            print(f"   [ERROR] {model_name}: {e}")
            failed.append(model_name)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for m in success:
        safe = m.replace("/", "_").replace(" ", "_")
        print(f"  ✅ {m}")
        print(f"     → {REPORTS_DIR}/{safe}_drift_report.html")
    for m in failed:
        print(f"  ❌ {m} — failed (see errors above)")
    print(f"\n  View in UI: evidently ui --workspace {WORKSPACE_PATH}")
    print(f"  Then open:  http://localhost:8000")
    print("=" * 60)

```

### Step 3 -- Create  deployment `evidently_deployment_job.yaml`.
evidently_deployment_job.yaml
- This is for single time execution. which is not for production. so, suggesting to not use job. instead of this use cronjob.
```
apiVersion: batch/v1
kind: Job
metadata:
  name: evidently-job
  namespace: vllm
spec:
  template:
    spec:
      containers:
      - name: evidently
        image: docker.xxx.xxx/devops/evidently:0.2
        imagePullPolicy: Always
        volumeMounts:
        - name: logs
          mountPath: /data
      restartPolicy: Never
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: hf-cache-pvc
```
Apply:
```
kubectl apply -f evidently_deployment_job.yaml
```
Cronjob.yaml
```
apiVersion: batch/v1
kind: CronJob
metadata:
  name: evidently-cron
  namespace: vllm
spec:
  schedule: "*/30 * * * *"   # every 30 minutes
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: evidently
            image: docker.merai.app/devops/evidently:0.2
            imagePullPolicy: Always
            command: ["python", "/app/evidently_runner.py"]

            volumeMounts:
            - name: logs
              mountPath: /data

          restartPolicy: OnFailure

          volumes:
          - name: logs
            persistentVolumeClaim:
              claimName: hf-cache-pvc
```
Apply:
```
kubectl apply -f Cronjob.yaml
```
> log file location on host: /data/kubernetes-nfs-storage/hf-cache/llm_logs.jsonl <br>
> html file location on host : /data/kubernetes-nfs-storage/hf-cache/reports/*.html <br>
> here, hf-cache-pvc storage path is `/data/kubernetes-nfs-storage/hf-cache/`.


### Ways to View reports:
1. Download in local and open in browser.
2. View in Evidently UI.
3. View in HTTP (if you have setup nginx or any web server to serve these html files).
4. Grafana

#### 1. Download in local and open in browser:
<img width="1890" height="921" alt="image" src="https://github.com/user-attachments/assets/0b4f48d7-2819-4106-8b5e-f8cb9f92863b" />


#### 2. View in Evidently UI
Deployment for Evidently-ui : `evidently-ui-deployment.yaml`
```YAML
apiVersion: apps/v1
kind: Deployment
metadata:
  name: evidently-ui
  namespace: vllm
spec:
  replicas: 1
  selector:
    matchLabels:
      app: evidently-ui
  template:
    metadata:
      labels:
        app: evidently-ui
    spec:
      containers:
      - name: evidently-ui
        image: docker.xxxx.xxxx/devops/evidently:0.2
        command:
          - evidently
          - ui
          - --workspace
          - /data/evidently_workspace
          - --host
          - 0.0.0.0
          - --port
          - "8000"
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: logs
          mountPath: /data
      volumes:
      - name: logs
        persistentVolumeClaim:
          claimName: hf-cache-pvc
---
apiVersion: v1
kind: Service
metadata:
  name: evidently-ui-service
  namespace: vllm
spec:
  type: NodePort
  selector:
    app: evidently-ui
  ports:
    - port: 8000
      targetPort: 8000
      nodePort: 30090
```
Apply:
```
kubectl apply -f evidently-ui-deployment.yaml
## check deployment and service
kubectl get deployments -n vllm
```
#### How to check in browser:
Open - [http://10.10.110.53:30090]

```
Select model -> Reports -> View Report.
```
<img width="1917" height="1035" alt="image" src="https://github.com/user-attachments/assets/9bd96f70-1675-40d9-b283-9d4d12049b4e" />


- Note : If Values are same in all records then it will not show in graph sometimes.
- it will check all the records and if changes in every record then it will show data drift.
- 
### Open the HTML file in browser:
<img width="1900" height="949" alt="image" src="https://github.com/user-attachments/assets/7b889612-f1ff-43a3-8cdb-204e58d9e139" />
<br>

## Final Architecture:

```
                ┌──────────────────────────┐
                │   LLM Proxy (FastAPI)    │
                │ (/nucurate, /llama, etc.)│
                └───────────┬──────────────┘
                            │
                            ▼
                ┌──────────────────────────┐
                │ Structured Logs (JSONL)  │
                │ /data/llm_logs.jsonl     │
                └───────────┬──────────────┘
                            │  (PVC: hf-cache-pvc)
                            ▼
        ┌──────────────────────────────────────────┐
        │  Kubernetes CronJob (Evidently)          │
        │   runs every 15–30 min                   │
        │   - feature extraction                   │
        │   - per-model drift                      │
        │   - workspace.add_run()                  │
        └───────────┬──────────────────────────────┘
                    │
                    ▼
        ┌───────────────────────────────────────┐
        │ Evidently Workspace (PVC)             │
        │ /data/evidently_workspace             │
        │ - stores runs                         │
        │ - stores drift history                │
        └───────────┬───────────────────────────┘
                    │
                    ▼
        ┌──────────────────────────────────────────┐
        │ Evidently UI (Deployment + NodePort)     │
        │ - reads workspace                        │
        │ - shows reports + history                │
        └──────────────────────────────────────────┘
```
