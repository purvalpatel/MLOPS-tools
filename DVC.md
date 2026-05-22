1. Install DVC

Install with S3 support:
```
pip install "dvc[s3]"
```
Verify:
```
dvc version
```

2. Create Project
```
mkdir ai-project
cd ai-project

git init
dvc init
```
Commit setup:
```
git add .
git commit -m "Initialize DVC"
```
3. Add Dataset

Example:
```
mkdir data
cp /datasets/train.parquet data/
```
Track dataset:
```
dvc add data/
```

This creates:
```
data.dvc
```
Git now tracks only metadata.

4. Configure NetApp S3 Remote

Example:
```
dvc remote add -d netapp s3://ml-datasets/dvc
```
Configure NetApp S3 endpoint:
```
dvc remote modify netapp endpointurl https://s3.company.local
```
Add credentials:
```
dvc remote modify netapp access_key_id YOUR_ACCESS_KEY

dvc remote modify netapp secret_access_key YOUR_SECRET_KEY
```
If using self-signed certs:
```
dvc remote modify netapp ssl_verify false
```
5. Push Dataset to NetApp
```
dvc push
```
Now actual files upload to:
```
s3://ml-datasets/dvc
```
inside your NetApp object storage.

6. Commit Metadata to Git
```
git add .
git commit -m "Add training dataset"
```

8. Clone on Another Machine

Another GPU node:
```
git clone <repo>
cd ai-project

pip install "dvc[s3]"

dvc pull
```
DVC downloads actual data from NetApp S3 automatically.
