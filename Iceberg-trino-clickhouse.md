# Trino

- `Trino` is distributed SQL Query engine.
- Allows you to run SQL on data stored in : Minio/s3, Iceberg, Mysql, PostgresQL, Kafka, Hive. without moving data.

# Iceberg

- `iceberg` stores metadata of table.
- If we insert some record in Mysql then iceberg stores actual data and its metadata into Minio.
- 1000 Mysql inserts -> Iceberg -> 1 Parquet file.

## Iceberg and Trino setup with docker-compose:

docker-compose.yaml
```
services:

  nessie:
    image: ghcr.io/projectnessie/nessie:latest
    container_name: nessie
    ports:
      - "19120:19120"

  trino:
    image: trinodb/trino:477
    container_name: trino
    ports:
      - "8080:8080"
    depends_on:
      - nessie
    volumes:
      - ./catalog:/etc/trino/catalog

```
Start the services:
```
docker-compose up -d
```
You can access Iceberg web UI:
```
http://localhost:19120/ui
```
You can access Trino web ui:
```
http://localhost:8080/ui
```

### Create Iceberg Catalog configuration in Trino:
nano catalog/liceberg.properties
```
connector.name=iceberg
iceberg.catalog.type=nessie
iceberg.nessie-catalog.uri=http://nessie:19120/api/v2
iceberg.nessie-catalog.ref=main
iceberg.nessie-catalog.default-warehouse-dir=s3://xxx/
fs.native-s3.enabled=true
s3.endpoint=https://mns3006.xx.xx
s3.region=us-east-1
s3.path-style-access=true
s3.aws-access-key=xxxxxxxxxxxxxxx
s3.aws-secret-key=_xxxxxxxxxxxxxxxxxxxxxxxx
```

### Connect Trino CLI:
```
docker exec -it trino trino
```
#### create schema:
```
CREATE SCHEMA iceberg.demo;
```
List Created Schema:
```
SHOW SCHEMAS FROM iceberg;
```
#### Create table
```
CREATE TABLE iceberg.demo.employees (
    id BIGINT,
    name VARCHAR,
    country VARCHAR,
    salary DOUBLE
);

```

#### Insert records:
```
INSERT INTO iceberg.demo.employees
VALUES
(1,'John','India',1000),
(2,'Mike','USA',2000),
(3,'Alice','UK',3000);
```

Iceberg will automatically create in s3:
```
xxxxx/
 └── demo
      └── employees
           ├── metadata/
           └── data/
                *.parquet
```

Parquet file is created on S3 storage:
```LOG
s3cmd ls s3://xx/demo/employees-ece672c1c8a6d48c6b8b658342d1ccb0cf/data/
2026-06-05 11:43          654  s3://xx/demo/employees-ece672c1c8ca648c6b8b6583421ccb0cf/data/20260605_114302_00003_6fwyt-e91a45d6-24e0-4366-8738-2cfce720c614.parquet
```

Then,
```
SELECT * FROM iceberg.demo.employees;
```
> Parquet files -> Actual data <br>
> Metadata -> Table information.

Flow:
```
 SQL Query
  |
 Trino
  |
 Iceberg 
  |
 Metadata data layer
  |
 S3/Minio/NetApp
  |
 Parquet Files
```

# ClickHouse
ClickHouse is a database.

It stores the data itself.
```
ClickHouse
    |
    +--> Local storage
    +--> Object storage (optional)
```
Its job is to ingest, store, index, compress, and query data efficiently. <br>

Primarily queries data stored in ClickHouse.

It has integrations, but federated querying is not its main purpose.

Trino + Iceberg
- Cheap storage
- S3 + Parquet

Store hundreds of terabytes economically.
- ClickHouse
- Storage + Compute

You are maintaining a database cluster.

More expensive at very large scales.

Suppose You have 10TB of data of orders.
```
minio
  |
Iceberg
```
Now, CEO wants to open dashboard and ask to get records. running all those queries will work but slower. <br>
So Companies load data into clickhouse.

ClickHouse build specifically for:
```
count()
sum()
avg()
GroupBy()
Analytics
Dashboard
```

### Final stack:
- Mysql : Applications
- Iceberg : Long-term storage
- Clickhouse : Dashboards, Reports, Routine analytics, realtime metrics

### Architecture:
```
Mysql
  |
Debezium
  |
Kafka
  |
Iceberg + Minio
  |
ClickHouse
  |
Grafana
```


