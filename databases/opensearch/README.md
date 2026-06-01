# Backup Opensearch 
- OpenSearch uses snapshots — point-in-time backups of indices stored in a snapshot repository. 
- You register a repository (a location to store snapshots), then trigger snapshots manually or on a schedule.
- Both nodes must share the same volume so either can write to it.

### Step 1: Permission denied error while registering snapshot directory.

**Solution:** <br>
Provide docker volume to permissions:
```
chown -R 1000:1000 _data/
```

### Step 2 : Register Snapshot directory:
```
curl -X PUT "https://localhost:9200/_snapshot/my_backup_repo"   -u "admin:password@2026"   --insecure   -H "Content-Type: application/json"   -d '{
    "type": "fs",
    "settings": {
      "location": "/mnt/snapshots"
    }
  }'
```
Output:
```
{"acknowledged":true}
```
Verify:
```
 curl -X GET "https://localhost:9200/_snapshot/my_backup_repo" \
  -u "admin:password@2026" \
  --insecure
```

### Step 3 : Take a Snapshot ( All Indices ):
```
curl -X PUT "https://localhost:9200/_snapshot/my_backup_repo/snapshot_$(date +%Y%m%d_%H%M%S)?wait_for_completion=true" \
  -u "admin:password@2026" \
  --insecure \
  -H "Content-Type: application/json" \
  -d '{"indices": "*", "include_global_state": true}'
```

List verify Snapshots:
```
# List all snapshots
curl -X GET "https://localhost:9200/_snapshot/my_backup_repo/_all" \
  -u "admin:password@2026" \
  --insecure
```

### Optional : Restore Snapshot:
```
curl -X POST "https://localhost:9200/_snapshot/my_backup_repo/snapshot_20250601_120000/_restore" \
  -u "admin:password@2026" \
  --insecure \
  -H "Content-Type: application/json" \
  -d '{
    "indices": "*",
    "ignore_unavailable": true,
    "include_global_state": false
  }'
```



## Snapshot management (Recommended approch):
- Schedule Snapshot
- Remove older snapshots automatically.

```
curl -X POST "https://localhost:9200/_plugins/_sm/policies/daily-backup" \
  -u "admin:password@2026" \
  --insecure \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Daily snapshot at 2 AM",
    "creation": {
      "schedule": {
        "cron": {
          "expression": "0 2 * * *",
          "timezone": "Asia/Kolkata"
        }
      },
      "time_limit": "1h"
    },
    "deletion": {
      "schedule": {
        "cron": {
          "expression": "0 3 * * *",
          "timezone": "Asia/Kolkata"
        }
      },
      "condition": {
        "max_count": 30,
        "max_age": "30d"
      },
      "time_limit": "1h"
    },
    "snapshot_config": {
      "repository": "my_backup_repo",
      "indices": "*",
      "include_global_state": true
    }
  }'

```
Check scheduled policy:
```
curl -X GET "https://localhost:9200/_plugins/_sm/policies/daily-backup" \
  -u "admin:password@2026" \
  --insecure
```

DELETE Snapshot management policy {Optional}:
```
curl -X DELETE "https://localhost:9200/_plugins/_sm/policies/daily-backup" \
  -u "admin:$OPENSEARCH_INITIAL_ADMIN_PASSWORD" \
  --insecure
```
