# cctv-analytics

## Google Drive analysis API

The model API runs on the host so PyTorch can use CUDA or Apple MPS. Keep
Postgres and ChromaDB in Docker:

```bash
docker compose up -d
source .venv/bin/activate
cd analytics_service/src
export DB_HOST=localhost DB_PORT=5433
export DB_NAME=cctv_analytics DB_USER=cctv DB_PASSWORD=cctvpass
uvicorn analytics_api:app --host 0.0.0.0 --port 8090
```

Submit a publicly shared Google Drive video:

```bash
curl -X POST http://localhost:8090/analysis/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "google_drive_url": "https://drive.google.com/file/d/FILE_ID/view?usp=sharing",
    "store_id": "store2",
    "camera_id": "camera_1",
    "demographics": true
  }'
```

Poll the returned job id:

```bash
curl http://localhost:8090/analysis/jobs/JOB_ID
```

The worker processes three evenly spaced frames per ten seconds (0.3 FPS),
does not encode an output video, writes identities/events to Postgres and
short-lived OSNet embeddings to ChromaDB, and deletes the downloaded video
afterward. The Drive file must be shared as **Anyone with the link**.
