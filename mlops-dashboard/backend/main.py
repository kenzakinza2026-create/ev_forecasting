import os
import glob
import time
from pathlib import Path

import httpx
import boto3
import mlflow
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-ui:5000")
PROMETHEUS_URL = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
MINIO_ENDPOINT = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
MINIO_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID", "minioadmin")
MINIO_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin123")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "mlflow-artifacts")
DRIFT_REPORTS_DIR = os.getenv("DRIFT_REPORTS_DIR", "/app/drift_reports")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow_client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

s3 = boto3.client(
    "s3",
    endpoint_url=MINIO_ENDPOINT,
    aws_access_key_id=MINIO_ACCESS_KEY,
    aws_secret_access_key=MINIO_SECRET_KEY,
)

app = FastAPI(title="DCForecaster MLOps Dashboard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve drift report HTML files directly so they can be iframed
os.makedirs(DRIFT_REPORTS_DIR, exist_ok=True)
app.mount("/reports", StaticFiles(directory=DRIFT_REPORTS_DIR), name="reports")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/mlflow/runs")
def mlflow_runs(max_results: int = 50):
    try:
        experiments = mlflow_client.search_experiments()
        exp_ids = [e.experiment_id for e in experiments] or ["0"]
        runs = mlflow_client.search_runs(
            experiment_ids=exp_ids,
            order_by=["attributes.start_time DESC"],
            max_results=max_results,
        )
        out = []
        for r in runs:
            out.append({
                "run_id": r.info.run_id,
                "run_name": r.info.run_name,
                "status": r.info.status,
                "start_time": r.info.start_time,
                "end_time": r.info.end_time,
                "metrics": r.data.metrics,
                "params": r.data.params,
                "tags": {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")},
            })
        return {"runs": out, "count": len(out)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MLflow error: {e}")


@app.get("/api/prometheus/query")
def prometheus_query(query: str = Query(..., description="PromQL expression")):
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": query})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prometheus error: {e}")


@app.get("/api/prometheus/targets")
def prometheus_targets():
    try:
        with httpx.Client(timeout=10) as client:
            resp = client.get(f"{PROMETHEUS_URL}/api/v1/targets")
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Prometheus error: {e}")


@app.get("/api/minio/artifacts")
def minio_artifacts(prefix: str = ""):
    try:
        resp = s3.list_objects_v2(Bucket=MINIO_BUCKET, Prefix=prefix)
        contents = resp.get("Contents", [])
        out = [{
            "key": c["Key"],
            "size_kb": round(c["Size"] / 1024, 1),
            "last_modified": c["LastModified"].isoformat(),
        } for c in contents]
        return {"bucket": MINIO_BUCKET, "objects": out, "count": len(out)}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MinIO error: {e}")


@app.get("/api/drift-reports")
def drift_reports():
    files = sorted(glob.glob(os.path.join(DRIFT_REPORTS_DIR, "*.html")), key=os.path.getmtime, reverse=True)
    out = []
    for f in files:
        name = os.path.basename(f)
        out.append({
            "name": name,
            "url": f"/reports/{name}",
            "modified": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(f))),
            "size_kb": round(os.path.getsize(f) / 1024, 1),
        })
    return {"reports": out, "count": len(out)}


# Serve the built React frontend (must be built into ./static before container start)
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
