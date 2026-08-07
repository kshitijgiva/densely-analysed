"""Local job API for downloading and sparsely analyzing Google Drive videos.

Run this service on the host (not in Docker) so PyTorch can use CUDA/MPS:
    uvicorn analytics_api:app --host 0.0.0.0 --port 8090
"""
import os
import shutil
import sys
import tempfile
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

import gdown
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR in sys.path:
    sys.path.remove(SRC_DIR)
sys.path.insert(0, SRC_DIR)

SAMPLE_FRAMES = 3
SAMPLE_WINDOW_SECONDS = 10
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "jobs"
MAX_WORKERS = int(os.environ.get("ANALYTICS_MAX_WORKERS", "1"))

app = FastAPI(title="CCTV Video Analysis Jobs API")
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs = {}
jobs_lock = threading.Lock()


class AnalysisRequest(BaseModel):
    google_drive_url: str
    store_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    camera_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    demographics: bool = True
    max_sampled_frames: int | None = Field(default=None, gt=0)
    reid_threshold: float | None = Field(
        default=None,
        description="Cosine similarity threshold for re-identification. Defaults to "
        "config.REID_THRESHOLD, which is tuned for dense, frame-by-frame tracking, not "
        "this endpoint's sparse sampling (SAMPLE_FRAMES per SAMPLE_WINDOW_SECONDS) - re-validate "
        "with validate_pipeline.py using matching --sample-frames/--sample-window-seconds "
        "and pass the suggested value here if footfall looks inflated.",
    )


def _validate_google_drive_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {
        "drive.google.com",
        "docs.google.com",
    }:
        raise ValueError("Only HTTPS Google Drive links are accepted")


def _update_job(job_id: str, **values) -> None:
    with jobs_lock:
        jobs[job_id].update(values)


def _process_job(job_id: str, request: AnalysisRequest) -> None:
    # Keep API startup light; load PyTorch/YOLO only when a worker starts a job.
    from render_tracked_video import run
    from persist import persist_identities

    temp_dir = Path(tempfile.mkdtemp(prefix=f"cctv-{job_id}-"))
    input_path = temp_dir / "input_video"
    metrics_path = RESULTS_DIR / f"{job_id}_metrics.json"

    try:
        _update_job(job_id, status="downloading", started_at=datetime.now(timezone.utc))
        downloaded = gdown.download(
            url=request.google_drive_url,
            output=str(input_path),
            quiet=False,
            fuzzy=True,
        )
        if not downloaded or not input_path.exists() or input_path.stat().st_size == 0:
            raise RuntimeError(
                "Google Drive download failed. Ensure the file is shared as "
                "'Anyone with the link'."
            )

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        _update_job(job_id, status="analyzing")
        run_kwargs = dict(
            video_source=str(input_path),
            output_path=str(temp_dir / "unused.mp4"),
            max_frames=request.max_sampled_frames,
            metrics_out_path=str(metrics_path),
            run_demographics=request.demographics,
            use_chroma=True,
            store_id=request.store_id,
            camera_id=request.camera_id,
            sample_frames=SAMPLE_FRAMES,
            sample_window_seconds=SAMPLE_WINDOW_SECONDS,
            write_video=False,
        )
        if request.reid_threshold is not None:
            run_kwargs["reid_threshold"] = request.reid_threshold
        result = run(**run_kwargs)

        _update_job(job_id, status="persisting")
        run_start = datetime.now(timezone.utc) - timedelta(
            seconds=result["source_duration_seconds"]
        )
        persisted = persist_identities(
            result["identities"],
            request.store_id,
            request.camera_id,
            result["fps"],
            run_start=run_start,
            camera_url=request.google_drive_url,
        )

        _update_job(
            job_id,
            status="completed",
            completed_at=datetime.now(timezone.utc),
            persisted_identities=persisted,
            metrics=result["report"],
            metrics_path=str(metrics_path),
        )
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            completed_at=datetime.now(timezone.utc),
            error=str(exc),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "sampling": {
            "frames": SAMPLE_FRAMES,
            "window_seconds": SAMPLE_WINDOW_SECONDS,
            "effective_fps": SAMPLE_FRAMES / SAMPLE_WINDOW_SECONDS,
        },
        "max_concurrent_jobs": MAX_WORKERS,
    }


@app.post("/analysis/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_analysis_job(request: AnalysisRequest):
    try:
        _validate_google_drive_url(request.google_drive_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc),
            "store_id": request.store_id,
            "camera_id": request.camera_id,
            "sampling": "3 frames per 10 seconds",
        }
    executor.submit(_process_job, job_id, request)
    return jobs[job_id]


@app.get("/analysis/jobs/{job_id}")
def get_analysis_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Analysis job not found")
        return dict(job)


@app.get("/analysis/jobs")
def list_analysis_jobs():
    with jobs_lock:
        return list(jobs.values())
