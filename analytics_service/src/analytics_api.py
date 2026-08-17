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
from zoneinfo import ZoneInfo

import gdown
from fastapi import FastAPI, HTTPException, status
from fastapi.encoders import ENCODERS_BY_TYPE
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

SRC_DIR = str(Path(__file__).resolve().parent)
if SRC_DIR in sys.path:
    sys.path.remove(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from config import LOCAL_VIDEO_DIR, REID_THRESHOLD_SPARSE  # noqa: E402

SAMPLE_FRAMES = 3
SAMPLE_WINDOW_SECONDS = 10
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "jobs"
LOCAL_VIDEO_ROOT = Path(LOCAL_VIDEO_DIR).resolve()
MAX_WORKERS = int(os.environ.get("ANALYTICS_MAX_WORKERS", "1"))

_IST = ZoneInfo("Asia/Kolkata")


def _format_datetime_ist(dt: datetime) -> str:
    """Render every datetime in API responses as IST, dropping sub-second noise."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_IST).strftime("%Y-%m-%d %H:%M:%S IST")


# fastapi's jsonable_encoder consults this shared dict for every response in the
# app (dict/list returns included, not just Pydantic models) - patching it here
# is the single choke point that affects every endpoint's datetime fields.
ENCODERS_BY_TYPE[datetime] = _format_datetime_ist

app = FastAPI(title="CCTV Video Analysis Jobs API")

_CORS_RAW = os.environ.get("CORS_ORIGINS", "*").strip()
_CORS_ALLOW_ALL = _CORS_RAW == "*"
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _CORS_ALLOW_ALL else [
        origin.strip() for origin in _CORS_RAW.split(",") if origin.strip()
    ],
    allow_credentials=not _CORS_ALLOW_ALL,
    allow_methods=["*"],
    allow_headers=["*"],
)

executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
jobs = {}
jobs_lock = threading.Lock()


class AnalysisRequest(BaseModel):
    # Exactly one source. google_drive_url is the normal path; local_video skips
    # the download and reads footage already sitting in data/raw (see
    # config.LOCAL_VIDEO_DIR) - the same analysis then runs unchanged.
    google_drive_url: str | None = None
    local_video: str | None = Field(
        default=None,
        description="Filename (or relative path) of a video under data/raw, e.g. "
        "'samplevideo.mp4'. Mutually exclusive with google_drive_url.",
    )
    store_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    camera_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_-]+$")
    demographics: bool = True
    # Wall-clock time when the video recording starts. Entry/exit timestamps are
    # then run_start + frame_idx/fps across the video length. ISO-8601, e.g.
    # "2026-08-07T10:00:00+05:30". If omitted, falls back to (now - duration).
    start_time: datetime | None = None
    sample_frames: int = Field(
        default=SAMPLE_FRAMES,
        gt=0,
        description="Evenly spaced frames to process per sample_window_seconds.",
    )
    sample_window_seconds: float = Field(
        default=SAMPLE_WINDOW_SECONDS,
        gt=0,
        description="Sampling window size in seconds.",
    )
    max_sampled_frames: int | None = Field(default=None, gt=0)
    reid_threshold: float | None = Field(
        default=None,
        description="Cosine similarity threshold for re-identification. Defaults to "
        "config.REID_THRESHOLD_SPARSE, tuned for this endpoint's sparse sampling "
        "(SAMPLE_FRAMES per SAMPLE_WINDOW_SECONDS) - re-validate with validate_pipeline.py "
        "using matching --sample-frames/--sample-window-seconds if footfall still looks "
        "inflated (or reduced) on your footage, and pass the suggested value here.",
    )

    @model_validator(mode="after")
    def _exactly_one_source(self):
        if bool(self.google_drive_url) == bool(self.local_video):
            raise ValueError(
                "Provide exactly one of 'google_drive_url' or 'local_video'."
            )
        return self


def _resolve_local_video(name: str) -> Path:
    """Map a caller-supplied name onto a real file inside data/raw.

    Resolved and re-checked against LOCAL_VIDEO_ROOT so '../../.env'-style
    input can't turn this into an arbitrary-file-read endpoint.
    """
    candidate = (LOCAL_VIDEO_ROOT / name).resolve()
    if not candidate.is_relative_to(LOCAL_VIDEO_ROOT):
        raise ValueError(f"local_video must be inside {LOCAL_VIDEO_ROOT}")
    if not candidate.is_file():
        raise ValueError(f"No such video: {candidate}")
    return candidate


def _normalize_start_time(start_time: datetime | None) -> datetime | None:
    if start_time is None:
        return None
    if start_time.tzinfo is None:
        return start_time.replace(tzinfo=timezone.utc)
    return start_time.astimezone(timezone.utc)


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
    from persist import persist_identities, persist_heatmap
    from validate_pipeline import estimate_reid_threshold

    temp_dir = Path(tempfile.mkdtemp(prefix=f"cctv-{job_id}-"))
    metrics_path = RESULTS_DIR / f"{job_id}_metrics.json"

    try:
        _update_job(job_id, started_at=datetime.now(timezone.utc))

        if request.local_video is not None:
            # Read straight from data/raw. input_path deliberately stays outside
            # temp_dir so the `finally` cleanup below can't delete the user's
            # source footage.
            _update_job(job_id, status="loading_local_video")
            input_path = _resolve_local_video(request.local_video)
            source_ref = f"local:{request.local_video}"
        else:
            _update_job(job_id, status="downloading")
            input_path = temp_dir / "input_video"
            # fuzzy=True is required for the installed gdown (5.x) to extract the file ID
            # from a /file/d/<id>/view?usp=sharing share link - without it, gdown silently
            # downloads Google's HTML viewer page instead of the video (still a nonzero-size
            # "success" by the check below, only failing later at cv2.VideoCapture).
            try:
                downloaded = gdown.download(
                    url=request.google_drive_url,
                    output=str(input_path),
                    quiet=False,
                    fuzzy=True,
                )
            except gdown.DownloadError as exc:
                raise RuntimeError(
                    "Google Drive download failed. Ensure the file is shared as "
                    "'Anyone with the link'."
                ) from exc
            if not downloaded or not input_path.exists() or input_path.stat().st_size == 0:
                raise RuntimeError(
                    "Google Drive download failed. Ensure the file is shared as "
                    "'Anyone with the link'."
                )
            source_ref = request.google_drive_url

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        reid_threshold = request.reid_threshold
        reid_threshold_source = "explicit"
        if reid_threshold is None:
            _update_job(job_id, status="calibrating")
            # This footage's own same-person/different-person similarity spread
            # varies by lighting/crowd/camera - a fixed global threshold
            # (REID_THRESHOLD_SPARSE) fragments real visitors into extra
            # identities on footage it wasn't tuned for. Estimate one from this
            # video itself (same detect+track+reid pass validate_pipeline.py
            # uses, just without its CLI/CSV side effects) before falling back.
            estimated = estimate_reid_threshold(
                str(input_path),
                sample_frames=request.sample_frames,
                sample_window_seconds=request.sample_window_seconds,
            )
            reid_threshold = estimated if estimated is not None else REID_THRESHOLD_SPARSE
            reid_threshold_source = "auto_calibrated" if estimated is not None else "default_fallback"

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
            sample_frames=request.sample_frames,
            sample_window_seconds=request.sample_window_seconds,
            write_video=False,
            reid_threshold=reid_threshold,
        )
        result = run(**run_kwargs)

        _update_job(job_id, status="persisting")
        duration_s = result["source_duration_seconds"]
        run_start = _normalize_start_time(request.start_time) or (
            datetime.now(timezone.utc) - timedelta(seconds=duration_s)
        )
        run_end = run_start + timedelta(seconds=duration_s)
        persisted = persist_identities(
            result["identities"],
            request.store_id,
            request.camera_id,
            result["fps"],
            run_start=run_start,
            camera_url=source_ref,
            require_demographics=request.demographics,
        )

        # Uploads/persists before temp_dir (which holds the local heatmap PNG)
        # is removed in `finally` below. A failed upload shouldn't fail an
        # otherwise-successful job - identities are already persisted above.
        heatmap_url = None
        heatmap_image_path = result["report"]["heatmap"].get("image_path")
        if heatmap_image_path:
            from cloudinary_upload import upload_heatmap

            try:
                heatmap_url = upload_heatmap(heatmap_image_path, request.store_id, request.camera_id)
                persist_heatmap(request.store_id, request.camera_id, heatmap_url)
            except Exception as exc:
                print(f"Warning: heatmap upload/persist failed for job {job_id}: {exc}")

        _update_job(
            job_id,
            status="completed",
            completed_at=datetime.now(timezone.utc),
            persisted_identities=persisted,
            metrics=result["report"],
            metrics_path=str(metrics_path),
            heatmap_url=heatmap_url,
            video_start_time=run_start,
            video_end_time=run_end,
            video_duration_seconds=duration_s,
            reid_threshold=reid_threshold,
            reid_threshold_source=reid_threshold_source,
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
    # Reject a bad source now, synchronously, rather than letting the worker
    # mark the job "failed" a few seconds later.
    try:
        if request.local_video is not None:
            source = f"local:{_resolve_local_video(request.local_video).name}"
        else:
            _validate_google_drive_url(request.google_drive_url)
            source = request.google_drive_url
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "created_at": datetime.now(timezone.utc),
            "source": source,
            "store_id": request.store_id,
            "camera_id": request.camera_id,
            "sampling": (
                f"{request.sample_frames} frames per "
                f"{request.sample_window_seconds:g} seconds"
            ),
            "video_start_time": _normalize_start_time(request.start_time),
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
