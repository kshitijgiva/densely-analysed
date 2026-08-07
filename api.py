import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from chatbot import chat as run_chat, generate_narrative
from db.postgres import (
    fetch_person,
    fetch_store,
    fetch_stores,
    get_average_dwell_seconds,
    get_demographics_breakdown,
    get_demographics_crosstab,
    get_footfall_count,
    get_footfall_time_series,
    get_latest_heatmap_url,
    list_entry_exit_logs,
    list_persons,
    list_significant_frames,
    upsert_store,
)

app = FastAPI(title="Store CCTV Analytics API")

# Comma-separated origins, or "*" to allow all (credentials must be off with "*").
# e.g. CORS_ORIGINS=http://localhost:3000,https://app.example.com
_CORS_RAW = os.environ.get("CORS_ORIGINS", "*").strip()
_CORS_ALLOW_ALL = _CORS_RAW == "*"
_CORS_ORIGINS = (
    ["*"]
    if _CORS_ALLOW_ALL
    else [origin.strip() for origin in _CORS_RAW.split(",") if origin.strip()]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_credentials=not _CORS_ALLOW_ALL,
    allow_methods=["*"],
    allow_headers=["*"],
)

_TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "1d": 24, "2d": 48}


class StoreIn(BaseModel):
    store_id: str
    store: str
    camera_url: str
    region: str
    metadata: Optional[dict] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatIn(BaseModel):
    messages: list[ChatMessage]


def _default_window(start: Optional[datetime], end: Optional[datetime]):
    """Default to the last 24 hours when no window is given."""
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(hours=24)
    return start, end


def _timeframe_window(timeframe: str):
    if timeframe not in _TIMEFRAME_HOURS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid timeframe '{timeframe}'. Use one of {sorted(_TIMEFRAME_HOURS)}.",
        )
    end = datetime.now(timezone.utc)
    start = end - timedelta(hours=_TIMEFRAME_HOURS[timeframe])
    return start, end


def _grouping_interval(start: datetime, end: datetime):
    """>24h windows group by day instead of hour, per the /reports contract."""
    if (end - start) <= timedelta(hours=24):
        return "hour", "hourly"
    return "day", "daily"


def _format_label(bucket_start: datetime, bucket: str):
    return bucket_start.strftime("%H:%M") if bucket == "hour" else bucket_start.strftime("%Y-%m-%d")


def _run_query(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except psycopg2.Error as e:
        raise HTTPException(status_code=500, detail=f"Database error: {e}")


@app.get("/")
async def root():
    return {"message": "Welcome to the CCTV Analytics API"}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/stores")
async def create_store(store: StoreIn):
    _run_query(upsert_store, store.store_id, store.store, store.camera_url, store.region, store.metadata)
    return {"store_id": store.store_id}


@app.get("/stores")
async def get_stores():
    return _run_query(fetch_stores)


@app.get("/stores/{store_id}")
async def get_store(store_id: str):
    store = _run_query(fetch_store, store_id)
    if store is None:
        raise HTTPException(status_code=404, detail=f"Store '{store_id}' not found")
    return store


@app.get("/stores/{store_id}/footfall")
async def store_footfall(store_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None):
    start, end = _default_window(start, end)
    footfall = _run_query(get_footfall_count, store_id, start, end)
    return {"store_id": store_id, "start": start, "end": end, "footfall": footfall}


@app.get("/stores/{store_id}/demographics")
async def store_demographics(store_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None):
    start, end = _default_window(start, end)
    breakdown = _run_query(get_demographics_breakdown, store_id, start, end)
    return {"store_id": store_id, "start": start, "end": end, **breakdown}


@app.get("/stores/{store_id}/persons")
async def store_persons(store_id: str, start: Optional[datetime] = None, end: Optional[datetime] = None,
                         limit: int = Query(default=100, le=1000)):
    start, end = _default_window(start, end)
    return _run_query(list_persons, store_id, start, end, limit)


@app.get("/persons/{person_id}")
async def get_person(person_id: str):
    person = _run_query(fetch_person, person_id)
    if person is None:
        raise HTTPException(status_code=404, detail=f"Person '{person_id}' not found")
    return person


@app.get("/stores/{store_id}/significant-frames")
async def store_significant_frames(store_id: str, start: Optional[datetime] = None,
                                    end: Optional[datetime] = None,
                                    limit: int = Query(default=10, le=100)):
    """The dashboard's per-store highlights: highest-importance flagged frames
    in the window, not every frame significant_frames.py saved locally."""
    start, end = _default_window(start, end)
    return _run_query(list_significant_frames, store_id, start, end, limit)


@app.get("/stores/{store_id}/entry-exit-logs")
async def store_entry_exit_logs(store_id: str, person_id: Optional[str] = None,
                                 start: Optional[datetime] = None, end: Optional[datetime] = None,
                                 limit: int = Query(default=100, le=1000)):
    start, end = _default_window(start, end)
    return _run_query(list_entry_exit_logs, store_id, person_id, start, end, limit)


@app.get("/overview")
async def overview(store_id: Optional[str] = None, timeframe: str = "1d"):
    start, end = _timeframe_window(timeframe)
    calculated_at = datetime.now(timezone.utc)

    footfall = _run_query(get_footfall_count, store_id, start, end)
    dwell = _run_query(get_average_dwell_seconds, store_id, start, end)
    series = _run_query(get_footfall_time_series, store_id, start, end, "hour")
    demographics = _run_query(get_demographics_crosstab, store_id, start, end)
    # Aggregate (no store_id) view has no single camera to show a heatmap for.
    heatmap_url = _run_query(get_latest_heatmap_url, store_id) if store_id else None

    kpis = {
        "total_footfall": footfall,
        # Not tracked yet - the pipeline never assigns detections to a zone
        # (see persist.py), so there's no checkout/conversion signal to compute this from.
        "conversion_rate": None,
        "average_dwell_time_seconds": dwell,
        # Not tracked yet - the pipeline is a batch job that persists a person's
        # entry+exit together only once the whole job finishes, so there's no
        # live "currently in store" state anywhere to count from.
        "active_visitors": 0,
    }

    return {
        "meta": {"store_id": store_id, "timeframe": timeframe, "calculated_at": calculated_at},
        "kpis": kpis,
        "footfall_time_series": [
            {"timestamp": row["bucket_start"], "count": row["count"]} for row in series
        ],
        "demographics": demographics,
        # None until a pipeline run with --persist has uploaded one for this store.
        "heatmap_image_url": heatmap_url,
        "narrative_summary": generate_narrative(kpis, store_id),
    }


@app.get("/reports")
async def reports(store_id: str, start_date: datetime, end_date: datetime):
    bucket, grouping_interval = _grouping_interval(start_date, end_date)

    footfall = _run_query(get_footfall_count, store_id, start_date, end_date)
    dwell = _run_query(get_average_dwell_seconds, store_id, start_date, end_date)
    series = _run_query(get_footfall_time_series, store_id, start_date, end_date, bucket)
    demographics = _run_query(get_demographics_crosstab, store_id, start_date, end_date)

    return {
        "meta": {
            "store_id": store_id,
            "start_date": start_date,
            "end_date": end_date,
            "grouping_interval": grouping_interval,
        },
        "kpis": {
            "total_footfall": footfall,
            "average_dwell_time_seconds": dwell,
            "conversion_rate": None,  # see /overview - no zone/checkout data in the pipeline yet
        },
        "time_series_trends": [
            {
                "label": _format_label(row["bucket_start"], bucket),
                "footfall": row["count"],
                "conversion_rate": None,
            }
            for row in series
        ],
        "zone_analytics": [],  # zones aren't tracked by the pipeline yet (see persist.py)
        "demographics": demographics,
    }


@app.post("/chat")
async def chat_endpoint(payload: ChatIn):
    if not payload.messages:
        raise HTTPException(status_code=400, detail="messages must not be empty")

    *history, last = payload.messages
    try:
        answer, _ = run_chat(last.content, [m.model_dump() for m in history])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    def stream():
        chunk_size = 40
        for i in range(0, len(answer), chunk_size):
            yield answer[i : i + chunk_size]

    return StreamingResponse(stream(), media_type="text/plain")
