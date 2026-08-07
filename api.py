from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from chatbot import chat as run_chat
from db.postgres import (
    fetch_person,
    fetch_store,
    fetch_stores,
    get_demographics_breakdown,
    get_footfall_count,
    list_entry_exit_logs,
    list_persons,
    upsert_store,
)

app = FastAPI(title="Store CCTV Analytics API")


class StoreIn(BaseModel):
    store_id: str
    store: str
    camera_url: str
    region: str
    metadata: Optional[dict] = None


class ChatIn(BaseModel):
    message: str
    history: Optional[list] = None


def _default_window(start: Optional[datetime], end: Optional[datetime]):
    """Default to the last 24 hours when no window is given."""
    end = end or datetime.now(timezone.utc)
    start = start or end - timedelta(hours=24)
    return start, end


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


@app.get("/stores/{store_id}/entry-exit-logs")
async def store_entry_exit_logs(store_id: str, person_id: Optional[str] = None,
                                 start: Optional[datetime] = None, end: Optional[datetime] = None,
                                 limit: int = Query(default=100, le=1000)):
    start, end = _default_window(start, end)
    return _run_query(list_entry_exit_logs, store_id, person_id, start, end, limit)


@app.post("/chat")
async def chat_endpoint(payload: ChatIn):
    try:
        answer, history = run_chat(payload.message, payload.history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"answer": answer, "history": history}
