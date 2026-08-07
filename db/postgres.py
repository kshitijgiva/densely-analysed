import os

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def get_connection():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "cctv_analytics"),
        user=os.environ.get("DB_USER", "cctv"),
        password=os.environ.get("DB_PASSWORD", "cctvpass"),
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
    )


def _execute(query, params=None, fetch=None):
    """fetch: None (no return), 'one', or 'all'."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(query, params)
            result = None
            if fetch == "one":
                result = cur.fetchone()
            elif fetch == "all":
                result = cur.fetchall()
        conn.commit()
        return result
    finally:
        conn.close()


# --- stores -----------------------------------------------------------------

def upsert_store(store_id, store, camera_url, region, metadata=None):
    _execute(
        """
        INSERT INTO stores (store_id, store, camera_url, region, metadata)
        VALUES (%s, %s, %s, %s, COALESCE(%s, '{}'::jsonb))
        ON CONFLICT (store_id) DO UPDATE SET
            store = EXCLUDED.store,
            camera_url = EXCLUDED.camera_url,
            region = EXCLUDED.region,
            metadata = EXCLUDED.metadata,
            updated_at = now();
        """,
        (store_id, store, camera_url, region, Json(metadata) if metadata else None),
    )


def fetch_stores():
    return _execute("SELECT * FROM stores ORDER BY store_id;", fetch="all")


def fetch_store(store_id):
    return _execute("SELECT * FROM stores WHERE store_id = %s;", (store_id,), fetch="one")


# --- persons ------------------------------------------------------------------

def upsert_person(person_id, store_id, camera_id, track_id, gender, gender_confidence,
                   age_group, age_confidence, needs_demographic_retry, first_seen, last_seen,
                   reid_embedding_ref=None, metadata=None):
    """person_id must be a UUID (str) - callers mint one with uuid.uuid4() for new persons."""
    _execute(
        """
        INSERT INTO persons (
            id, store_id, camera_id, track_id, reid_embedding_ref,
            gender, gender_confidence, age_group, age_confidence,
            needs_demographic_retry, first_seen, last_seen, metadata
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s, '{}'::jsonb))
        ON CONFLICT (id) DO UPDATE SET
            reid_embedding_ref = EXCLUDED.reid_embedding_ref,
            gender = EXCLUDED.gender,
            gender_confidence = EXCLUDED.gender_confidence,
            age_group = EXCLUDED.age_group,
            age_confidence = EXCLUDED.age_confidence,
            needs_demographic_retry = EXCLUDED.needs_demographic_retry,
            last_seen = EXCLUDED.last_seen,
            metadata = EXCLUDED.metadata;
        """,
        (person_id, store_id, camera_id, track_id, reid_embedding_ref,
         gender, gender_confidence, age_group, age_confidence,
         needs_demographic_retry, first_seen, last_seen,
         Json(metadata) if metadata else None),
    )


def fetch_person(person_id):
    return _execute("SELECT * FROM persons WHERE id = %s;", (person_id,), fetch="one")


def list_persons(store_id=None, start_time=None, end_time=None, limit=100):
    conditions, params = [], []
    if store_id:
        conditions.append("store_id = %s")
        params.append(store_id)
    if start_time:
        conditions.append("first_seen >= %s")
        params.append(start_time)
    if end_time:
        conditions.append("first_seen <= %s")
        params.append(end_time)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    return _execute(
        f"SELECT * FROM persons {where} ORDER BY first_seen DESC LIMIT %s;",
        tuple(params), fetch="all",
    )


# --- entry/exit + zone events -------------------------------------------------

def insert_entry_exit_event(person_id, store_id, camera_id, event_type, event_time, zone_id=None):
    _execute(
        """
        INSERT INTO entry_exit_logs (person_id, store_id, camera_id, zone_id, event_type, event_time)
        VALUES (%s, %s, %s, %s, %s, %s);
        """,
        (person_id, store_id, camera_id, zone_id, event_type, event_time),
    )


def list_entry_exit_logs(store_id=None, person_id=None, start_time=None, end_time=None, limit=100):
    conditions, params = [], []
    if store_id:
        conditions.append("store_id = %s")
        params.append(store_id)
    if person_id:
        conditions.append("person_id = %s")
        params.append(person_id)
    if start_time:
        conditions.append("event_time >= %s")
        params.append(start_time)
    if end_time:
        conditions.append("event_time <= %s")
        params.append(end_time)
    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)
    return _execute(
        f"SELECT * FROM entry_exit_logs {where} ORDER BY event_time DESC LIMIT %s;",
        tuple(params), fetch="all",
    )


# --- aggregate queries (S3.7: "REST API for querying stats") -----------------

def get_footfall_count(store_id, start_time, end_time):
    """Unique footfall = distinct persons with an 'entry' event in the window."""
    row = _execute(
        """
        SELECT COUNT(DISTINCT person_id) AS footfall
        FROM entry_exit_logs
        WHERE store_id = %s AND event_type = 'entry'
          AND event_time >= %s AND event_time <= %s;
        """,
        (store_id, start_time, end_time), fetch="one",
    )
    return row["footfall"] if row else 0


def get_demographics_breakdown(store_id, start_time, end_time):
    """Gender/age-group counts for persons first seen in the window."""
    gender_rows = _execute(
        """
        SELECT COALESCE(gender, 'unknown') AS gender, COUNT(*) AS count
        FROM persons
        WHERE store_id = %s AND first_seen >= %s AND first_seen <= %s
        GROUP BY gender;
        """,
        (store_id, start_time, end_time), fetch="all",
    )
    age_rows = _execute(
        """
        SELECT COALESCE(age_group, 'unknown') AS age_group, COUNT(*) AS count
        FROM persons
        WHERE store_id = %s AND first_seen >= %s AND first_seen <= %s
        GROUP BY age_group;
        """,
        (store_id, start_time, end_time), fetch="all",
    )
    return {
        "gender_breakdown": {r["gender"]: r["count"] for r in gender_rows},
        "age_group_breakdown": {r["age_group"]: r["count"] for r in age_rows},
    }
