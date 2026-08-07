import os

import psycopg2
from psycopg2.extras import Json, RealDictCursor


def get_connection():
    # Prefer a full URL (Railway / Prisma) when set; else discrete DB_* vars.
    database_url = os.environ.get("DATABASE_URL")
    if database_url:
        return psycopg2.connect(database_url)

    kwargs = dict(
        dbname=os.environ.get("DB_NAME", "cctv_analytics"),
        user=os.environ.get("DB_USER", "cctv"),
        password=os.environ.get("DB_PASSWORD", "cctvpass"),
        host=os.environ.get("DB_HOST", "localhost"),
        port=os.environ.get("DB_PORT", "5432"),
    )
    sslmode = os.environ.get("DB_SSLMODE")
    if sslmode:
        kwargs["sslmode"] = sslmode
    return psycopg2.connect(**kwargs)


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
    """Unique footfall = distinct persons with an 'entry' event in the window.
    store_id=None aggregates across all stores (Giva Owner view)."""
    conditions = ["event_type = 'entry'", "event_time >= %s", "event_time <= %s"]
    params = [start_time, end_time]
    if store_id:
        conditions.append("store_id = %s")
        params.append(store_id)
    row = _execute(
        f"""
        SELECT COUNT(DISTINCT person_id) AS footfall
        FROM entry_exit_logs
        WHERE {' AND '.join(conditions)};
        """,
        tuple(params), fetch="one",
    )
    return row["footfall"] if row else 0


def insert_significant_frame(store_id, camera_id, event_time, person_count,
                              motion_ratio, reasons, importance_score, image_url):
    _execute(
        """
        INSERT INTO significant_frames (
            store_id, camera_id, event_time, person_count,
            motion_ratio, reasons, importance_score, image_url
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s);
        """,
        (store_id, camera_id, event_time, person_count,
         motion_ratio, reasons, importance_score, image_url),
    )


def list_significant_frames(store_id, start_time=None, end_time=None, limit=10):
    """Highest-importance frames first within the window - the small set of
    highlights a store dashboard shows, not every frame that was saved."""
    conditions, params = ["store_id = %s"], [store_id]
    if start_time:
        conditions.append("event_time >= %s")
        params.append(start_time)
    if end_time:
        conditions.append("event_time <= %s")
        params.append(end_time)
    where = f"WHERE {' AND '.join(conditions)}"
    params.append(limit)
    return _execute(
        f"""SELECT * FROM significant_frames {where}
            ORDER BY importance_score DESC, event_time DESC LIMIT %s;""",
        tuple(params), fetch="all",
    )


def upsert_camera_heatmap(store_id, camera_id, image_url):
    """One row per (store_id, camera_id) - each call overwrites the previous
    URL/timestamp rather than accumulating history."""
    _execute(
        """
        INSERT INTO camera_heatmaps (store_id, camera_id, image_url, updated_at)
        VALUES (%s, %s, %s, now())
        ON CONFLICT (store_id, camera_id) DO UPDATE SET
            image_url = EXCLUDED.image_url,
            updated_at = now();
        """,
        (store_id, camera_id, image_url),
    )


def get_latest_heatmap_url(store_id):
    """Most recently updated camera's heatmap for this store - good enough for
    a single-image /overview card even if a store has more than one camera."""
    row = _execute(
        """
        SELECT image_url FROM camera_heatmaps
        WHERE store_id = %s
        ORDER BY updated_at DESC
        LIMIT 1;
        """,
        (store_id,), fetch="one",
    )
    return row["image_url"] if row else None


def get_average_dwell_seconds(store_id, start_time, end_time):
    """Average visit duration (last_seen - first_seen) for persons first seen in
    the window. store_id=None aggregates across all stores."""
    conditions = ["first_seen >= %s", "first_seen <= %s"]
    params = [start_time, end_time]
    if store_id:
        conditions.append("store_id = %s")
        params.append(store_id)
    row = _execute(
        f"""
        SELECT AVG(EXTRACT(EPOCH FROM (last_seen - first_seen))) AS avg_dwell
        FROM persons
        WHERE {' AND '.join(conditions)};
        """,
        tuple(params), fetch="one",
    )
    avg_dwell = row["avg_dwell"] if row else None
    return round(float(avg_dwell), 1) if avg_dwell is not None else 0.0


def get_footfall_time_series(store_id, start_time, end_time, bucket="hour"):
    """Distinct-person footfall counts bucketed by hour or day, for the
    /overview and /reports time-series charts. store_id=None aggregates
    across all stores."""
    if bucket not in ("hour", "day"):
        raise ValueError("bucket must be 'hour' or 'day'")
    conditions = ["event_type = 'entry'", "event_time >= %s", "event_time <= %s"]
    params = [start_time, end_time]
    if store_id:
        conditions.append("store_id = %s")
        params.append(store_id)
    return _execute(
        f"""
        SELECT date_trunc(%s, event_time) AS bucket_start, COUNT(DISTINCT person_id) AS count
        FROM entry_exit_logs
        WHERE {' AND '.join(conditions)}
        GROUP BY bucket_start
        ORDER BY bucket_start;
        """,
        (bucket, *params), fetch="all",
    )


def get_demographics_crosstab(store_id, start_time, end_time):
    """Gender totals plus an age-group x gender breakdown, shaped for the
    /overview and /reports API contract (gender as a list of {name, value},
    age_groups cross-tabbed by gender). store_id=None aggregates across all
    stores."""
    conditions = ["first_seen >= %s", "first_seen <= %s"]
    params = [start_time, end_time]
    if store_id:
        conditions.append("store_id = %s")
        params.append(store_id)
    where = " AND ".join(conditions)

    gender_rows = _execute(
        f"""
        SELECT COALESCE(gender, 'unknown') AS gender, COUNT(*) AS count
        FROM persons
        WHERE {where}
        GROUP BY gender;
        """,
        tuple(params), fetch="all",
    )
    crosstab_rows = _execute(
        f"""
        SELECT COALESCE(age_group, 'unknown') AS age_group,
               COALESCE(gender, 'unknown') AS gender,
               COUNT(*) AS count
        FROM persons
        WHERE {where}
        GROUP BY age_group, gender;
        """,
        tuple(params), fetch="all",
    )

    gender = [{"name": r["gender"].capitalize(), "value": r["count"]} for r in gender_rows]

    age_groups = {}
    for row in crosstab_rows:
        entry = age_groups.setdefault(row["age_group"], {"group": row["age_group"]})
        entry[row["gender"]] = row["count"]

    return {"gender": gender, "age_groups": list(age_groups.values())}


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
