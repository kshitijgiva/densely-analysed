import os

import psycopg2
from psycopg2.extras import RealDictCursor

def get_connection():
    return psycopg2.connect(
        dbname=os.environ.get("DB_NAME", "cctv_analytics"),
        user=os.environ.get("DB_USER", "cctv"),
        password=os.environ.get("DB_PASSWORD", "cctvpass"),
        host=os.environ.get("DB_HOST", "localhost"),
        port=5432
    )

def upsert_person(person_id, gender, age_group, metadata=None):
    """Insert or update a person record."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO persons (person_id, gender, age_group, metadata)
        VALUES (%s, %s, %s, COALESCE(%s, '{}'::jsonb))
        ON CONFLICT (person_id) DO UPDATE SET
            gender = EXCLUDED.gender,
            age_group = EXCLUDED.age_group,
            metadata = EXCLUDED.metadata,
            updated_at = CURRENT_TIMESTAMP;
    """, (person_id, gender, age_group, metadata))
    conn.commit()
    cur.close()
    conn.close()

def insert_entry_exit(person_id, event_type, store_id="default_store", metadata=None, event_time=None):
    """Insert an entry or exit log."""
    conn = get_connection()
    cur = conn.cursor()
    if event_time:
        cur.execute("""
            INSERT INTO entry_exit_logs (person_id, store_id, type, timestamp, metadata)
            VALUES (%s, %s, %s, %s, COALESCE(%s, '{}'::jsonb));
        """, (person_id, store_id, event_type, event_time, metadata))
    else:
        cur.execute("""
            INSERT INTO entry_exit_logs (person_id, store_id, type, metadata)
            VALUES (%s, %s, %s, COALESCE(%s, '{}'::jsonb));
        """, (person_id, store_id, event_type, metadata))
    conn.commit()
    cur.close()
    conn.close()
