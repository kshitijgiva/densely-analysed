"""
Lightweight run-history store for the analytics pipeline, using SQLite instead
of Postgres/TimescaleDB - so the anomaly-alerting feature works today without
needing Docker/M4 stood up. Schema is intentionally small: one row per
pipeline run, holding just the aggregate numbers alerts.py needs a baseline
for. Swap this for a real Postgres table (see the design doc's persons/
entry_exit_logs schema) once M4 exists - the row shape below maps onto an
aggregated view of that data.
"""
import json
import os
import sqlite3
import time

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "metrics_history.db")
DB_PATH = os.path.normpath(DB_PATH)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at REAL NOT NULL,
    video_source TEXT,
    segment_label TEXT,
    frames_processed INTEGER,
    avg_people_per_frame REAL,
    max_people_in_a_frame INTEGER,
    unique_identities INTEGER,
    gender_male_count INTEGER,
    gender_female_count INTEGER,
    avg_gender_confidence REAL,
    avg_age_confidence REAL,
    age_group_breakdown TEXT
);
"""


def _connect():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(_SCHEMA)
    return conn


def log_run(report, video_source, segment_label=None):
    """Append one pipeline run's metrics report (see render_tracked_video.py's
    `report` dict) as a row. Safe to call even if report['demographics_m3'] is
    None (run_demographics=False) - gender/age fields are just left null."""
    demo = report.get("demographics_m3") or {}
    gender_breakdown = demo.get("gender_breakdown", {})

    conn = _connect()
    with conn:
        conn.execute(
            """INSERT INTO runs (
                logged_at, video_source, segment_label, frames_processed,
                avg_people_per_frame, max_people_in_a_frame, unique_identities,
                gender_male_count, gender_female_count,
                avg_gender_confidence, avg_age_confidence, age_group_breakdown
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(), video_source, segment_label, report["frames_processed"],
                report["detection"]["avg_people_per_frame"],
                report["detection"]["max_people_in_a_frame"],
                report["reid_m2"]["unique_persistent_identities"],
                gender_breakdown.get("male"), gender_breakdown.get("female"),
                demo.get("avg_gender_confidence"), demo.get("avg_age_confidence"),
                json.dumps(demo.get("age_group_breakdown", {})),
            ),
        )
    conn.close()


def get_runs(video_source=None, limit=100):
    """Most recent runs first, optionally filtered to one video source."""
    conn = _connect()
    conn.row_factory = sqlite3.Row
    if video_source:
        rows = conn.execute(
            "SELECT * FROM runs WHERE video_source = ? ORDER BY id DESC LIMIT ?",
            (video_source, limit),
        ).fetchall()
    else:
        rows = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]
