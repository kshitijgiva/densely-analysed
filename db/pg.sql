-- Schema for the Store CCTV Analytics Platform (design doc S4).
-- Requires TimescaleDB (see docker-compose.yaml, which uses the
-- timescale/timescaledb image). Aggregated stats only - no raw imagery or
-- re-id vectors live here (those stay in ChromaDB with a short TTL); this
-- schema is safe to retain long-term per the design doc's S6 retention policy.

CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS stores (
    store_id    VARCHAR(64) PRIMARY KEY,
    store       VARCHAR(255) NOT NULL,
    camera_url  VARCHAR(255) NOT NULL,
    region      VARCHAR(255) NOT NULL,
    metadata    JSONB DEFAULT '{}'::jsonb,
    inserted_at TIMESTAMPTZ DEFAULT now(),
    updated_at  TIMESTAMPTZ DEFAULT now()
);

-- One row per re-identified person (a "visit"), not a named individual.
CREATE TABLE IF NOT EXISTS persons (
    id                      UUID PRIMARY KEY,
    store_id                VARCHAR(64) REFERENCES stores(store_id),
    camera_id               VARCHAR(64) NOT NULL,
    track_id                INTEGER,
    reid_embedding_ref      TEXT,       -- reference to a ChromaDB vector id, not the vector itself
    gender                  VARCHAR(16),
    gender_confidence       FLOAT,
    age_group               VARCHAR(32),
    age_confidence          FLOAT,
    needs_demographic_retry BOOLEAN DEFAULT FALSE,
    first_seen              TIMESTAMPTZ NOT NULL,
    last_seen               TIMESTAMPTZ NOT NULL,
    metadata                JSONB DEFAULT '{}'::jsonb
);
CREATE INDEX IF NOT EXISTS idx_persons_store_first_seen ON persons (store_id, first_seen);

CREATE TABLE IF NOT EXISTS zones (
    id        TEXT PRIMARY KEY,
    store_id  VARCHAR(64) REFERENCES stores(store_id),
    camera_id VARCHAR(64) NOT NULL,
    name      TEXT,
    polygon   JSONB   -- pixel coordinates defining the zone
);

-- event_time must be part of the key for TimescaleDB hypertable partitioning.
CREATE TABLE IF NOT EXISTS entry_exit_logs (
    id         BIGSERIAL,
    person_id  UUID NOT NULL REFERENCES persons(id),
    store_id   VARCHAR(64) REFERENCES stores(store_id),
    camera_id  VARCHAR(64) NOT NULL,
    zone_id    TEXT REFERENCES zones(id),
    event_type VARCHAR(16) CHECK (event_type IN ('entry', 'exit', 'zone_enter', 'zone_exit')),
    event_time TIMESTAMPTZ NOT NULL,
    PRIMARY KEY (id, event_time)
);
SELECT create_hypertable('entry_exit_logs', 'event_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_entry_exit_person_time ON entry_exit_logs (person_id, event_time);
CREATE INDEX IF NOT EXISTS idx_entry_exit_store_time ON entry_exit_logs (store_id, event_time);

-- Dashboard "highlights" per store/camera - a small, ranked subset of frames
-- flagged by analytics_service/src/significant_frames.py (footfall change or
-- motion-based movement change), not a full frame dump. image_url is a
-- pointer into object storage, same as persons.reid_embedding_ref points into
-- ChromaDB - no raw imagery in this table, per the header note above.
CREATE TABLE IF NOT EXISTS significant_frames (
    id               BIGSERIAL,
    store_id         VARCHAR(64) REFERENCES stores(store_id),
    camera_id        VARCHAR(64) NOT NULL,
    event_time       TIMESTAMPTZ NOT NULL,
    person_count     INTEGER,
    motion_ratio     FLOAT,
    reasons          TEXT[],
    importance_score FLOAT,
    image_url        TEXT NOT NULL,
    PRIMARY KEY (id, event_time)
);
SELECT create_hypertable('significant_frames', 'event_time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_significant_frames_store_time ON significant_frames (store_id, event_time DESC);
CREATE INDEX IF NOT EXISTS idx_significant_frames_store_importance ON significant_frames (store_id, importance_score DESC);
