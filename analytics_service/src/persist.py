"""
Bridge between an in-memory pipeline run (identities produced by detection ->
ByteTrack -> OSNet re-id -> MiVOLO demographics) and the Postgres/TimescaleDB
schema from db/pg.sql (M4).

The pipeline scripts key identities by a small in-process int for display, while
each PersonIdentity carries the stable UUID shared by Postgres and ChromaDB.
This module writes one `persons` row plus an 'entry'/'exit' event pair per
identity - there's no zone/dwell logic yet (design doc S5), so those are the
only two events available.

Frame indices aren't wall-clock time, so event timestamps are reconstructed
as `run_start + frame_idx / fps` - real enough to preserve relative ordering
and durations within a single run, even though the source video's own
recording time is unknown.
"""
import sys
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root, for `db`
from db.postgres import (  # noqa: E402
    insert_entry_exit_event,
    insert_significant_frame,
    upsert_person,
    upsert_store,
)


def persist_identities(identities, store_id, camera_id, fps, run_start,
                        store_name=None, camera_url=None, region=None):
    """identities: {identity_id: PersonIdentity}, as built by the pipeline scripts."""
    upsert_store(
        store_id,
        store=store_name or store_id,
        camera_url=camera_url or "unknown",
        region=region or "unknown",
    )

    persisted = 0
    for identity in identities.values():
        if not identity.is_footfall_eligible():
            continue

        person_id = identity.person_id
        first_seen = run_start + timedelta(seconds=identity.first_seen / fps)
        last_seen = run_start + timedelta(seconds=identity.last_seen / fps)

        upsert_person(
            person_id=person_id,
            store_id=store_id,
            camera_id=camera_id,
            track_id=None,  # multiple raw track_ids can map to one identity via re-id merges
            gender=identity.gender,
            gender_confidence=identity.gender_confidence,
            age_group=identity.age,
            age_confidence=identity.age_confidence,
            needs_demographic_retry=identity.gender is None or identity.age is None,
            first_seen=first_seen,
            last_seen=last_seen,
            reid_embedding_ref=identity.reid_embedding_ref,
        )
        insert_entry_exit_event(person_id, store_id, camera_id, "entry", first_seen)
        insert_entry_exit_event(person_id, store_id, camera_id, "exit", last_seen)
        persisted += 1

    return persisted


def persist_significant_frames(events, store_id, camera_id):
    """events: manifest entries from significant_frames.py's run() - each needs
    event_time/person_count/motion_ratio/reasons/importance_score/image_path.
    Only image_path (a URL/path reference) is written to Postgres, not the
    image itself - see db/pg.sql's note on significant_frames.image_url."""
    for event in events:
        insert_significant_frame(
            store_id=store_id,
            camera_id=camera_id,
            event_time=event["event_time"],
            person_count=event["person_count"],
            motion_ratio=event["motion_ratio"],
            reasons=event["reasons"],
            importance_score=event["importance_score"],
            image_url=event["image_path"],
        )
    return len(events)
