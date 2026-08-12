"""
Filters out tracked "people" that never actually move - mannequins, printed
posters, and on-screen images of people all detect and track as a person
just fine (YOLO+ByteTrack+OSNet have no notion of "alive"), but unlike a real
shopper they stay in the same spot for most of the clip.

Filtering happens per resolved *identity*, not per raw ByteTrack track_id:
sparse sampling (see render_tracked_video.py) means a static object's track
often fragments into several track_ids across the video (large gaps between
sampled frames break ByteTrack's buffer), which OSNet re-id then stitches
back into one identity. So "has this thing ever moved" has to be judged
after re-id, by resolving every detection's track_id through the final
track_id -> identity_id map.
"""
import math
from collections import defaultdict

from config import (
    STATIC_OBJECT_MAX_DISPLACEMENT_RATIO,
    STATIC_OBJECT_MIN_APPEARANCES,
    STATIC_OBJECT_MIN_SPAN_SECONDS,
)
from heatmap import foot_point


def compute_static_identity_ids(
    track_id_to_identity,
    detections_log,
    frame_width,
    frame_height,
    fps,
    min_span_seconds=STATIC_OBJECT_MIN_SPAN_SECONDS,
    max_displacement_ratio=STATIC_OBJECT_MAX_DISPLACEMENT_RATIO,
    min_appearances=STATIC_OBJECT_MIN_APPEARANCES,
):
    """Return the set of identity_ids that never moved enough, for long
    enough, to plausibly be a live person rather than a static prop.

    detections_log entries are (frame_idx, track_id, features, bbox), as
    collected by render_tracked_video.run(). track_id_to_identity is the
    final track_id -> identity_id map from that same run.
    """
    frame_diag = math.hypot(frame_width, frame_height)
    if frame_diag <= 0 or fps <= 0:
        return set()

    points_by_identity = defaultdict(list)  # identity_id -> [(frame_idx, x, y)]
    for frame_idx, track_id, _features, bbox in detections_log:
        identity_id = track_id_to_identity.get(track_id)
        if identity_id is None:
            continue
        x, y = foot_point(bbox)
        points_by_identity[identity_id].append((frame_idx, x, y))

    static_ids = set()
    for identity_id, points in points_by_identity.items():
        if len(points) < min_appearances:
            continue

        first_frame = min(p[0] for p in points)
        last_frame = max(p[0] for p in points)
        span_seconds = (last_frame - first_frame) / fps
        if span_seconds < min_span_seconds:
            continue

        xs = [p[1] for p in points]
        ys = [p[2] for p in points]
        displacement = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if displacement / frame_diag <= max_displacement_ratio:
            static_ids.add(identity_id)

    return static_ids
