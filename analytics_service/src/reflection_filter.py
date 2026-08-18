"""
Flags resolved identities that are actually mirror reflections of another,
concurrently-visible identity, rather than a second real person.

Why this can't just be a merge in identity.py: a real person and their
mirror reflection produce two boxes in the *same frame*, and match_identity's
exclude_ids rule (see identity.py) specifically forbids merging two
simultaneously-visible boxes into one identity - that rule exists to stop
two real people standing together from collapsing into one, but it also
means a reflection can never merge back into the real person's identity and
instead mints a brand-new one. This module catches that after the fact, the
same way static_objects.py catches static mannequins/posters post-hoc:

  1. overlap: does this identity's bbox mostly sit inside the camera's
     mirror ROI (from mirror_segmentation.py)?
  2. correspondence: is there another, concurrently-visible identity that
     sits mostly *outside* the mirror, looks similar (OSNet embedding), and
     moves in lockstep with it (correlated foot-point displacement)?

Only identities that pass both checks are dropped - overlap alone is too
weak a signal (a real person can walk in front of a mirror without being
one), so this never discards an identity that has no plausible real-person
counterpart to be a reflection *of*.
"""
import math
from collections import defaultdict

import numpy as np
from scipy.spatial.distance import cosine

from config import (
    MIRROR_APPEARANCE_SIMILARITY_THRESHOLD,
    MIRROR_MIN_APPEARANCES,
    MIRROR_MIN_SHARED_FRAMES,
    MIRROR_MOTION_CORRELATION_THRESHOLD,
    MIRROR_OVERLAP_THRESHOLD,
    MIRROR_REAL_OVERLAP_MAX,
)
from heatmap import foot_point


def _bbox_mirror_overlap(bbox, mirror_mask, frame_height, frame_width):
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), frame_width))
    x2 = max(0, min(int(x2), frame_width))
    y1 = max(0, min(int(y1), frame_height))
    y2 = max(0, min(int(y2), frame_height))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    region = mirror_mask[y1:y2, x1:x2]
    return float(region.mean()) if region.size else 0.0


def _motion_correlation(points_a, points_b, min_shared_frames):
    """Pearson correlation of frame-to-frame foot-point displacement
    magnitude between two identities, over the frames they share. A
    reflection moves in lockstep with the real person casting it, so their
    displacement-magnitude time series should be strongly correlated even
    though their absolute positions differ."""
    shared_frames = sorted(set(points_a) & set(points_b))
    if len(shared_frames) < min_shared_frames + 1:
        return 0.0

    disp_a, disp_b = [], []
    for prev_f, cur_f in zip(shared_frames, shared_frames[1:]):
        ax1, ay1 = points_a[prev_f]
        ax2, ay2 = points_a[cur_f]
        bx1, by1 = points_b[prev_f]
        bx2, by2 = points_b[cur_f]
        disp_a.append(math.hypot(ax2 - ax1, ay2 - ay1))
        disp_b.append(math.hypot(bx2 - bx1, by2 - by1))

    if len(disp_a) < min_shared_frames:
        return 0.0

    disp_a, disp_b = np.array(disp_a), np.array(disp_b)
    std_a, std_b = disp_a.std(), disp_b.std()
    if std_a == 0 and std_b == 0:
        # Pearson correlation is undefined for two constant sequences, but a
        # person walking at a steady pace past a mirror is exactly the case
        # we most want to catch - constant, near-equal speed is itself
        # strong lockstep evidence, not an absence of signal.
        mean_a, mean_b = disp_a.mean(), disp_b.mean()
        if max(mean_a, mean_b) < 1e-6:
            return 0.0  # both stationary - no motion evidence either way
        return 1.0 - abs(mean_a - mean_b) / max(mean_a, mean_b)
    if std_a == 0 or std_b == 0:
        return 0.0  # one moved steadily, the other didn't - not lockstep
    return float(np.corrcoef(disp_a, disp_b)[0, 1])


def compute_reflection_identity_ids(
    track_id_to_identity,
    detections_log,
    identities,
    mirror_mask,
    frame_width,
    frame_height,
    fps,
    min_appearances=MIRROR_MIN_APPEARANCES,
    overlap_threshold=MIRROR_OVERLAP_THRESHOLD,
    real_overlap_max=MIRROR_REAL_OVERLAP_MAX,
    appearance_similarity_threshold=MIRROR_APPEARANCE_SIMILARITY_THRESHOLD,
    motion_correlation_threshold=MIRROR_MOTION_CORRELATION_THRESHOLD,
    min_shared_frames=MIRROR_MIN_SHARED_FRAMES,
):
    """Return the set of identity_ids judged to be mirror reflections of
    another identity in `identities`, and therefore excludable from footfall.

    detections_log entries are (frame_idx, track_id, features, bbox), as
    collected by render_tracked_video.run(). track_id_to_identity is that
    same run's final track_id -> identity_id map. mirror_mask is the boolean
    (frame_height, frame_width) array from mirror_segmentation.py.
    """
    if mirror_mask is None:
        return set()

    foot_points = defaultdict(dict)   # identity_id -> {frame_idx: (x, y)}
    overlaps = defaultdict(list)      # identity_id -> [overlap_ratio, ...]
    for frame_idx, track_id, _features, bbox in detections_log:
        identity_id = track_id_to_identity.get(track_id)
        if identity_id is None or identity_id not in identities:
            continue  # dangling mapping to an identity already dropped (e.g. static-object filter)
        foot_points[identity_id][frame_idx] = foot_point(bbox)
        overlaps[identity_id].append(
            _bbox_mirror_overlap(bbox, mirror_mask, frame_height, frame_width)
        )

    candidates = {
        identity_id
        for identity_id, ratios in overlaps.items()
        if len(ratios) >= min_appearances and (sum(ratios) / len(ratios)) >= overlap_threshold
    }
    if not candidates:
        return set()

    mean_overlap = {
        identity_id: sum(ratios) / len(ratios) for identity_id, ratios in overlaps.items()
    }
    representative = {
        identity_id: identities[identity_id].representative_embedding()
        for identity_id in identities
    }

    reflection_ids = set()
    for candidate_id in candidates:
        candidate_embedding = representative.get(candidate_id)
        if candidate_embedding is None:
            continue

        for other_id, other_identity in identities.items():
            if other_id == candidate_id or other_id in candidates:
                continue  # a reflection needs a real counterpart, not another reflection
            if mean_overlap.get(other_id, 0.0) > real_overlap_max:
                continue  # counterpart must itself sit mostly outside the mirror

            other_embedding = representative.get(other_id)
            if other_embedding is None:
                continue
            similarity = 1 - cosine(candidate_embedding, other_embedding)
            if similarity < appearance_similarity_threshold:
                continue

            correlation = _motion_correlation(
                foot_points[candidate_id], foot_points[other_id], min_shared_frames
            )
            if correlation >= motion_correlation_threshold:
                reflection_ids.add(candidate_id)
                break

    return reflection_ids
