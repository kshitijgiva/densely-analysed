"""
Validation harness for M1 (detection + ByteTrack) and M2 (OSNet re-id).

M1 check: are track IDs stable over the test footage, or fragmented/switching?
M2 check: does the cosine-similarity re-id threshold merge broken tracks back
into the same person without merging two different people together (the
"did it double count" spot-check)?

Pass --sample-frames/--sample-window-seconds to validate under the same sparse
sampling render_tracked_video.py/analytics_api.py use for big videos. Same-person
similarity is measured between *consecutive processed frames*, so validating on
every frame (the default) only tells you what the threshold should be for
dense, frame-by-frame tracking (realtime.py) - a few-seconds-apart sample gap
gives naturally lower same-person similarity (motion, pose, lighting), and
reusing the dense-tracking threshold there fragments one real person into
several new identities, inflating footfall with no real increase in people.

Usage:
    python validate_pipeline.py [--max-frames N] [--threshold F]
    python validate_pipeline.py --sample-frames 3 --sample-window-seconds 10

Outputs a console report plus a per-detection CSV at
../results/reid_validation_log.csv for manual spot-checking.
"""
import argparse
import csv
import os
import statistics
import time

import cv2

from config import VIDEO_SOURCE, REID_THRESHOLD
from detection import load_detection_model, detect_people
from reid import OSNetReID
from identity import PersonIdentity, match_identity


def cosine_sim(a, b):
    return float((a * b).sum())  # both vectors are L2-normalized by OSNetReID


def run(video_source, max_frames, threshold, sample_frames=0, sample_window_seconds=10):
    if sample_frames < 0:
        raise ValueError("sample_frames cannot be negative")
    if sample_frames > 0 and sample_window_seconds <= 0:
        raise ValueError("sample_window_seconds must be positive")

    detection_model = load_detection_model()
    reid_model = OSNetReID()

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_interval = 1.0
    if sample_frames > 0:
        sample_interval = fps * sample_window_seconds / sample_frames
        print(f"Sampling {sample_frames} frames per {sample_window_seconds}s "
              f"(every {sample_interval:.1f} source frames) - matches the sparse-sampling "
              f"path in render_tracked_video.py/analytics_api.py")

    track_stats = {}   # track_id -> {"first": frame_idx, "last": frame_idx, "count": int}
    detections = []    # (frame_idx, track_id, embedding, bbox)

    frame_idx = 0
    processed_frames = 0
    next_sample_frame = 0.0
    start = time.time()
    while max_frames is None or processed_frames < max_frames:
        if sample_frames > 0:
            frame_idx = int(round(next_sample_frame))
            if total_frames > 0 and frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            next_sample_frame += sample_interval
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_people(detection_model, frame)
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            embedding = reid_model.extract_features(crop)
            if embedding is None:
                continue

            stats = track_stats.setdefault(track_id, {"first": frame_idx, "last": frame_idx, "count": 0})
            stats["last"] = frame_idx
            stats["count"] += 1

            detections.append((frame_idx, track_id, embedding, (x1, y1, x2, y2)))

        processed_frames += 1
        if sample_frames == 0:
            frame_idx += 1
        if processed_frames % 100 == 0:
            print(f"...processed {processed_frames} frames")

    cap.release()
    elapsed = time.time() - start
    print(f"\nProcessed {processed_frames} frames in {elapsed:.1f}s ({processed_frames / elapsed:.1f} FPS)")

    report_m1(track_stats)
    same_id_sims, diff_id_sims = collect_similarity_pairs(detections)
    report_m2_threshold(same_id_sims, diff_id_sims)
    report_m2_identity_merge(detections, threshold)


def report_m1(track_stats):
    print("\n=== M1: track ID stability ===")
    lengths = [s["count"] for s in track_stats.values()]
    if not lengths:
        print("No tracks detected.")
        return

    print(f"Unique track IDs: {len(track_stats)}")
    print(f"Track length (detections/track) - min: {min(lengths)}, "
          f"median: {statistics.median(lengths):.1f}, max: {max(lengths)}")

    fragments = sum(1 for l in lengths if l <= 2)
    print(f"Tracks with <=2 detections (likely fragments / spurious IDs): "
          f"{fragments} ({100 * fragments / len(lengths):.1f}%)")


def collect_similarity_pairs(detections):
    """Positive pairs: same track_id, different times. Negative pairs: different
    track_ids visible in the same frame (guaranteed different people)."""
    by_track = {}
    by_frame = {}
    for frame_idx, track_id, embedding, _ in detections:
        by_track.setdefault(track_id, []).append(embedding)
        by_frame.setdefault(frame_idx, []).append((track_id, embedding))

    same_id_sims = []
    for embeddings in by_track.values():
        for i in range(len(embeddings) - 1):
            same_id_sims.append(cosine_sim(embeddings[i], embeddings[i + 1]))

    diff_id_sims = []
    for pairs in by_frame.values():
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                if pairs[i][0] != pairs[j][0]:
                    diff_id_sims.append(cosine_sim(pairs[i][1], pairs[j][1]))

    return same_id_sims, diff_id_sims


def percentiles(values, ps=(0, 5, 25, 50, 75, 95, 100)):
    if not values:
        return {p: None for p in ps}
    values = sorted(values)
    out = {}
    for p in ps:
        idx = min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))
        out[p] = values[idx]
    return out


def report_m2_threshold(same_id_sims, diff_id_sims):
    print("\n=== M2: re-id similarity distributions ===")
    print(f"Same-person pairs (same track, consecutive detections): {len(same_id_sims)}")
    print(f"Different-person pairs (different tracks, same frame): {len(diff_id_sims)}")

    same_p = percentiles(same_id_sims)
    diff_p = percentiles(diff_id_sims)
    print(f"Same-person similarity  - p5: {same_p[5]}, median: {same_p[50]}, p95: {same_p[95]}")
    print(f"Diff-person similarity  - p5: {diff_p[5]}, median: {diff_p[50]}, p95: {diff_p[95]}")

    if same_id_sims and diff_id_sims:
        suggested = (same_p[5] + diff_p[95]) / 2
        print(f"Suggested threshold (midpoint of same-p5 / diff-p95): {suggested:.3f}")
        if same_p[5] < diff_p[95]:
            print("WARNING: same-person and different-person distributions overlap - "
                  "no single threshold perfectly separates them on this footage.")
    else:
        print("Not enough pairs to suggest a threshold from this footage.")


def report_m2_identity_merge(detections, threshold):
    print(f"\n=== M2: identity merge simulation (threshold={threshold}) ===")

    identities = {}
    track_id_to_identity = {}
    identity_counter = 1000
    false_merges = 0
    log_rows = []

    detections_sorted = sorted(detections, key=lambda d: d[0])
    seen_identities_this_frame = {}

    for frame_idx, track_id, embedding, bbox in detections_sorted:
        claimed = seen_identities_this_frame.setdefault(frame_idx, set())

        if track_id in track_id_to_identity:
            identity_id = track_id_to_identity[track_id]
            identities[identity_id].add_appearance(embedding, frame_idx)
            event, similarity = "existing_track", None
        else:
            matched_id, similarity = match_identity(embedding, identities, exclude_ids=claimed)
            if matched_id is not None and similarity > threshold:
                identity_id = matched_id
                identities[identity_id].add_appearance(embedding, frame_idx)
                track_id_to_identity[track_id] = identity_id
                event = "reid_matched"
            else:
                identity_id = identity_counter
                identity_counter += 1
                identity = PersonIdentity(identity_id)
                identity.add_appearance(embedding, frame_idx)
                identities[identity_id] = identity
                track_id_to_identity[track_id] = identity_id
                event = "new_identity"

        if identity_id in seen_identities_this_frame[frame_idx]:
            false_merges += 1
            event += "_FALSE_MERGE"
        seen_identities_this_frame[frame_idx].add(identity_id)

        log_rows.append([frame_idx, track_id, identity_id, event, similarity])

    unique_tracks = len({row[1] for row in log_rows})
    unique_identities = len({row[2] for row in log_rows})
    print(f"Unique track IDs: {unique_tracks}")
    print(f"Unique persistent identities after re-id merge: {unique_identities}")
    print(f"Same-frame false merges (two people collapsed into one identity): {false_merges}")
    if false_merges:
        print("^ inspect these _FALSE_MERGE rows in the CSV log - either the threshold is too "
              "low, or two different people happen to look very similar in appearance (e.g. "
              "matching clothing) to OSNet at this threshold.")

    log_path = os.path.join(os.path.dirname(__file__), "..", "results", "reid_validation_log.csv")
    log_path = os.path.normpath(log_path)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "track_id", "identity_id", "event", "similarity"])
        writer.writerows(log_rows)
    print(f"Per-detection log written to {log_path} for manual spot-checking.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-source", default=VIDEO_SOURCE,
                         help="File path, RTSP/HTTP stream URL, or webcam index")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=REID_THRESHOLD)
    parser.add_argument("--sample-frames", type=int, default=0,
                         help="Process this many evenly spaced frames per sampling window "
                              "(0 = every frame). Set to match the analytics_api.py/"
                              "render_tracked_video.py sampling you're validating.")
    parser.add_argument("--sample-window-seconds", type=float, default=10,
                         help="Sampling window size in seconds (used with --sample-frames)")
    args = parser.parse_args()

    run(args.video_source, args.max_frames, args.threshold,
        sample_frames=args.sample_frames,
        sample_window_seconds=args.sample_window_seconds)
