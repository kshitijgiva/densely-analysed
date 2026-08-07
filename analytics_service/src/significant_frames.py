"""
Pull "significant" frames out of a (potentially all-day) camera feed instead
of saving/processing every frame.

Two independent signals decide significance:
  - footfall change: the tracked person count differs from the count at the
    last saved frame (someone entered/left).
  - movement change: a cheap background-subtraction motion score exceeds a
    threshold, even if the person count didn't change (e.g. people
    rearranging within the frame).

A per-frame MOG2 background-subtraction check also acts as a cheap gate
*before* running detection: frames with near-zero motion never reach YOLO at
all, which is what keeps this usable on a full day of near-empty footage.

video_source can be a file path, an RTSP/HTTP stream URL, or a webcam index
(0) - cv2.VideoCapture handles all three the same way, so this runs against
a live camera feed the same way it runs against a stored video.

Every debounced event is saved locally and listed in manifest.json (a full
audit trail), but with --persist only the top --top-k by importance_score
(weighted footfall-change magnitude + motion ratio) are written to Postgres'
significant_frames table via persist.persist_significant_frames - a 350-store
dashboard wants a handful of highlights per store, not every flagged frame.

Usage:
    python significant_frames.py --video-source ../../data/raw/sample_video.mp4 [--show]
    python significant_frames.py --video-source rtsp://... --store-id store_12 \
        --camera-id camera_1 --persist --top-k 10
"""
import argparse
import json
import os
import time
from datetime import datetime, timedelta, timezone

import cv2

from config import VIDEO_SOURCE
from detection import load_detection_model, detect_people
from utils import draw_boxes

_ANALYTICS_SERVICE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_OUTPUT_DIR = os.path.join(_ANALYTICS_SERVICE_DIR, "results", "significant_frames")

MOTION_GATE_RATIO = 0.005   # below this, skip detection entirely - nothing moved
MOVEMENT_SIGNIFICANT_RATIO = 0.02  # above this, frame counts as a "movement change"
MIN_GAP_SECONDS = 2.0       # debounce - don't save more than one frame per this many seconds
TOP_K_PER_RUN = 10          # dashboard highlight cap when --persist is used


def _motion_ratio(bg_subtractor, frame, resize_width=320):
    h, w = frame.shape[:2]
    small = cv2.resize(frame, (resize_width, int(resize_width * h / w)))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    fgmask = bg_subtractor.apply(gray)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)  # drop MOG2's shadow value (127)
    return cv2.countNonZero(fgmask) / fgmask.size


def run(video_source, output_dir=DEFAULT_OUTPUT_DIR,
        motion_gate_ratio=MOTION_GATE_RATIO,
        movement_significant_ratio=MOVEMENT_SIGNIFICANT_RATIO,
        min_gap_seconds=MIN_GAP_SECONDS, max_frames=None, show=False,
        store_id=None, camera_id=None, persist=False, top_k=TOP_K_PER_RUN,
        run_start=None):
    """store_id/camera_id/persist: when persist=True, the top `top_k` events
    by importance_score are written to Postgres (see persist.py) - store_id
    and camera_id are then required. run_start anchors video_time_seconds to
    a wall-clock timestamp (defaults to now(), appropriate for a live feed;
    pass an explicit run_start for a recorded video whose real capture time
    is known)."""
    if persist and not (store_id and camera_id):
        raise ValueError("--persist requires --store-id and --camera-id")

    os.makedirs(output_dir, exist_ok=True)
    run_start = run_start or datetime.now(timezone.utc)

    detection_model = load_detection_model()
    cap = cv2.VideoCapture(video_source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)

    manifest = []
    last_significant_count = None
    last_saved_time = -min_gap_seconds
    frame_idx = 0
    frames_detected_on = 0
    start_time = time.time()

    while max_frames is None or frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        video_time = frame_idx / fps

        motion_ratio = _motion_ratio(bg_subtractor, frame)
        if motion_ratio < motion_gate_ratio:
            if show:
                cv2.imshow("Significant Frames", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            continue

        frames_detected_on += 1
        results = detect_people(detection_model, frame)
        person_count = sum(1 for box in results[0].boxes if box.id is not None)

        first_detection = last_significant_count is None and person_count > 0
        footfall_change = last_significant_count is not None and person_count != last_significant_count
        movement_change = motion_ratio >= movement_significant_ratio
        enough_gap = (video_time - last_saved_time) >= min_gap_seconds

        if (first_detection or footfall_change or movement_change) and enough_gap:
            reasons = [r for r, flag in (
                ("first_detection", first_detection),
                ("footfall_change", footfall_change),
                ("movement_change", movement_change),
            ) if flag]

            filename = f"frame_{frame_idx:07d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)

            count_delta = abs(person_count - (last_significant_count or 0))
            # Footfall change is the stronger signal for "worth a dashboard
            # slot" than raw motion - weighted higher so a busy-but-static
            # frame doesn't crowd out an actual entry/exit event.
            importance_score = round(2.0 * count_delta + motion_ratio, 4)

            manifest.append({
                "frame_idx": frame_idx,
                "video_time_seconds": round(video_time, 2),
                "event_time": run_start + timedelta(seconds=video_time),
                "person_count": person_count,
                "motion_ratio": round(motion_ratio, 4),
                "reasons": reasons,
                "importance_score": importance_score,
                "image_path": filepath,
            })
            print(f"[frame {frame_idx}] SIGNIFICANT ({', '.join(reasons)}) "
                  f"- {person_count} people, motion={motion_ratio:.3f}, "
                  f"importance={importance_score:.3f}")

            last_significant_count = person_count
            last_saved_time = video_time

        if show:
            annotated = draw_boxes(frame.copy(), results, {}, {})
            if manifest and manifest[-1]["frame_idx"] == frame_idx:
                cv2.putText(annotated, "SIGNIFICANT", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            cv2.imshow("Significant Frames", annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if show:
        cv2.destroyAllWindows()

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)  # default=str: serialize event_time (datetime)

    elapsed = time.time() - start_time
    print(f"\nProcessed {frame_idx} frames ({frames_detected_on} passed the motion gate) "
          f"in {elapsed:.1f}s")
    print(f"Saved {len(manifest)} significant frames to {output_dir}")
    print(f"Manifest: {manifest_path}")

    if persist:
        from persist import persist_significant_frames

        highlights = sorted(manifest, key=lambda e: e["importance_score"], reverse=True)[:top_k]
        persist_significant_frames(highlights, store_id, camera_id)
        print(f"Persisted {len(highlights)}/{len(manifest)} highlights to Postgres "
              f"(store={store_id}, camera={camera_id})")

    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-source", default=VIDEO_SOURCE,
                         help="File path, RTSP/HTTP stream URL, or webcam index")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--motion-gate-ratio", type=float, default=MOTION_GATE_RATIO,
                         help="Below this fraction of changed pixels, skip detection entirely")
    parser.add_argument("--movement-significant-ratio", type=float, default=MOVEMENT_SIGNIFICANT_RATIO,
                         help="Above this fraction of changed pixels, save the frame even if footfall count is unchanged")
    parser.add_argument("--min-gap-seconds", type=float, default=MIN_GAP_SECONDS,
                         help="Minimum time between saved frames")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--show", action="store_true", help="Display a live preview window")
    parser.add_argument("--store-id", default=None)
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--persist", action="store_true",
                         help="Write the top --top-k highlights to Postgres (requires --store-id/--camera-id "
                              "and the db service from docker-compose.yaml)")
    parser.add_argument("--top-k", type=int, default=TOP_K_PER_RUN,
                         help="Max highlights persisted per run, ranked by importance_score")
    args = parser.parse_args()

    video_source = args.video_source
    if isinstance(video_source, str) and video_source.isdigit():
        video_source = int(video_source)  # webcam index

    run(video_source, output_dir=args.output_dir,
        motion_gate_ratio=args.motion_gate_ratio,
        movement_significant_ratio=args.movement_significant_ratio,
        min_gap_seconds=args.min_gap_seconds,
        max_frames=args.max_frames, show=args.show,
        store_id=args.store_id, camera_id=args.camera_id,
        persist=args.persist, top_k=args.top_k)
