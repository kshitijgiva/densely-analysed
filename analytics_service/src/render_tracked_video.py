"""
Process a video file end-to-end (detection -> ByteTrack -> OSNet re-id) and
write out an annotated copy with bounding boxes, live track IDs, and
persistent re-id identity IDs burned into each frame. No live display window
required, so this runs headless.

Also collects the same model metrics as validate_pipeline.py (track-ID
stability, re-id similarity distributions) plus live per-frame stats
(processing FPS, detection confidence, people-in-frame count), prints a
summary report at the end, and saves it as JSON next to the output video.

Two demographics backends are available:
    mivolo-body      body-only crop through the transformers-hosted mivolo_v2
                      checkpoint (demographics.py) - default, lighter weight.
    mivolo-official   the official MiVOLO repo's own person+face YOLO detector
                      + fused face+body age/gender model (mivolo_official.py) -
                      requires the checkpoints in weights/mivolo/, and runs a
                      second, much heavier YOLO model per frame that needs a
                      demographics retry.

Usage:
    python render_tracked_video.py [--input PATH] [--output PATH] [--max-frames N]
                                    [--metrics-out PATH] [--demographics-backend {mivolo-body,mivolo-official}]
"""
import argparse
import json
import os
import statistics
import time

import cv2

from config import VIDEO_SOURCE, REID_THRESHOLD
from detection import load_detection_model, detect_people
from reid import OSNetReID
from identity import PersonIdentity, match_identity, needs_demographic_retry
from utils import draw_boxes
from validate_pipeline import percentiles, collect_similarity_pairs


def run(video_source, output_path, max_frames, metrics_out_path,
        run_demographics=True, demographics_backend="mivolo-body"):
    detection_model = load_detection_model()
    reid_model = OSNetReID()

    mivolo_official = None
    if run_demographics:
        if demographics_backend == "mivolo-official":
            import mivolo_official
        else:
            from demographics import estimate_demographics

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    identities = {}            # {identity_id: PersonIdentity}
    track_id_to_identity = {}  # {track_id: identity_id}
    identity_counter = 1000

    track_stats = {}     # track_id -> {"first", "last", "count"} (M1)
    detections_log = []  # (frame_idx, track_id, embedding, bbox) (M2)
    det_confidences = [] # every YOLO detection confidence seen
    people_per_frame = []

    frame_idx = 0
    new_identity_count = 0
    reid_match_count = 0
    proc_fps_ema = None  # exponential moving average of per-frame processing FPS
    start = time.time()

    while max_frames is None or frame_idx < max_frames:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_people(detection_model, frame)
        official_frame_results = None  # lazily computed, cached per-frame

        people_in_frame = 0
        for box in results[0].boxes:
            if box.conf is not None:
                det_confidences.append(float(box.conf.item()))

            if box.id is None:
                continue
            people_in_frame += 1

            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            person_img = frame[y1:y2, x1:x2]

            features = reid_model.extract_features(person_img)
            if features is None:
                continue

            stats = track_stats.setdefault(track_id, {"first": frame_idx, "last": frame_idx, "count": 0})
            stats["last"] = frame_idx
            stats["count"] += 1
            detections_log.append((frame_idx, track_id, features, (x1, y1, x2, y2)))

            if track_id in track_id_to_identity:
                identity_id = track_id_to_identity[track_id]
                identities[identity_id].add_appearance(features, frame_idx)
            else:
                matched_id, similarity = match_identity(features, identities)
                if matched_id is not None and similarity > REID_THRESHOLD:
                    identity_id = matched_id
                    identities[identity_id].add_appearance(features, frame_idx)
                    track_id_to_identity[track_id] = identity_id
                    reid_match_count += 1
                    print(f"[frame {frame_idx}] Re-ID: track {track_id} -> identity {identity_id} (sim={similarity:.3f})")
                else:
                    identity_id = identity_counter
                    identity_counter += 1
                    identities[identity_id] = PersonIdentity(identity_id)
                    identities[identity_id].add_appearance(features, frame_idx)
                    track_id_to_identity[track_id] = identity_id
                    new_identity_count += 1
                    print(f"[frame {frame_idx}] New identity {identity_id} (track {track_id})")

            if run_demographics:
                video_time = frame_idx / fps
                identity = identities[identity_id]
                needs_gender, needs_age = needs_demographic_retry(identity, video_time)
                if needs_gender or needs_age:
                    if demographics_backend == "mivolo-official":
                        if official_frame_results is None:
                            official_frame_results = mivolo_official.estimate_demographics_for_frame(frame)
                        match = mivolo_official.match_bbox((x1, y1, x2, y2), official_frame_results)
                        if match is not None:
                            gender_result = {"gender": match["gender"], "confidence": match["confidence"]}
                            age_result = {"age": match["age_group"], "age_years": match["age_years"],
                                          "confidence": match["confidence"]}
                            if needs_gender and match["gender"] is not None:
                                identity.update_gender(gender_result, video_time)
                            if needs_age and match["age_group"] is not None:
                                identity.update_age(age_result, video_time)
                    else:
                        age_result, gender_result = estimate_demographics(person_img)
                        if needs_gender:
                            identity.update_gender(gender_result, video_time)
                        if needs_age:
                            identity.update_age(age_result, video_time)

        people_per_frame.append(people_in_frame)

        frame = draw_boxes(frame, results, track_id_to_identity, identities)

        frame_proc_time = time.time() - frame_start
        instant_fps = 1.0 / frame_proc_time if frame_proc_time > 0 else 0.0
        proc_fps_ema = instant_fps if proc_fps_ema is None else (0.9 * proc_fps_ema + 0.1 * instant_fps)

        overlay_lines = [
            f"Frame: {frame_idx}" + (f"/{total_frames}" if total_frames > 0 else ""),
            f"Proc FPS: {proc_fps_ema:.1f}",
            f"People in frame: {people_in_frame}",
            f"Identities so far: {len(identities)}",
        ]
        for i, line in enumerate(overlay_lines):
            cv2.putText(frame, line, (10, 30 + 25 * i),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        writer.write(frame)

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"...processed {frame_idx} frames")

    cap.release()
    writer.release()

    elapsed = time.time() - start
    lengths = [s["count"] for s in track_stats.values()]
    fragments = sum(1 for l in lengths if l <= 2)
    same_id_sims, diff_id_sims = collect_similarity_pairs(detections_log)
    same_p = percentiles(same_id_sims)
    diff_p = percentiles(diff_id_sims)

    demographics_m3 = None
    if run_demographics:
        genders = [i.gender for i in identities.values() if i.gender is not None]
        ages = [i.age for i in identities.values() if i.age is not None]
        gender_confs = [i.gender_confidence for i in identities.values() if i.gender is not None]
        age_confs = [i.age_confidence for i in identities.values() if i.age is not None]
        demographics_m3 = {
            "identities_with_gender": len(genders),
            "identities_with_age": len(ages),
            "gender_breakdown": {g: genders.count(g) for g in set(genders)},
            "age_group_breakdown": {a: ages.count(a) for a in set(ages)},
            "avg_gender_confidence": round(statistics.mean(gender_confs), 3) if gender_confs else None,
            "avg_age_confidence": round(statistics.mean(age_confs), 3) if age_confs else None,
        }

    report = {
        "frames_processed": frame_idx,
        "elapsed_seconds": round(elapsed, 1),
        "avg_processing_fps": round(frame_idx / max(elapsed, 1e-9), 2),
        "detection": {
            "count": len(det_confidences),
            "avg_confidence": round(statistics.mean(det_confidences), 3) if det_confidences else None,
            "min_confidence": round(min(det_confidences), 3) if det_confidences else None,
            "max_people_in_a_frame": max(people_per_frame) if people_per_frame else 0,
            "avg_people_per_frame": round(statistics.mean(people_per_frame), 2) if people_per_frame else 0,
        },
        "tracking_m1": {
            "unique_track_ids": len(track_stats),
            "track_length_min": min(lengths) if lengths else None,
            "track_length_median": statistics.median(lengths) if lengths else None,
            "track_length_max": max(lengths) if lengths else None,
            "fragment_tracks_pct": round(100 * fragments / len(lengths), 1) if lengths else None,
        },
        "reid_m2": {
            "reid_threshold": REID_THRESHOLD,
            "same_person_pairs": len(same_id_sims),
            "diff_person_pairs": len(diff_id_sims),
            "same_person_sim_p5": same_p[5],
            "same_person_sim_median": same_p[50],
            "diff_person_sim_p95": diff_p[95],
            "unique_persistent_identities": len(identities),
            "new_identities": new_identity_count,
            "reid_merges": reid_match_count,
        },
        "demographics_m3": demographics_m3,
    }

    print(f"\n=== Run summary ===")
    print(f"Processed {frame_idx} frames in {elapsed:.1f}s ({report['avg_processing_fps']:.1f} FPS)")
    print(f"Detections: {report['detection']['count']} "
          f"(avg conf {report['detection']['avg_confidence']}, "
          f"max {report['detection']['max_people_in_a_frame']} people/frame)")
    print(f"M1 tracking: {report['tracking_m1']['unique_track_ids']} tracks, "
          f"length min/median/max = {report['tracking_m1']['track_length_min']}/"
          f"{report['tracking_m1']['track_length_median']}/{report['tracking_m1']['track_length_max']}, "
          f"{report['tracking_m1']['fragment_tracks_pct']}% fragments")
    print(f"M2 re-id: same-person sim p5={same_p[5]}, diff-person sim p95={diff_p[95]} "
          f"-> {len(identities)} persistent identities ({new_identity_count} new, {reid_match_count} merges)")
    if demographics_m3:
        print(f"M3 demographics: gender breakdown {demographics_m3['gender_breakdown']}, "
              f"age breakdown {demographics_m3['age_group_breakdown']}, "
              f"avg confidence (gender/age) = {demographics_m3['avg_gender_confidence']}/"
              f"{demographics_m3['avg_age_confidence']}")
    print(f"Annotated video written to {output_path}")

    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)
    with open(metrics_out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics report written to {metrics_out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=VIDEO_SOURCE)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "tracked_output.mp4"))
    parser.add_argument("--metrics-out", default=None,
                         help="Where to save the JSON metrics report (default: alongside --output, same basename)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-demographics", action="store_true",
                         help="Skip age/gender estimation (M3) - faster, M1/M2 tracking only")
    parser.add_argument("--demographics-backend", choices=["mivolo-body", "mivolo-official"],
                         default="mivolo-body",
                         help="mivolo-body: transformers mivolo_v2, body-only (default). "
                              "mivolo-official: official repo's own detector + fused face+body model, heavier.")
    args = parser.parse_args()

    output_path = os.path.normpath(args.output)
    if args.metrics_out:
        metrics_out_path = os.path.normpath(args.metrics_out)
    else:
        base, _ = os.path.splitext(output_path)
        metrics_out_path = base + "_metrics.json"

    run(args.input, output_path, args.max_frames, metrics_out_path,
        run_demographics=not args.no_demographics,
        demographics_backend=args.demographics_backend)
