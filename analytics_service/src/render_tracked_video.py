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

from config import VIDEO_SOURCE, REID_THRESHOLD, MIN_DEMOGRAPHICS_CONFIDENCE
from detection import load_detection_model, detect_people
from reid import OSNetReID
from identity import (
    PersonIdentity,
    match_identity,
    needs_demographic_retry,
    demographics_pass_threshold,
)
from utils import draw_boxes
from validate_pipeline import percentiles, collect_similarity_pairs
from heatmap import HeatmapAccumulator, render_heatmap
import metrics_store


def run(video_source, output_path, max_frames, metrics_out_path,
        run_demographics=True, demographics_backend="mivolo-body",
        use_chroma=False, store_id="store_1", camera_id="camera_1",
        sample_frames=0, sample_window_seconds=10, write_video=True,
        reid_threshold=REID_THRESHOLD):
    """reid_threshold defaults to the config value, which was tuned on dense,
    frame-by-frame footage (see validate_pipeline.py) where consecutive
    same-person appearances are captured a fraction of a second apart. When
    sample_frames > 0, consecutive appearances of the same person are instead
    seconds apart, so their OSNet similarity is naturally lower - reusing the
    dense-tracking threshold here causes the same real person to be split
    into multiple new identities (inflating footfall with no real increase
    in people). Re-validate with `validate_pipeline.py --sample-frames
    --sample-window-seconds` matching this run's sampling and pass the
    suggested threshold in explicitly."""
    if sample_frames < 0:
        raise ValueError("sample_frames cannot be negative")
    if sample_frames > 0 and sample_window_seconds <= 0:
        raise ValueError("sample_window_seconds must be positive")

    detection_model = load_detection_model()
    reid_model = OSNetReID()

    chroma = None
    if use_chroma:
        from services.chroma import ChromaDBClient

        chroma = ChromaDBClient()
        purged = chroma.purge_expired()
        if purged:
            print(f"Purged {purged} expired ChromaDB embeddings")

    mivolo_official = None
    if run_demographics:
        if demographics_backend == "mivolo-official":
            import mivolo_official
        else:
            from demographics import estimate_demographics

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_source}")

    start_frame = 0
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    sample_interval = 1.0
    output_fps = fps
    if sample_frames > 0:
        output_fps = sample_frames / sample_window_seconds
        sample_interval = fps / output_fps

    writer = None
    if write_video:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, output_fps, (width, height))

    identities = {}            # {identity_id: PersonIdentity}
    track_id_to_identity = {}  # {track_id: identity_id}
    person_id_to_identity = {} # {stable UUID: identity_id}
    identity_counter = 1000

    track_stats = {}     # track_id -> {"first", "last", "count"} (M1)
    detections_log = []  # (frame_idx, track_id, embedding, bbox) (M2)
    det_confidences = [] # every YOLO detection confidence seen
    people_per_frame = []
    heatmap_hex_size = 40
    heatmap_acc = HeatmapAccumulator(width, height, hex_size=heatmap_hex_size)

    frame_idx = -1
    last_source_frame = -1
    processed_frames = 0
    next_sample_frame = 0.0
    new_identity_count = 0
    reid_match_count = 0
    chroma_match_count = 0
    proc_fps_ema = None  # exponential moving average of per-frame processing FPS
    last_raw_frame = None
    start = time.time()

    while max_frames is None or processed_frames < max_frames:
        frame_start = time.time()
        if sample_frames > 0:
            frame_idx = int(round(next_sample_frame))
            if total_frames > 0 and frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            next_sample_frame += sample_interval
        else:
            frame_idx = processed_frames

        ret, frame = cap.read()
        if not ret:
            break
        last_source_frame = frame_idx
        last_raw_frame = frame

        results = detect_people(detection_model, frame)
        official_frame_results = None  # lazily computed, cached per-frame

        people_in_frame = 0
        claimed_this_frame = set()  # identities already matched to a track in this frame
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
            heatmap_acc.add_bbox((x1, y1, x2, y2))

            if track_id in track_id_to_identity:
                identity_id = track_id_to_identity[track_id]
                identities[identity_id].add_appearance(features, frame_idx)
            else:
                matched_id, similarity = match_identity(features, identities, exclude_ids=claimed_this_frame)
                if matched_id is not None and similarity > reid_threshold:
                    identity_id = matched_id
                    identities[identity_id].add_appearance(features, frame_idx)
                    track_id_to_identity[track_id] = identity_id
                    reid_match_count += 1
                    print(f"[frame {frame_idx}] Re-ID: track {track_id} -> identity {identity_id} (sim={similarity:.3f})")
                else:
                    chroma_match = None
                    if chroma is not None:
                        chroma_match = chroma.match_identity(
                            features, store_id, reid_threshold
                        )

                    if chroma_match is not None:
                        person_id = chroma_match["person_id"]
                        identity_id = person_id_to_identity.get(person_id)
                        if identity_id is None:
                            identity_id = identity_counter
                            identity_counter += 1
                            identities[identity_id] = PersonIdentity(
                                identity_id, first_seen=frame_idx, person_id=person_id
                            )
                            person_id_to_identity[person_id] = identity_id
                        identity = identities[identity_id]
                        identity.add_appearance(features, frame_idx)
                        identity.reid_embedding_ref = chroma_match["vector_id"]
                        chroma_match_count += 1
                        print(
                            f"[frame {frame_idx}] Chroma Re-ID: track {track_id} "
                            f"-> identity {identity_id} (sim={chroma_match['similarity']:.3f})"
                        )
                    else:
                        age_result = gender_result = None
                        if run_demographics:
                            if demographics_backend == "mivolo-official":
                                if official_frame_results is None:
                                    official_frame_results = (
                                        mivolo_official.estimate_demographics_for_frame(frame)
                                    )
                                match = mivolo_official.match_bbox(
                                    (x1, y1, x2, y2), official_frame_results
                                )
                                if match is None:
                                    continue
                                gender_result = {
                                    "gender": match["gender"],
                                    "confidence": match["confidence"],
                                }
                                age_result = {
                                    "age": match["age_group"],
                                    "age_years": match["age_years"],
                                    "confidence": match["confidence"],
                                }
                            else:
                                age_result, gender_result = estimate_demographics(person_img)
                            if not demographics_pass_threshold(gender_result, age_result):
                                print(
                                    f"[frame {frame_idx}] Skip new identity for track {track_id}: "
                                    f"demographics confidence "
                                    f"{float((gender_result or {}).get('confidence') or 0):.2f} "
                                    f"< {MIN_DEMOGRAPHICS_CONFIDENCE:.2f}"
                                )
                                continue

                        identity_id = identity_counter
                        identity_counter += 1
                        identity = PersonIdentity(identity_id, first_seen=frame_idx)
                        identity.add_appearance(features, frame_idx)
                        if gender_result is not None:
                            video_time = (start_frame + frame_idx) / fps
                            identity.update_gender(gender_result, video_time)
                            identity.update_age(age_result, video_time)
                        identities[identity_id] = identity
                        person_id_to_identity[identity.person_id] = identity_id
                        new_identity_count += 1
                        print(f"[frame {frame_idx}] New identity {identity_id} (track {track_id})")

                    if chroma is not None:
                        identity = identities[identity_id]
                        identity.reid_embedding_ref = chroma.upsert_identity(
                            identity.person_id,
                            identity.representative_embedding(),
                            store_id,
                            camera_id,
                        )
                    track_id_to_identity[track_id] = identity_id

            claimed_this_frame.add(identity_id)

            if run_demographics:
                video_time = (start_frame + frame_idx) / fps
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

        frame_proc_time = time.time() - frame_start
        instant_fps = 1.0 / frame_proc_time if frame_proc_time > 0 else 0.0
        proc_fps_ema = instant_fps if proc_fps_ema is None else (0.9 * proc_fps_ema + 0.1 * instant_fps)

        if writer is not None:
            frame = draw_boxes(frame, results, track_id_to_identity, identities)
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

        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"...processed {processed_frames} sampled frames (source frame {frame_idx})")

    cap.release()
    if writer is not None:
        writer.release()

    if chroma is not None:
        for identity in identities.values():
            representative = identity.representative_embedding()
            if representative is not None:
                identity.reid_embedding_ref = chroma.upsert_identity(
                    identity.person_id,
                    representative,
                    store_id,
                    camera_id,
                )

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
        "frames_processed": processed_frames,
        "last_source_frame": last_source_frame,
        "sampling": {
            "frames_per_window": sample_frames or None,
            "window_seconds": sample_window_seconds if sample_frames else None,
            "effective_fps": round(output_fps, 3),
        },
        "elapsed_seconds": round(elapsed, 1),
        "avg_processing_fps": round(processed_frames / max(elapsed, 1e-9), 2),
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
            "reid_threshold": reid_threshold,
            "same_person_pairs": len(same_id_sims),
            "diff_person_pairs": len(diff_id_sims),
            "same_person_sim_p5": same_p[5],
            "same_person_sim_median": same_p[50],
            "diff_person_sim_p95": diff_p[95],
            "unique_persistent_identities": len(identities),
            "new_identities": new_identity_count,
            "reid_merges": reid_match_count,
            "chroma_matches": chroma_match_count,
        },
        "demographics_m3": demographics_m3,
        "heatmap": {
            "hex_size": heatmap_acc.hex_size,
            "occupied_cells": len(heatmap_acc.counts),
            "total_visits": heatmap_acc.total_visits(),
            "max_cell_count": heatmap_acc.max_count(),
        },
    }

    base, _ = os.path.splitext(output_path)
    heatmap_image_path = base + "_heatmap.png"
    heatmap_data_path = base + "_heatmap.json"
    if heatmap_acc.total_visits() > 0 and last_raw_frame is not None:
        heatmap_frame = render_heatmap(last_raw_frame, heatmap_acc)
        cv2.imwrite(heatmap_image_path, heatmap_frame)
        heatmap_acc.save(heatmap_data_path)
        report["heatmap"]["image_path"] = heatmap_image_path
        report["heatmap"]["data_path"] = heatmap_data_path

    print(f"\n=== Run summary ===")
    print(
        f"Processed {processed_frames} sampled frames through source frame {last_source_frame} "
        f"in {elapsed:.1f}s ({report['avg_processing_fps']:.1f} FPS)"
    )
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
    if writer is not None:
        print(f"Annotated video written to {output_path}")

    os.makedirs(os.path.dirname(metrics_out_path), exist_ok=True)
    with open(metrics_out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Metrics report written to {metrics_out_path}")

    return {
        "report": report,
        "identities": identities,
        "fps": fps,
        "frames_processed": processed_frames,
        "source_duration_seconds": max(last_source_frame + 1, 0) / fps,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=VIDEO_SOURCE)
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__), "..", "results", "tracked_output.mp4"))
    parser.add_argument("--metrics-out", default=None,
                         help="Where to save the JSON metrics report (default: alongside --output, same basename)")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--sample-frames", type=int, default=0,
                         help="Process this many evenly spaced frames per sampling window (0 = every frame)")
    parser.add_argument("--sample-window-seconds", type=float, default=10,
                         help="Sampling window size in seconds (used with --sample-frames)")
    parser.add_argument("--reid-threshold", type=float, default=REID_THRESHOLD,
                         help="Cosine similarity threshold for re-identification. The config "
                              "default is tuned for dense, frame-by-frame tracking - when using "
                              "--sample-frames, re-validate with validate_pipeline.py using the "
                              "same sampling and pass the suggested value here.")
    parser.add_argument("--no-video", action="store_true",
                         help="Do not encode an annotated output video")
    parser.add_argument("--no-demographics", action="store_true",
                         help="Skip age/gender estimation (M3) - faster, M1/M2 tracking only")
    parser.add_argument("--demographics-backend", choices=["mivolo-body", "mivolo-official"],
                         default="mivolo-body",
                         help="mivolo-body: transformers mivolo_v2, body-only (default). "
                              "mivolo-official: official repo's own detector + fused face+body model, heavier.")
    parser.add_argument("--persist", action="store_true",
                         help="Write resulting identities + entry/exit events to Postgres (M4). "
                              "Requires the db service from docker-compose.yaml to be reachable.")
    parser.add_argument("--chroma", action="store_true",
                         help="Use ChromaDB for short-lived cross-run/cross-camera re-identification. "
                              "Enabled automatically by --persist.")
    parser.add_argument("--store-id", default="store_1")
    parser.add_argument("--camera-id", default="camera_1")
    args = parser.parse_args()

    output_path = os.path.normpath(args.output)
    if args.metrics_out:
        metrics_out_path = os.path.normpath(args.metrics_out)
    else:
        base, _ = os.path.splitext(output_path)
        metrics_out_path = base + "_metrics.json"

    result = run(args.input, output_path, args.max_frames, metrics_out_path,
                 run_demographics=not args.no_demographics,
                 demographics_backend=args.demographics_backend,
                 use_chroma=args.chroma or args.persist,
                 store_id=args.store_id,
                 camera_id=args.camera_id,
                 sample_frames=args.sample_frames,
                 sample_window_seconds=args.sample_window_seconds,
                 write_video=not args.no_video,
                 reid_threshold=args.reid_threshold)

    if args.persist:
        from datetime import datetime, timedelta, timezone
        from persist import persist_identities, persist_heatmap

        video_duration = result["source_duration_seconds"]
        count = persist_identities(
            result["identities"], args.store_id, args.camera_id, result["fps"],
            run_start=datetime.now(timezone.utc) - timedelta(seconds=video_duration),
        )
        print(
            f"Persisted {count} identities to Postgres and ChromaDB "
            f"(store={args.store_id}, camera={args.camera_id})"
        )

        heatmap_image_path = result["report"]["heatmap"].get("image_path")
        if heatmap_image_path:
            from cloudinary_upload import upload_heatmap

            try:
                heatmap_url = upload_heatmap(heatmap_image_path, args.store_id, args.camera_id)
                persist_heatmap(args.store_id, args.camera_id, heatmap_url)
                print(f"Heatmap uploaded and persisted: {heatmap_url}")
            except Exception as e:
                print(f"Warning: heatmap upload/persist failed ({e}); "
                      f"identities were still persisted above.")
