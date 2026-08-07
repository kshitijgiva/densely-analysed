"""
M3 accuracy validation for demographics (age/gender) against a labeled sample.

Re-id (M2) can bootstrap its own ground truth (two boxes in the same frame are
definitely different people). Age/gender has no such free signal - there is no
substitute for a person looking at the crop and writing down what they actually
see. This script splits that into two steps:

    extract   Run the real M1+M2 pipeline over VIDEO_SOURCE, pick one
              representative crop per track (largest bbox = clearest view),
              run the production demographics model (demographics.py,
              body-only MiVOLO v2 - the same path realtime.py uses) on it, and
              write a labeling worksheet: one crop image per row plus the
              model's own prediction, with true_gender/true_age_group left
              blank for a human to fill in.

    evaluate  Read back the worksheet (only rows with true_* filled in) and
              report accuracy: gender accuracy overall and split by the
              confidence threshold identity.needs_demographic_retry already
              uses (0.85) - this checks whether that threshold actually
              separates trustworthy predictions from ones worth retrying -
              plus an age-group confusion matrix and adjacent-bucket-tolerant
              accuracy (MiVOLO's age head is a regression bucketed after the
              fact, so a true 34-year-old scored 35 is an off-by-one bucket
              edge case, not a real miss).

IMPORTANT: labels filled in by a script maintainer eyeballing crops are a
useful bootstrap for a single-store proof of concept, but they are one
person's visual judgment, not audited ground truth. Re-label (or spot check)
with a second reviewer before treating these numbers as a business-facing
accuracy claim.

Usage:
    python validate_demographics.py extract [--max-tracks N] [--min-crop-height PX]
    python validate_demographics.py evaluate
"""
import argparse
import csv
import inspect
import json
import os

import cv2

from config import VIDEO_SOURCE
from detection import load_detection_model, detect_people
from identity import needs_demographic_retry

LABELS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "demographics_labeling")
LABELS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "demographics_labels.csv")
REPORT_JSON = os.path.join(os.path.dirname(__file__), "..", "results", "demographics_validation_report.json")

# Read straight from identity.needs_demographic_retry's own default rather than
# duplicating the number here, so this stays correct if that default changes.
GENDER_CONFIDENCE_THRESHOLD = inspect.signature(needs_demographic_retry).parameters["gender_thresh"].default

FIELDNAMES = [
    "track_id", "frame", "image_path",
    "pred_gender", "pred_gender_confidence",
    "pred_age_group", "pred_age_years",
    "true_gender", "true_age_group", "notes",
]


def extract(video_source, max_tracks, min_crop_height):
    from demographics import estimate_demographics

    detection_model = load_detection_model()
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video source: {video_source}")

    best_crop = {}  # track_id -> (area, frame_idx, bbox, crop_image)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_people(detection_model, frame)
        for box in results[0].boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area == 0 or (y2 - y1) < min_crop_height:
                continue

            current = best_crop.get(track_id)
            if current is None or area > current[0]:
                best_crop[track_id] = (area, frame_idx, (x1, y1, x2, y2), frame[y1:y2, x1:x2].copy())

        frame_idx += 1

    cap.release()
    print(f"Scanned {frame_idx} frames, found {len(best_crop)} unique tracks.")

    track_ids = list(best_crop.keys())
    if max_tracks is not None:
        track_ids = track_ids[:max_tracks]

    os.makedirs(LABELS_DIR, exist_ok=True)
    rows = []
    for track_id in track_ids:
        _, frame_idx, bbox, crop = best_crop[track_id]
        image_path = os.path.join(LABELS_DIR, f"track_{track_id}.jpg")
        cv2.imwrite(image_path, crop)

        age_result, gender_result = estimate_demographics(crop)
        rows.append({
            "track_id": track_id,
            "frame": frame_idx,
            "image_path": os.path.relpath(image_path, os.path.dirname(LABELS_CSV)),
            "pred_gender": gender_result["gender"],
            "pred_gender_confidence": round(gender_result["confidence"], 3),
            "pred_age_group": age_result["age"],
            "pred_age_years": age_result.get("age_years"),
            "true_gender": "",
            "true_age_group": "",
            "notes": "",
        })
        print(f"track {track_id}: pred gender={gender_result['gender']} "
              f"({gender_result['confidence']:.2f}), age={age_result['age']}")

    os.makedirs(os.path.dirname(LABELS_CSV), exist_ok=True)
    with open(LABELS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} crops to {LABELS_DIR}")
    print(f"Labeling worksheet: {LABELS_CSV}")
    print("Fill in true_gender ('male'/'female') and true_age_group "
          "(e.g. '25-34') for each row, then run: python validate_demographics.py evaluate")


def _age_group_index(label, buckets):
    for i, b in enumerate(buckets):
        if b == label:
            return i
    return None


def evaluate():
    if not os.path.isfile(LABELS_CSV):
        raise FileNotFoundError(f"No labels found at {LABELS_CSV} - run 'extract' first.")

    with open(LABELS_CSV, newline="") as f:
        rows = list(csv.DictReader(f))

    total_rows = len(rows)
    labeled = [r for r in rows if r["true_gender"].strip() or r["true_age_group"].strip()]
    print(f"{len(labeled)}/{total_rows} rows have ground truth filled in "
          f"({total_rows - len(labeled)} still blank).")
    if not labeled:
        print("Nothing to evaluate yet - fill in true_gender/true_age_group in the CSV first.")
        return

    buckets = ["0-12", "13-17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]

    gender_rows = [r for r in labeled if r["true_gender"].strip()]
    gender_correct = sum(1 for r in gender_rows if r["pred_gender"] == r["true_gender"].strip())
    gender_accuracy = gender_correct / len(gender_rows) if gender_rows else None

    high_conf = [r for r in gender_rows if float(r["pred_gender_confidence"] or 0) >= GENDER_CONFIDENCE_THRESHOLD]
    low_conf = [r for r in gender_rows if float(r["pred_gender_confidence"] or 0) < GENDER_CONFIDENCE_THRESHOLD]
    high_conf_acc = (sum(1 for r in high_conf if r["pred_gender"] == r["true_gender"].strip()) / len(high_conf)
                      if high_conf else None)
    low_conf_acc = (sum(1 for r in low_conf if r["pred_gender"] == r["true_gender"].strip()) / len(low_conf)
                     if low_conf else None)

    confusion = {}
    for r in gender_rows:
        true_g, pred_g = r["true_gender"].strip(), r["pred_gender"]
        confusion.setdefault(true_g, {}).setdefault(pred_g, 0)
        confusion[true_g][pred_g] += 1

    age_rows = [r for r in labeled if r["true_age_group"].strip()]
    age_exact = sum(1 for r in age_rows if r["pred_age_group"] == r["true_age_group"].strip())
    age_exact_accuracy = age_exact / len(age_rows) if age_rows else None

    age_adjacent = 0
    for r in age_rows:
        true_idx = _age_group_index(r["true_age_group"].strip(), buckets)
        pred_idx = _age_group_index(r["pred_age_group"], buckets)
        if true_idx is not None and pred_idx is not None and abs(true_idx - pred_idx) <= 1:
            age_adjacent += 1
    age_adjacent_accuracy = age_adjacent / len(age_rows) if age_rows else None

    report = {
        "labeled_rows": len(labeled),
        "total_rows": total_rows,
        "gender": {
            "n": len(gender_rows),
            "accuracy": round(gender_accuracy, 3) if gender_accuracy is not None else None,
            "accuracy_at_or_above_confidence_threshold": {
                "threshold": GENDER_CONFIDENCE_THRESHOLD,
                "n": len(high_conf),
                "accuracy": round(high_conf_acc, 3) if high_conf_acc is not None else None,
            },
            "accuracy_below_confidence_threshold": {
                "n": len(low_conf),
                "accuracy": round(low_conf_acc, 3) if low_conf_acc is not None else None,
            },
            "confusion_matrix": confusion,
        },
        "age_group": {
            "n": len(age_rows),
            "exact_match_accuracy": round(age_exact_accuracy, 3) if age_exact_accuracy is not None else None,
            "adjacent_bucket_accuracy": round(age_adjacent_accuracy, 3) if age_adjacent_accuracy is not None else None,
        },
    }

    print("\n=== M3: demographics accuracy (labeled sample) ===")
    print(f"Gender accuracy: {report['gender']['accuracy']} (n={report['gender']['n']})")
    print(f"  >= {GENDER_CONFIDENCE_THRESHOLD} confidence: "
          f"{report['gender']['accuracy_at_or_above_confidence_threshold']['accuracy']} "
          f"(n={report['gender']['accuracy_at_or_above_confidence_threshold']['n']})")
    print(f"  <  {GENDER_CONFIDENCE_THRESHOLD} confidence: "
          f"{report['gender']['accuracy_below_confidence_threshold']['accuracy']} "
          f"(n={report['gender']['accuracy_below_confidence_threshold']['n']})")
    print(f"  confusion matrix (true -> predicted counts): {confusion}")
    print(f"Age-group exact-match accuracy: {report['age_group']['exact_match_accuracy']} "
          f"(n={report['age_group']['n']})")
    print(f"Age-group adjacent-bucket accuracy: {report['age_group']['adjacent_bucket_accuracy']}")

    if (high_conf_acc is not None and low_conf_acc is not None
            and high_conf_acc <= low_conf_acc):
        print(f"\nWARNING: high-confidence predictions were not more accurate than low-confidence "
              f"ones on this sample - the {GENDER_CONFIDENCE_THRESHOLD} retry threshold in "
              f"identity.needs_demographic_retry may not be a meaningful signal yet (small sample?).")

    os.makedirs(os.path.dirname(REPORT_JSON), exist_ok=True)
    with open(REPORT_JSON, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport written to {REPORT_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)

    p_extract = sub.add_parser("extract")
    p_extract.add_argument("--max-tracks", type=int, default=None)
    p_extract.add_argument("--min-crop-height", type=int, default=80,
                            help="Skip crops shorter than this (too small to judge reliably)")

    sub.add_parser("evaluate")

    args = parser.parse_args()
    if args.mode == "extract":
        extract(VIDEO_SOURCE, args.max_tracks, args.min_crop_height)
    else:
        evaluate()
