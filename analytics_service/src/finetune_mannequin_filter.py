"""
Fine-tune YOLOv8n so it stops firing "person" on mannequins / printed or
on-screen images of people, using hard-negative mining instead of manual
bounding-box annotation:

  - Mannequin-only photos: whatever the *current* detector calls "person"
    here is a false positive by construction -> written as an empty YOLO
    label (teaches the loss "no person here").
  - Real-people photos: the current detector's own person boxes are kept as
    class-0 labels (self-distillation), reinforcing correct behavior so
    training doesn't degenerate into suppressing all people. A small
    `EXCLUDED_BOXES` list drops the handful of boxes that landed on a wall
    poster/ad screen instead of a real person (verified by hand - see
    below) so those don't get mislabeled as positives.

Source images/videos live under data/finetune_mannequin/raw/ (see
analytics_service/README or the fine-tune plan for how they were sourced -
all CC-licensed Wikimedia Commons photos/videos, chosen because Commons is
directly downloadable without an API key, unlike Pexels/Pixabay which block
non-browser requests).

Usage:
    python finetune_mannequin_filter.py prepare-data
    python finetune_mannequin_filter.py train
    python finetune_mannequin_filter.py evaluate --weights yolov8n.pt --label baseline
    python finetune_mannequin_filter.py evaluate --weights weights/yolov8n_mannequin_ft.pt --label finetuned
"""
import argparse
import glob
import os
import random
import shutil

import cv2

from config import _select_device
from detection import load_detection_model

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data", "finetune_mannequin")
RAW_MANNEQUIN_DIR = os.path.join(DATA_ROOT, "raw", "mannequin_photos")
RAW_PEOPLE_DIR = os.path.join(DATA_ROOT, "raw", "people_photos")
EVAL_VIDEOS_DIR = os.path.join(DATA_ROOT, "raw", "eval_videos")
DATASET_DIR = os.path.join(DATA_ROOT, "dataset")
WEIGHTS_OUT = os.path.join(os.path.dirname(__file__), "weights", "yolov8n_mannequin_ft.pt")

# Boxes that landed on a wall poster/ad screen instead of a real person,
# found by rendering every detection and eyeballing it (see the fine-tune
# plan's "auto-label review" step) - excluded so they don't get mislabeled
# as positive "person" examples. Format: {filename: [(x1,y1,x2,y2), ...]}
# matched by >50% IoU against the detector's own boxes at prepare-data time.
EXCLUDED_BOXES = {
    "File_2024___free_download_photo_of_shopping_people_in_the_modern_interior_design.jpg": [
        (805, 315, 852, 413),  # backlit ad poster of a woman, top right of the mall interior
    ],
}

CONF_THRESHOLD = 0.4
VAL_RATIO = 0.2


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def _detect_boxes(model, img):
    results = model.predict(img, classes=0, conf=CONF_THRESHOLD, verbose=False)
    boxes = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(float, box.xyxy[0])
        boxes.append((x1, y1, x2, y2))
    return boxes


def _write_label(label_path, img_shape, boxes):
    h, w = img_shape[:2]
    lines = []
    for x1, y1, x2, y2 in boxes:
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        bw, bh = (x2 - x1) / w, (y2 - y1) / h
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    with open(label_path, "w") as f:
        f.write("\n".join(lines))


def prepare_data():
    model = load_detection_model()

    mannequin_paths = sorted(glob.glob(os.path.join(RAW_MANNEQUIN_DIR, "*")))
    people_paths = sorted(glob.glob(os.path.join(RAW_PEOPLE_DIR, "*")))

    examples = []  # (image_path, boxes) - boxes=[] means an explicit hard negative
    dropped_poster_boxes = 0

    for path in mannequin_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        # Every "person" box here is a false positive by construction -
        # discard them (empty label), don't keep them as anything.
        examples.append((path, []))

    for path in people_paths:
        img = cv2.imread(path)
        if img is None:
            continue
        boxes = _detect_boxes(model, img)
        excluded = EXCLUDED_BOXES.get(os.path.basename(path), [])
        kept = []
        for b in boxes:
            if any(_iou(b, ex) > 0.5 for ex in excluded):
                dropped_poster_boxes += 1
                continue
            kept.append(b)
        examples.append((path, kept))

    print(f"{len(mannequin_paths)} mannequin photos (-> empty labels), "
          f"{len(people_paths)} people photos, "
          f"{dropped_poster_boxes} poster/ad box(es) excluded from positives")

    random.Random(0).shuffle(examples)
    split = max(1, int(len(examples) * VAL_RATIO))
    val_examples, train_examples = examples[:split], examples[split:]

    if os.path.exists(DATASET_DIR):
        shutil.rmtree(DATASET_DIR)
    for split_name, split_examples in [("train", train_examples), ("val", val_examples)]:
        img_dir = os.path.join(DATASET_DIR, "images", split_name)
        lbl_dir = os.path.join(DATASET_DIR, "labels", split_name)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for path, boxes in split_examples:
            img = cv2.imread(path)
            stem = os.path.splitext(os.path.basename(path))[0]
            dest_img = os.path.join(img_dir, stem + ".jpg")
            cv2.imwrite(dest_img, img)
            _write_label(os.path.join(lbl_dir, stem + ".txt"), img.shape, boxes)

    data_yaml = os.path.join(DATASET_DIR, "data.yaml")
    with open(data_yaml, "w") as f:
        f.write(
            f"path: {os.path.abspath(DATASET_DIR)}\n"
            "train: images/train\n"
            "val: images/val\n"
            "names:\n"
            "  0: person\n"
        )
    print(f"Dataset written to {DATASET_DIR} "
          f"({len(train_examples)} train / {len(val_examples)} val images)")
    print(f"data.yaml: {data_yaml}")


def train(epochs=10, imgsz=640, lr0=0.0002, freeze=10, optimizer="SGD"):
    from ultralytics import YOLO

    data_yaml = os.path.join(DATASET_DIR, "data.yaml")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError("Run `prepare-data` first")

    model = YOLO("yolov8n.pt")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        optimizer=optimizer,  # "auto" silently ignores lr0 - pin it so lr0 actually applies
        lr0=lr0,              # tiny LR: 44 images should nudge the decision boundary, not retrain it
        freeze=freeze,        # freeze the backbone, only adapt detection head - less catastrophic forgetting
        patience=5,
        device=_select_device(),
        project=os.path.join(DATA_ROOT, "runs"),
        name="mannequin_ft",
        exist_ok=True,
    )
    best = os.path.join(results.save_dir, "weights", "best.pt")
    os.makedirs(os.path.dirname(WEIGHTS_OUT), exist_ok=True)
    shutil.copy(best, WEIGHTS_OUT)
    print(f"Fine-tuned weights saved to {WEIGHTS_OUT}")


def evaluate(weights, label, sample_fps=2, max_frames=60):
    from ultralytics import YOLO

    model = YOLO(weights).to(_select_device())
    print(f"\n=== {label} ({weights}) ===")
    for video_name in ["mannequin_eval.webm", "people_eval.webm"]:
        path = os.path.join(EVAL_VIDEOS_DIR, video_name)
        if not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, int(fps / sample_fps))
        frame_idx = 0
        sampled = 0
        detections = 0
        confs = []
        while sampled < max_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            results = model.predict(frame, classes=0, conf=CONF_THRESHOLD, verbose=False)
            boxes = results[0].boxes
            detections += len(boxes)
            confs.extend(float(b.conf.item()) for b in boxes)
            sampled += 1
            frame_idx += step
            if total and frame_idx >= total:
                break
        cap.release()
        avg_conf = sum(confs) / len(confs) if confs else 0.0
        print(f"  {video_name}: {sampled} frames sampled, "
              f"{detections} person detection(s), avg_conf={avg_conf:.2f}, "
              f"{detections / sampled:.2f} detections/frame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("prepare-data")

    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=10)
    p_train.add_argument("--imgsz", type=int, default=640)
    p_train.add_argument("--lr0", type=float, default=0.0002)
    p_train.add_argument("--freeze", type=int, default=10)
    p_train.add_argument("--optimizer", default="SGD")

    p_eval = sub.add_parser("evaluate")
    p_eval.add_argument("--weights", required=True)
    p_eval.add_argument("--label", required=True)
    p_eval.add_argument("--sample-fps", type=float, default=2)
    p_eval.add_argument("--max-frames", type=int, default=60)

    args = parser.parse_args()
    if args.command == "prepare-data":
        prepare_data()
    elif args.command == "train":
        train(epochs=args.epochs, imgsz=args.imgsz, lr0=args.lr0,
              freeze=args.freeze, optimizer=args.optimizer)
    elif args.command == "evaluate":
        evaluate(args.weights, args.label, sample_fps=args.sample_fps, max_frames=args.max_frames)
