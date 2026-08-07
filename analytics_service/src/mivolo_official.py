"""
Demographics via the official MiVOLO repo (https://github.com/WildChlamydia/MiVOLO)
instead of the transformers-hosted mivolo_v2 checkpoint used in demographics.py.

Runs MiVOLO's own person+face YOLO detector (yolov8x_person_face.pt) plus its
fused face+body age/gender model (model_imdb_cross_person_4.24_99.46.pth.tar),
which is the actual "fused face+body" setup the design doc describes, rather
than the body-only workaround in demographics.py.

Checkpoints (not included, download separately - see README links on the repo):
    weights/mivolo/yolov8x_person_face.pt
    weights/mivolo/model_imdb_cross_person_4.24_99.46.pth.tar

This runs a second, much heavier YOLO model (yolov8x vs. the yolov8n used for
tracking) per frame, so it's gated to only fire when a tracked identity
actually needs a demographics retry - see render_tracked_video.py.
"""
import os

from mivolo.model.mi_volo import MiVOLO
from mivolo.model.yolo_detector import Detector

from config import _select_device
from demographics import _age_to_group

WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights", "mivolo")
DETECTOR_WEIGHTS = os.path.join(WEIGHTS_DIR, "yolov8x_person_face.pt")
AGE_GENDER_CHECKPOINT = os.path.join(WEIGHTS_DIR, "model_imdb_cross_person_4.24_99.46.pth.tar")

_detector = None
_age_gender_model = None


def initialize():
    global _detector, _age_gender_model
    if _detector is not None:
        return

    if not os.path.isfile(DETECTOR_WEIGHTS) or not os.path.isfile(AGE_GENDER_CHECKPOINT):
        raise FileNotFoundError(
            f"Missing MiVOLO checkpoints in {WEIGHTS_DIR}. Expected "
            f"yolov8x_person_face.pt and model_imdb_cross_person_4.24_99.46.pth.tar "
            f"(see https://github.com/WildChlamydia/MiVOLO for download links)."
        )

    device = _select_device()
    half = device == "cuda"  # fp16 needs a CUDA GPU; CPU/MPS run fp32

    print("Loading official MiVOLO person+face detector (yolov8x)...")
    _detector = Detector(DETECTOR_WEIGHTS, device=device, half=half, verbose=False)
    print("Loading official MiVOLO fused face+body age/gender model...")
    _age_gender_model = MiVOLO(
        AGE_GENDER_CHECKPOINT, device=device, half=half,
        use_persons=True, disable_faces=False, verbose=False,
    )
    print("Official MiVOLO pipeline loaded")


def estimate_demographics_for_frame(frame):
    """Run MiVOLO's own detector + fused age/gender model on a full frame.

    Returns a list of {bbox: (x1,y1,x2,y2), age_group, age_years, gender, confidence}
    for each detected person (face results are merged into their associated
    person via face_to_person_map, so callers only need to look at persons).
    """
    initialize()

    detected = _detector.predict(frame)
    _age_gender_model.predict(frame, detected)

    results = []
    boxes = detected.yolo_results.boxes
    names = detected.yolo_results.names
    for ind in range(detected.n_objects):
        if names[int(boxes[ind].cls)] != "person":
            continue
        age = detected.ages[ind]
        gender = detected.genders[ind]
        gender_score = detected.gender_scores[ind]
        if age is None and gender is None:
            continue
        x1, y1, x2, y2 = detected.get_bbox_by_ind(ind).cpu().numpy().tolist()
        results.append({
            "bbox": (x1, y1, x2, y2),
            "age_group": _age_to_group(age) if age is not None else None,
            "age_years": round(age, 1) if age is not None else None,
            "gender": gender,
            "confidence": gender_score if gender_score is not None else 0.0,
        })
    return results


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def match_bbox(target_bbox, frame_results, iou_thresh=0.3):
    """Find the frame_results entry whose bbox best overlaps target_bbox."""
    best, best_iou = None, iou_thresh
    for r in frame_results:
        score = _iou(target_bbox, r["bbox"])
        if score > best_iou:
            best_iou, best = score, r
    return best
