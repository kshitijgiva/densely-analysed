"""
Age/gender demographics via MiVOLO v2 (https://huggingface.co/iitolstykh/mivolo_v2),
run in body-only mode (config.DEMOGRAPHICS_MODEL == "body") so no separate face
detector/crop is required - matches the design doc's use of full-body crops from
the existing person tracker.

Loaded through `transformers` with trust_remote_code=True, which executes the
model repo's own Python files (configuration_mivolo.py, modeling_mivolo.py,
mivolo_image_processor.py) on first load. Reviewed on 2026-08-07: plain PyTorch
module/config/preprocessing code, no network or filesystem access beyond the
standard HF cache.
"""
import cv2
import torch
from transformers import AutoConfig, AutoImageProcessor, AutoModelForImageClassification

from config import _select_device

MIVOLO_CHECKPOINT = "iitolstykh/mivolo_v2"

# (upper bound inclusive, label) - matches the age_group examples in the design doc
_AGE_BUCKETS = [
    (12, "0-12"), (17, "13-17"), (24, "18-24"), (34, "25-34"),
    (44, "35-44"), (54, "45-54"), (64, "55-64"), (200, "65+"),
]

_model = None
_processor = None
_config = None
_device = None


def _age_to_group(age):
    for upper, label in _AGE_BUCKETS:
        if age <= upper:
            return label
    return _AGE_BUCKETS[-1][1]


def initialize_demographics_model():
    """Load the MiVOLO v2 model once (lazy, on first use)."""
    global _model, _processor, _config, _device
    if _model is not None:
        return

    print("Loading MiVOLO v2 demographics model...")
    _device = torch.device(_select_device())
    dtype = torch.float16 if _device.type == "cuda" else torch.float32

    _config = AutoConfig.from_pretrained(MIVOLO_CHECKPOINT, trust_remote_code=True)
    _model = AutoModelForImageClassification.from_pretrained(
        MIVOLO_CHECKPOINT, trust_remote_code=True, torch_dtype=dtype
    ).to(_device)
    _model.eval()
    _processor = AutoImageProcessor.from_pretrained(MIVOLO_CHECKPOINT, trust_remote_code=True)
    print("MiVOLO v2 model loaded successfully")


def estimate_demographics(person_img):
    """Run MiVOLO on a single full-body crop (BGR numpy array) and return age+gender.

    Body-only mode: no face crop is available from the upstream tracker, so the
    face branch is fed an all-zero tensor (MiVOLO's own preprocessing does this
    for any None entry) and the prediction relies on the body branch alone.
    """
    initialize_demographics_model()

    if person_img is None or person_img.size == 0 or person_img.shape[0] < 50 or person_img.shape[1] < 25:
        return {"age": None, "confidence": 0.0}, {"gender": None, "confidence": 0.0}

    try:
        faces_input = _processor(images=[None])["pixel_values"]
        body_input = _processor(images=[person_img])["pixel_values"]
        faces_input = faces_input.to(dtype=_model.dtype, device=_device)
        body_input = body_input.to(dtype=_model.dtype, device=_device)

        with torch.no_grad():
            output = _model(faces_input=faces_input, body_input=body_input)

        age = round(output.age_output[0].item(), 1)
        gender_prob = output.gender_probs[0].item()
        gender = _config.gender_id2label[output.gender_class_idx[0].item()]

        # MiVOLO's age head is a regression with no native per-sample confidence.
        # Reuse the (softmax) gender probability from the same forward pass as a
        # proxy for "how legible this crop was", so the existing retry logic in
        # identity.needs_demographic_retry (which expects a confidence for both
        # age and gender) keeps working without a separate uncertainty estimate.
        age_result = {"age": _age_to_group(age), "age_years": age, "confidence": gender_prob}
        gender_result = {"gender": gender, "confidence": gender_prob}
        return age_result, gender_result
    except Exception as e:
        print(f"Demographics estimation error: {str(e)}")
        return {"age": None, "confidence": 0.0}, {"gender": None, "confidence": 0.0}


def estimate_gender_demographics(track_id, full_body_img):
    """Kept for compatibility with existing call sites (realtime.py). Runs a
    full MiVOLO forward pass - see estimate_demographics for the combined call."""
    _, gender_result = estimate_demographics(full_body_img)
    return gender_result


def estimate_age_demographics(track_id, full_body_img):
    """Kept for compatibility with existing call sites (realtime.py). Runs a
    full MiVOLO forward pass - see estimate_demographics for the combined call."""
    age_result, _ = estimate_demographics(full_body_img)
    return age_result
