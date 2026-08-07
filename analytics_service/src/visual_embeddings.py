"""
CLIP image/text encoder for semantic visual search over person crops.

This is a SEPARATE embedding space from OSNet (reid.py): OSNet is trained for
appearance re-identification (same person, different frame) and isn't aligned
with any text encoder, so it can't answer "find people wearing a red jacket".
CLIP's image and text encoders share one space, which is what makes that kind
of query possible - at the cost of a second model and a second, longer-lived
Chroma collection (see services/visual_search.py and the retention note
there). Not a replacement for OSNet re-id - a separate, optional capability.
"""
import base64
import io

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from config import _select_device

CLIP_CHECKPOINT = "openai/clip-vit-base-patch32"
THUMBNAIL_SIZE = (64, 96)  # (width, height) - kept small since it's stored as base64 in Chroma metadata

_model = None
_processor = None
_device = None


def _initialize():
    global _model, _processor, _device
    if _model is not None:
        return
    print("Loading CLIP model for visual search...")
    _device = torch.device(_select_device())
    _model = CLIPModel.from_pretrained(CLIP_CHECKPOINT).to(_device)
    _model.eval()
    _processor = CLIPProcessor.from_pretrained(CLIP_CHECKPOINT)
    print("CLIP model loaded successfully")


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def encode_image(person_img) -> np.ndarray:
    """person_img: BGR numpy array (a person crop). Returns an L2-normalized 512-dim vector."""
    import cv2

    _initialize()
    rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    inputs = _processor(images=Image.fromarray(rgb), return_tensors="pt").to(_device)
    with torch.no_grad():
        # this transformers version returns the full vision model output, with
        # .pooler_output overwritten to the projected (image-text-shared-space) embedding
        features = _model.get_image_features(**inputs).pooler_output
    return _normalize(features[0].cpu().numpy().astype(np.float32))


def encode_text(query: str) -> np.ndarray:
    """Returns an L2-normalized 512-dim vector in the same space as encode_image."""
    _initialize()
    inputs = _processor(text=[query], return_tensors="pt", padding=True).to(_device)
    with torch.no_grad():
        features = _model.get_text_features(**inputs).pooler_output
    return _normalize(features[0].cpu().numpy().astype(np.float32))


def make_thumbnail(person_img, quality=70) -> str:
    """Small base64 JPEG for displaying a search result - not the original crop."""
    import cv2

    rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
    thumb = Image.fromarray(rgb)
    thumb.thumbnail(THUMBNAIL_SIZE)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")
