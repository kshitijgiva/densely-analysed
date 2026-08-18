"""
Per-camera mirror/reflective-surface segmentation, used by reflection_filter.py
to flag tracked "people" that are actually reflections.

Wraps a vendored copy of ICCV2019_MirrorNet (Yang et al., "Where Is My
Mirror?", ICCV 2019 - github.com/Mhaiyang/ICCV2019_MirrorNet, see
vendor/mirrornet/), a single-image mirror segmentation CNN. Mirrors don't
move for a fixed CCTV camera, so this only needs to run MirrorNet on a
handful of sampled frames once per camera: get_or_build_mirror_mask()
averages those predictions into a stable ROI mask and caches it to disk.
MirrorNet is never run inside the main per-frame detection loop.

Disabled unless MIRROR_FILTER_ENABLED and MIRRORNET_WEIGHTS_PATH point at an
actual downloaded checkpoint (not vendored here - see config.py). When
unavailable, get_or_build_mirror_mask() returns (None, meta) rather than
raising, so a run degrades to "mirror filtering skipped" instead of failing.
"""
import os

import cv2
import numpy as np
import torch

from config import (
    MIRROR_MASK_AGG_THRESHOLD,
    MIRROR_MASK_CACHE_DIR,
    MIRROR_CALIBRATION_FRAMES,
    MIRRORNET_INPUT_SCALE,
    MIRRORNET_WEIGHTS_PATH,
    _select_device,
)

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_segmenter = None  # lazily built (net, device) singleton for this process


class MirrorSegmenterUnavailable(RuntimeError):
    """MirrorNet weights aren't present - callers should disable mirror
    filtering for this run rather than crash the analytics pipeline."""


def _load_segmenter():
    global _segmenter
    if _segmenter is not None:
        return _segmenter

    if not os.path.isfile(MIRRORNET_WEIGHTS_PATH):
        raise MirrorSegmenterUnavailable(
            f"MirrorNet checkpoint not found at {MIRRORNET_WEIGHTS_PATH} - download "
            "MirrorNet.pth from github.com/Mhaiyang/ICCV2019_MirrorNet and set "
            "ANALYTICS_MIRRORNET_WEIGHTS to its path, or leave ANALYTICS_MIRROR_FILTER unset/0."
        )

    from vendor.mirrornet.mirrornet import MirrorNet

    device = _select_device()
    # backbone_path=None is intentional: MirrorNet.pth already contains
    # trained ResNeXt101 weights, so there's no need for the separate
    # ImageNet-pretrained resnext_101_32x4d.pth the reference repo's
    # training script uses to initialize the backbone before fine-tuning.
    net = MirrorNet(backbone_path=None)
    state_dict = torch.load(MIRRORNET_WEIGHTS_PATH, map_location=device)
    net.load_state_dict(state_dict, strict=True)
    net.to(device)
    net.eval()
    _segmenter = (net, device)
    return _segmenter


def predict_mirror_probability(frame_bgr):
    """Return a float32 [0, 1] mirror-probability map, same H/W as frame_bgr."""
    net, device = _load_segmenter()
    h, w = frame_bgr.shape[:2]

    resized = cv2.resize(frame_bgr, (MIRRORNET_INPUT_SCALE, MIRRORNET_INPUT_SCALE),
                          interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    with torch.no_grad():
        _, _, _, f_1 = net(tensor)  # f_1: finest side output, already sigmoid'd (net.eval())

    prob = f_1.squeeze(0).squeeze(0).cpu().numpy()
    return cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)


def _cache_path(camera_id):
    os.makedirs(MIRROR_MASK_CACHE_DIR, exist_ok=True)
    safe_id = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(camera_id))
    return os.path.join(MIRROR_MASK_CACHE_DIR, f"{safe_id}.npy")


def get_or_build_mirror_mask(cap, camera_id, width, height, start_frame, total_frames,
                              num_calibration_frames=MIRROR_CALIBRATION_FRAMES,
                              agg_threshold=MIRROR_MASK_AGG_THRESHOLD):
    """Return (mask, meta) for this camera. mask is a boolean (height, width)
    array marking pixels judged to be inside a mirror/reflective surface, or
    None if MirrorNet is unavailable or no frames could be read.

    Reads frames directly from `cap` (seeking around), so callers must
    re-seek `cap` to their own starting position after calling this -
    render_tracked_video.run() does so unconditionally right after.
    """
    cache_file = _cache_path(camera_id)
    if os.path.isfile(cache_file):
        mask = np.load(cache_file)
        return mask, {"enabled": True, "source": "cache", "cache_file": cache_file}

    try:
        _load_segmenter()
    except MirrorSegmenterUnavailable as exc:
        print(f"[mirror] {exc}")
        return None, {"enabled": True, "source": "unavailable", "reason": str(exc)}

    span = total_frames if total_frames > 0 else num_calibration_frames
    sample_positions = np.linspace(
        start_frame, start_frame + max(span - 1, 0), num=num_calibration_frames, dtype=int
    )

    prob_sum = np.zeros((height, width), dtype=np.float64)
    sampled = 0
    for pos in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ret, frame = cap.read()
        if not ret:
            continue
        prob_sum += predict_mirror_probability(frame)
        sampled += 1

    if sampled == 0:
        return None, {"enabled": True, "source": "no_frames_read"}

    mean_prob = prob_sum / sampled
    mask = mean_prob >= agg_threshold
    np.save(cache_file, mask)
    meta = {
        "enabled": True,
        "source": "calibrated",
        "frames_sampled": sampled,
        "mirror_pixel_fraction": round(float(mask.mean()), 4),
        "cache_file": cache_file,
    }
    print(
        f"[mirror] Calibrated mirror ROI for camera '{camera_id}' from {sampled} frame(s) "
        f"({meta['mirror_pixel_fraction'] * 100:.1f}% of frame flagged) -> cached at {cache_file}"
    )
    return mask, meta
