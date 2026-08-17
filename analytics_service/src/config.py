# Configuration settings
import os

import torch


def _select_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_ANALYTICS_SERVICE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

LOCAL_VIDEO_DIR = os.path.join(_REPO_ROOT, "data", "raw")  # Local footage the job API may read
VIDEO_SOURCE = os.path.join(LOCAL_VIDEO_DIR, "samplevideo.mp4")  # Path to CCTV footage
OUTPUT_CSV = os.path.join(_ANALYTICS_SERVICE_DIR, "results", "analytics_table.csv")  # Output file path
FRAME_SKIP = 5  # Process every 5th frame for performance
MIN_CONFIDENCE = 0.5  # Minimum detection confidence
TRACKER_CONFIG = "bytetrack.yaml"  # Ultralytics tracker config (ByteTrack)

# Detector weights - override via ANALYTICS_DETECTION_WEIGHTS to A/B a
# fine-tuned checkpoint against the stock COCO weights without code changes.
# NOTE: analytics_service/finetune_mannequin_filter.py fine-tunes YOLOv8n on
# hard-negative mannequin photos to reduce false "person" hits, but the
# checkpoint produced from the initial ~45-photo Commons dataset collapsed
# detection confidence to near-zero on real people too (see the fine-tune
# plan's evaluate step) - it is NOT safe to point this at yet. Re-run
# finetune_mannequin_filter.py with a much larger/more balanced dataset
# before trying this env var against production footage.
DETECTION_WEIGHTS = os.environ.get("ANALYTICS_DETECTION_WEIGHTS", "yolov8n.pt")

# Re-identification (OSNet via torchreid)
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_CHECKPOINT = "osnet_x1_0_market1501.pt"  # market1501-trained weights, auto-downloaded on first run
REID_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
REID_THRESHOLD = 0.94  # Cosine similarity threshold for re-identification, tuned via validate_pipeline.py
                       # (separates same-person p5=0.95 from different-person p95=0.76 on test footage)
                       # NOTE: tuned on dense, frame-by-frame footage (consecutive same-person
                       # detections a fraction of a second apart). The sparse-sampling path used
                       # for big videos (render_tracked_video.py/analytics_api.py with
                       # sample_frames > 0) sees consecutive appearances seconds apart, so real
                       # same-person similarity is naturally lower - reusing this default there
                       # fragments one person into several new identities and inflates footfall.
                       # Re-validate with `validate_pipeline.py --sample-frames --sample-window-seconds`
                       # matching your sampling and pass the result via --reid-threshold / the
                       # analysis job's reid_threshold field.

REID_THRESHOLD_SPARSE = 0.85  # Default for analytics_api.py's 3-frames/10s sampling.
                       # Measured via `validate_pipeline.py --sample-frames 3 --sample-window-seconds 10`
                       # on data/raw/samplevideo6.mp4: same-person sim p5=0.61, diff-person p95=0.75 -
                       # the two distributions overlap on this footage, so no single threshold fully
                       # separates them. 0.85 was chosen from a sweep (with match_identity's same-frame
                       # exclusion applied - see identity.py) as the point where residual same-frame
                       # false merges drop to single digits (9/37 tracks) while still resolving some
                       # fragmentation (37 tracks -> 33 identities); 0.88+ eliminates false merges
                       # entirely but stops merging fragmented tracks altogether (37 -> 37, no better
                       # than the dense-tuned 0.94). Re-validate on your own footage/sampling before
                       # trusting this number, and inspect results/reid_validation_log.csv for
                       # _FALSE_MERGE rows if footfall still looks wrong.

# Short-lived cross-process/cross-camera re-identification store.
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
CHROMADB_TENANT = os.environ.get("CHROMADB_TENANT", "default_tenant")
CHROMADB_DATABASE = os.environ.get("CHROMADB_DATABASE", "default_database")
CHROMADB_COLLECTION = os.environ.get("CHROMADB_COLLECTION", "person_reid")
CHROMADB_TTL_HOURS = int(os.environ.get("CHROMADB_TTL_HOURS", "48"))

AGE_GENDER_MODEL = "MiVOLO"  # iitolstykh/mivolo_v2 via transformers, see demographics.py
DEMOGRAPHICS_MODEL = "body"  # Options: 'face', 'body', 'hybrid' - only 'body' is implemented
# Reject weak MiVOLO reads: no new identity / no footfall entry below this.
# 0.80 balances recall vs precision on results/demographics_labels.csv (the
# labeled spot-check set): 0.97 only keeps 48% of real visitors (100%
# precision) vs 0.80's 86% recall at 83% precision. Was accidentally set to
# 0.97 in the commit that added tests/test_identity_confidence.py, which
# still asserts 0.80 - keep this in sync with that test.
MIN_DEMOGRAPHICS_CONFIDENCE = 0.80

# Static-object filtering: a resolved identity (post re-id, so ByteTrack
# fragments across sample-window gaps are already merged) that sits in
# roughly the same spot for a large chunk of the clip is almost certainly a
# mannequin or a printed/on-screen image of a person, not a live shopper -
# YOLO+ByteTrack+OSNet have no notion of "alive" and will happily detect,
# track, and re-id a completely static object. See static_objects.py.
# These are initial heuristic defaults, NOT yet validated against real
# footage - re-tune against actual store clips with known mannequins/posters
# the same way REID_THRESHOLD_SPARSE above was tuned via validate_pipeline.py,
# by inspecting the "filtering" section of the job's metrics report.
STATIC_OBJECT_MIN_SPAN_SECONDS = 45          # must persist across this much of the clip
STATIC_OBJECT_MAX_DISPLACEMENT_RATIO = 0.03  # foot-point bounding extent / frame diagonal
STATIC_OBJECT_MIN_APPEARANCES = 4            # need enough samples for a meaningful signal

VERTEX_PROJECT_ID = "visual-similarity-459311"
VERTEX_PROJECT_LOCATION = "us-central1"