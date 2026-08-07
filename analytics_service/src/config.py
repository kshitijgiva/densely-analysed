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

VIDEO_SOURCE = os.path.join(_REPO_ROOT, "data", "raw", "sample_video.mp4")  # Path to CCTV footage
OUTPUT_CSV = os.path.join(_ANALYTICS_SERVICE_DIR, "results", "analytics_table.csv")  # Output file path
FRAME_SKIP = 5  # Process every 5th frame for performance
MIN_CONFIDENCE = 0.5  # Minimum detection confidence
TRACKER_CONFIG = "bytetrack.yaml"  # Ultralytics tracker config (ByteTrack)

# Re-identification (OSNet via torchreid)
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_CHECKPOINT = "osnet_x1_0_market1501.pt"  # market1501-trained weights, auto-downloaded on first run
REID_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
REID_THRESHOLD = 0.94  # Cosine similarity threshold for re-identification, tuned via validate_pipeline.py
                       # (separates same-person p5=0.95 from different-person p95=0.76 on test footage)

# Short-lived cross-process/cross-camera re-identification store.
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = int(os.environ.get("CHROMADB_PORT", "8000"))
CHROMADB_TENANT = os.environ.get("CHROMADB_TENANT", "default_tenant")
CHROMADB_DATABASE = os.environ.get("CHROMADB_DATABASE", "default_database")
CHROMADB_COLLECTION = os.environ.get("CHROMADB_COLLECTION", "person_reid")
CHROMADB_TTL_HOURS = int(os.environ.get("CHROMADB_TTL_HOURS", "48"))

AGE_GENDER_MODEL = "MiVOLO"  # iitolstykh/mivolo_v2 via transformers, see demographics.py
DEMOGRAPHICS_MODEL = "body"  # Options: 'face', 'body', 'hybrid' - only 'body' is implemented
VERTEX_PROJECT_ID = "visual-similarity-459311"
VERTEX_PROJECT_LOCATION = "us-central1"