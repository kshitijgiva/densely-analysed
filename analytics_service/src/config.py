# Configuration settings
import os

VIDEO_SOURCE = "../../data/raw/sample_video.mp4"  # Path to CCTV footage
OUTPUT_CSV = "../../results/analytics_table.csv"  # Output file path
FRAME_SKIP = 5  # Process every 5th frame for performance
MIN_CONFIDENCE = 0.5  # Minimum detection confidence
TRACKER_CONFIG = "bytetrack.yaml"  # Ultralytics tracker config (ByteTrack)

# Re-identification (OSNet via torchreid)
REID_MODEL_NAME = "osnet_x1_0"
REID_MODEL_CHECKPOINT = "osnet_x1_0_market1501.pt"  # market1501-trained weights, auto-downloaded on first run
REID_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "weights")
REID_THRESHOLD = 0.94  # Cosine similarity threshold for re-identification, tuned via validate_pipeline.py
                       # (separates same-person p5=0.95 from different-person p95=0.76 on test footage)

AGE_GENDER_MODEL = "MiniVGG"  # Lightweight model for Mac
DEMOGRAPHICS_MODEL = "body"  # Options: 'face', 'body', 'hybrid'
VERTEX_PROJECT_ID = "visual-similarity-459311"
VERTEX_PROJECT_LOCATION = "us-central1"