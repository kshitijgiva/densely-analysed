from ultralytics import YOLO
import torch
from config import TRACKER_CONFIG, MIN_CONFIDENCE

def load_detection_model():
    """Load YOLO model with device optimization"""
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO('yolov8n.pt').to(device)
    return model

def detect_people(model, frame):
    """Detect and track people in frame using ByteTrack for ID continuity"""
    results = model.track(
        frame,
        persist=True,
        classes=0,  # Class 0 = person
        verbose=False,
        conf=MIN_CONFIDENCE,
        tracker=TRACKER_CONFIG,
    )
    return results
