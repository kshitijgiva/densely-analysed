from ultralytics import YOLO
from config import TRACKER_CONFIG, MIN_CONFIDENCE, _select_device

def load_detection_model():
    """Load YOLO model with device optimization"""
    model = YOLO('yolov8n.pt').to(_select_device())
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
