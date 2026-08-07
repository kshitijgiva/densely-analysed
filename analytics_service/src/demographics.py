import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Global model and processor instances
_gender_processor = None
_gender_model = None
_age_processor = None
_age_model = None

def initialize_gender_model(device='cpu'):
    """Initialize the gender recognition model once"""
    global _gender_processor, _gender_model
    if _gender_processor is None or _gender_model is None:
        print("Loading gender recognition model...")
        _gender_processor = AutoImageProcessor.from_pretrained("NTQAI/pedestrian_gender_recognition")
        _gender_model = AutoModelForImageClassification.from_pretrained("NTQAI/pedestrian_gender_recognition")
        _gender_model.to(device)
        _gender_model.eval()
        print("Gender model loaded successfully")

def initialize_age_model(device='cpu'):
    """Initialize the age recognition model once"""
    global _age_processor, _age_model
    if _age_processor is None or _age_model is None:
        print("Loading age recognition model...")
        _age_processor = AutoImageProcessor.from_pretrained("NTQAI/pedestrian_age_recognition")
        _age_model = AutoModelForImageClassification.from_pretrained("NTQAI/pedestrian_age_recognition")
        _age_model.to(device)
        _age_model.eval()
        print("Age model loaded successfully")

def estimate_gender_demographics(track_id, full_body_img):
    """Estimate gender from full-body image"""
    global _gender_processor, _gender_model
    
    # Initialize model on first call
    if _gender_processor is None or _gender_model is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        initialize_gender_model(device)
    
    # Skip processing if image is invalid
    if full_body_img.size == 0 or full_body_img.shape[0] < 50 or full_body_img.shape[1] < 25:
        return {'gender': None, 'confidence': 0.0}
    
    try:
        # Convert OpenCV BGR to PIL RGB
        rgb_img = cv2.cvtColor(full_body_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Preprocess image
        inputs = _gender_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(_gender_model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = _gender_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction and confidence
        confidence, pred = torch.max(probs, dim=1)
        gender = _gender_model.config.id2label[pred.item()]
        
        return {
            'gender': gender,
            'confidence': confidence.item()
        }
    except Exception as e:
        print(f"Gender recognition error: {str(e)}")
        return {'gender': None, 'confidence': 0.0}

def estimate_age_demographics(track_id, full_body_img):
    """Estimate age from full-body image"""
    global _age_processor, _age_model
    
    # Initialize model on first call
    if _age_processor is None or _age_model is None:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        initialize_age_model(device)
    
    # Skip processing if image is invalid
    if full_body_img.size == 0 or full_body_img.shape[0] < 50 or full_body_img.shape[1] < 25:
        return {'age': None, 'confidence': 0.0}
    
    try:
        # Convert OpenCV BGR to PIL RGB
        rgb_img = cv2.cvtColor(full_body_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
        
        # Preprocess image
        inputs = _age_processor(images=pil_img, return_tensors="pt")
        inputs = {k: v.to(_age_model.device) for k, v in inputs.items()}
        
        # Run inference
        with torch.no_grad():
            outputs = _age_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
        
        # Get prediction and confidence
        confidence, pred = torch.max(probs, dim=1)
        age = _age_model.config.id2label[pred.item()]
        
        return {
            'age': age,
            'confidence': confidence.item()
        }
    except Exception as e:
        print(f"Age recognition error: {str(e)}")
        return {'age': None, 'confidence': 0.0}
    
def needs_demographic_retry(identity, current_time):
    needs_gender = (
        identity['gender'] is None or 
        (identity['gender_confidence'] < 0.85 and
         (current_time - identity['last_gender_attempt']) >= 3)
    )
    needs_age = (
        identity['age'] is None or 
        (identity['age_confidence'] < 0.7 and
         (current_time - identity['last_age_attempt']) >= 3)
    )
    return needs_gender, needs_age


# import os
# os.environ['TF_USE_LEGACY_KERAS'] = '1'
# from deepface import DeepFace

# def estimate_demographics(track_id, cropped_img):
#     """Estimate age and gender from cropped image"""
#     if np.all(cropped_img == 0):
#         return {'age': None, 'gender': None, 'confidence': 0.0}
    
#     try:
#         print("Estimating demographics...")
#         analysis = DeepFace.analyze(
#             cropped_img, 
#             actions=['age', 'gender'],
#             detector_backend='retinaface',
#             enforce_detection=False,
#             silent=True
#         )
#         print(f"Demographics analysis:success")
#         print(f"for track_id {track_id}: {analysis}")

#         # Handle both list and dict output
#         if isinstance(analysis, list):
#             result = analysis[0]
#         else:
#             result = analysis

#         age = result.get('age')
#         gender = result.get('dominant_gender')
#         # DeepFace may not always provide 'gender_confidence'
#         confidence = result.get('gender', {}).get(gender.capitalize(), 1.0) if isinstance(result.get('gender'), dict) else 1.0

#         return {
#             'age': age,
#             'gender': gender,
#             'confidence': float(confidence)
#         }
#     except Exception as e:
#         print(f"Demographics estimation failed: {e}")
#         return {'age': None, 'gender': None, 'confidence': 0.0}


