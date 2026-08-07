import cv2
import numpy as np

def draw_boxes(frame, results, track_id_map, identity_db):
    """
    Draw bounding boxes with identity information on the frame
    
    Args:
        frame: Input video frame
        results: Detection results from YOLO
        track_id_map: Dictionary mapping track_id to identity_id
        identity_db: Dictionary of PersonIdentity objects
        
    Returns:
        Frame with drawn boxes and information
    """
    for box in results[0].boxes:
        if not box.id:
            continue
            
        track_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get identity information
        identity_id = track_id_map.get(track_id)
        identity = identity_db.get(identity_id) if identity_id else None
        
        # Draw bounding box
        color = (0, 255, 0)  # Green for detected person
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare information text
        info_text = f"Track: {track_id}"
        if identity:
            # Display persistent identity information
            gender_text = f"{identity.gender}" if identity.gender else "Unknown"
            age_text = f"{identity.age}" if identity.age else "Unknown"
            info_text = f"ID {identity.id}: {gender_text}/{age_text}"
            
            # Add confidence indicators
            conf_text = f"G:{identity.gender_confidence:.1f} A:{identity.age_confidence:.1f}"
            cv2.putText(frame, conf_text, (x1, y2 + 35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw text background
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + 200, y1), (0, 0, 0), -1)
        
        # Draw identity text
        cv2.putText(frame, info_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame
