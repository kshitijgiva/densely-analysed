import cv2
import time
from detection import load_detection_model, detect_people
from reid import OSNetReID
from demographics import estimate_demographics
from utils import draw_boxes
from config import VIDEO_SOURCE, REID_THRESHOLD, MIN_DEMOGRAPHICS_CONFIDENCE
from identity import (
    PersonIdentity,
    match_identity,
    needs_demographic_retry,
    demographics_pass_threshold,
)

# Initialize models
detection_model = load_detection_model()
reid_model = OSNetReID()
cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_delay = int(1000 / fps) if fps > 0 else 33

# State management
identities = {}             # {identity_id: PersonIdentity}
track_id_to_identity = {}   # {track_id: identity_id}
identity_counter = 1000     # Start from 1000 to distinguish from track IDs
loop_count = 0

print(f"Running real-time analytics at {fps:.1f} FPS")
print("Press 'q' to quit, 'r' to reset loop")

while True:
    start_time = time.time()
    current_time = time.time()
    
    ret, frame = cap.read()
    if not ret:
        loop_count += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        print(f"Restarting video (loop {loop_count})")
        continue

    results = detect_people(detection_model, frame)
    
    for box in results[0].boxes:
        if not box.id:
            continue
            
        track_id = int(box.id.item())
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        person_img = frame[y1:y2, x1:x2]
        
        # Extract re-ID features
        features = reid_model.extract_features(person_img)
        if features is None:
            continue  # Skip if feature extraction failed
        
        # Existing track: update identity
        if track_id in track_id_to_identity:
            identity_id = track_id_to_identity[track_id]
            identity = identities[identity_id]
            
            # Update appearance history
            identity.add_appearance(features, current_time)
            
            # Handle demographic retries
            needs_gender, needs_age = needs_demographic_retry(identity, current_time)

            if needs_gender or needs_age:
                age_demo, gender_demo = estimate_demographics(person_img)

                if needs_gender and identity.update_gender(gender_demo, current_time):
                    print(f"Gender update: ID {identity_id} - {gender_demo}")

                if needs_age and identity.update_age(age_demo, current_time):
                    print(f"Age update: ID {identity_id} - {age_demo}")

        # New track: find or create identity
        else:
            # Try to match existing identity
            matched_id, similarity = match_identity(features, identities)

            if matched_id and similarity > REID_THRESHOLD:  # Re-identification threshold
                print(f"Re-ID: Track {track_id} → Identity {matched_id} (sim={similarity:.2f})")
                identity = identities[matched_id]
                identity.add_appearance(features, current_time)
                track_id_to_identity[track_id] = matched_id
            else:
                # Initial demographics gate new IDs (and therefore footfall).
                age_demo, gender_demo = estimate_demographics(person_img)
                if not demographics_pass_threshold(gender_demo, age_demo):
                    print(
                        f"Skip new identity for track {track_id}: "
                        f"demographics confidence "
                        f"{float((gender_demo or {}).get('confidence') or 0):.2f} "
                        f"< {MIN_DEMOGRAPHICS_CONFIDENCE:.2f}"
                    )
                    continue

                identity_id = identity_counter
                identity_counter += 1
                identity = PersonIdentity(identity_id)

                identity.add_appearance(features, current_time)
                identity.update_gender(gender_demo, current_time)
                identity.update_age(age_demo, current_time)

                identities[identity_id] = identity
                track_id_to_identity[track_id] = identity_id
                print(f"New Identity {identity_id}: Gender={gender_demo}, Age={age_demo}")
    
    # Visualization
    frame = draw_boxes(frame, results, track_id_to_identity, identities)
    
    # Display loop count
    cv2.putText(frame, f"Loop: {loop_count}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Display FPS
    proc_time = (time.time() - start_time) * 1000
    fps_text = f"FPS: {1000/proc_time:.1f}" if proc_time > 0 else "FPS: ?"
    cv2.putText(frame, fps_text, (frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    cv2.imshow('CCTV Analytics - Real-time', frame)
    
    # Frame rate control
    proc_time = (time.time() - start_time) * 1000
    wait_time = max(1, int(frame_delay - proc_time))
    
    key = cv2.waitKey(wait_time) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        identities = {}
        track_id_to_identity = {}
        identity_counter = 1000
        loop_count = 0
        print("System reset")

cap.release()
cv2.destroyAllWindows()
print("Real-time processing ended")
