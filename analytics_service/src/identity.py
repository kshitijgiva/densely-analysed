import time
from scipy.spatial.distance import cosine

class PersonIdentity:
    def __init__(self, identity_id):
        self.id = identity_id
        self.gender = None
        self.gender_confidence = 0.0
        self.age = None
        self.age_confidence = 0.0
        self.last_gender_attempt = 0
        self.last_age_attempt = 0
        self.appearances = []  # List of (features, timestamp)
        self.last_seen = time.time()
    
    def add_appearance(self, features, timestamp):
        """Add new appearance to history"""
        self.appearances.append((features, timestamp))
        self.last_seen = timestamp
        # Keep only last 10 appearances
        if len(self.appearances) > 10:
            self.appearances = self.appearances[-10:]
    
    def update_gender(self, gender_data, timestamp):
        self.gender = gender_data['gender']
        self.gender_confidence = gender_data['confidence']
        self.last_gender_attempt = timestamp
    
    def update_age(self, age_data, timestamp):
        self.age = age_data['age']
        self.age_confidence = age_data['confidence']
        self.last_age_attempt = timestamp

def match_identity(query_features, identity_db):
    """Match features against all identities in database"""
    best_match = None
    best_similarity = -1
    
    if query_features is None:
        return None, -1
    
    for identity_id, identity in identity_db.items():
        for features, _ in identity.appearances:
            if features is None:
                continue
                
            # Calculate cosine similarity
            similarity = 1 - cosine(query_features, features)
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = identity_id
    
    return best_match, best_similarity

def needs_demographic_retry(identity, current_time, 
                            gender_thresh=0.85, age_thresh=0.7,
                            retry_interval=3):
    """Check if demographics need retry"""
    needs_gender = (
        identity.gender is None or 
        (identity.gender_confidence < gender_thresh and
         (current_time - identity.last_gender_attempt) >= retry_interval)
    )
    
    needs_age = (
        identity.age is None or 
        (identity.age_confidence < age_thresh and
         (current_time - identity.last_age_attempt) >= retry_interval)
    )
    
    return needs_gender, needs_age
