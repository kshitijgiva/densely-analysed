import time
import uuid

import numpy as np
from scipy.spatial.distance import cosine

from config import MIN_DEMOGRAPHICS_CONFIDENCE


class PersonIdentity:
    def __init__(self, identity_id, first_seen=None, person_id=None):
        self.id = identity_id
        self.person_id = person_id or str(uuid.uuid4())
        self.reid_embedding_ref = None
        self.gender = None
        self.gender_confidence = 0.0
        self.age = None
        self.age_confidence = 0.0
        self.last_gender_attempt = 0
        self.last_age_attempt = 0
        self.gender_attempts = 0
        self.age_attempts = 0
        self.appearances = []  # List of (features, timestamp)
        self.first_seen = first_seen if first_seen is not None else time.time()
        self.last_seen = self.first_seen

    def add_appearance(self, features, timestamp):
        """Add new appearance to history"""
        self.appearances.append((features, timestamp))
        self.last_seen = timestamp
        # Keep only last 10 appearances
        if len(self.appearances) > 10:
            self.appearances = self.appearances[-10:]

    def representative_embedding(self):
        """Return an L2-normalized centroid of the retained OSNet appearances."""
        features = [feature for feature, _ in self.appearances if feature is not None]
        if not features:
            return None
        centroid = np.mean(features, axis=0)
        norm = np.linalg.norm(centroid)
        return centroid / norm if norm else None

    def is_footfall_eligible(self):
        """Only high-confidence demographics count toward footfall/new visits."""
        return (
            self.gender is not None
            and self.gender_confidence >= MIN_DEMOGRAPHICS_CONFIDENCE
        )

    def update_gender(self, gender_data, timestamp):
        self.gender_attempts += 1
        self.last_gender_attempt = timestamp
        gender = gender_data.get("gender")
        confidence = float(gender_data.get("confidence") or 0.0)
        if gender is None or confidence < MIN_DEMOGRAPHICS_CONFIDENCE:
            return False
        if self.gender is not None and confidence < self.gender_confidence:
            return False
        self.gender = gender
        self.gender_confidence = confidence
        return True

    def update_age(self, age_data, timestamp):
        self.age_attempts += 1
        self.last_age_attempt = timestamp
        age = age_data.get("age")
        confidence = float(age_data.get("confidence") or 0.0)
        if age is None or confidence < MIN_DEMOGRAPHICS_CONFIDENCE:
            return False
        if self.age is not None and confidence < self.age_confidence:
            return False
        self.age = age
        self.age_confidence = confidence
        return True

def match_identity(query_features, identity_db, exclude_ids=None):
    """Match features against all identities in database.

    exclude_ids skips identities already claimed by another track visible in
    the current frame - two simultaneously-visible boxes can never be the
    same person, so letting them compete for the same identity is always a
    false merge, independent of the similarity threshold."""
    best_match = None
    best_similarity = -1

    if query_features is None:
        return None, -1

    for identity_id, identity in identity_db.items():
        if exclude_ids and identity_id in exclude_ids:
            continue
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
                            gender_thresh=None, age_thresh=None,
                            retry_interval=3, max_attempts=3):
    """Check if demographics need retry"""
    if gender_thresh is None:
        gender_thresh = MIN_DEMOGRAPHICS_CONFIDENCE
    if age_thresh is None:
        age_thresh = MIN_DEMOGRAPHICS_CONFIDENCE
    needs_gender = (
        identity.gender_attempts < max_attempts and
        (
            identity.gender is None or
            (identity.gender_confidence < gender_thresh and
             (current_time - identity.last_gender_attempt) >= retry_interval)
        )
    )

    needs_age = (
        identity.age_attempts < max_attempts and
        (
            identity.age is None or
            (identity.age_confidence < age_thresh and
             (current_time - identity.last_age_attempt) >= retry_interval)
        )
    )

    return needs_gender, needs_age


def demographics_pass_threshold(gender_demo, age_demo=None):
    """True when a demographics read is strong enough to mint a new identity."""
    if not gender_demo or gender_demo.get("gender") is None:
        return False
    return float(gender_demo.get("confidence") or 0.0) >= MIN_DEMOGRAPHICS_CONFIDENCE
