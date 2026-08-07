from identity import PersonIdentity, demographics_pass_threshold
from config import MIN_DEMOGRAPHICS_CONFIDENCE


def test_demographics_pass_threshold():
    assert demographics_pass_threshold({"gender": "female", "confidence": 0.81})
    assert not demographics_pass_threshold({"gender": "female", "confidence": 0.79})
    assert not demographics_pass_threshold({"gender": None, "confidence": 0.99})
    assert not demographics_pass_threshold(None)


def test_update_gender_rejects_below_threshold():
    identity = PersonIdentity(1)
    assert not identity.update_gender({"gender": "female", "confidence": 0.5}, 1.0)
    assert identity.gender is None
    assert identity.is_footfall_eligible() is False


def test_update_gender_accepts_and_blocks_downgrade():
    identity = PersonIdentity(1)
    assert identity.update_gender({"gender": "female", "confidence": 0.9}, 1.0)
    assert identity.is_footfall_eligible()
    assert not identity.update_gender({"gender": "male", "confidence": 0.85}, 2.0)
    assert identity.gender == "female"
    assert identity.gender_confidence == 0.9


def test_min_threshold_constant():
    assert MIN_DEMOGRAPHICS_CONFIDENCE == 0.80
