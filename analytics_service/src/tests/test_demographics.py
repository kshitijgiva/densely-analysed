import numpy as np

from demographics import estimate_demographics


def test_none_crop_returns_matching_confidence_keys():
    age_result, gender_result = estimate_demographics(None)
    assert age_result == {"age": None, "confidence": 0.0}
    assert gender_result == {"gender": None, "confidence": 0.0}


def test_too_small_crop_returns_matching_confidence_keys():
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)  # below the 50x25 minimum
    age_result, gender_result = estimate_demographics(tiny)
    assert age_result == {"age": None, "confidence": 0.0}
    assert gender_result == {"gender": None, "confidence": 0.0}
    # Both dicts must carry "confidence" (not "age_confidence") - this is
    # what identity.PersonIdentity.update_age/update_gender read.
    assert "confidence" in age_result
    assert "confidence" in gender_result


def test_empty_crop_returns_matching_confidence_keys():
    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    age_result, gender_result = estimate_demographics(empty)
    assert age_result == {"age": None, "confidence": 0.0}
    assert gender_result == {"gender": None, "confidence": 0.0}


def test_guard_path_does_not_trigger_model_load():
    """The too-small/None guard must return before initialize_demographics_model()
    runs, so calling estimate_demographics with a bad crop never pays for
    loading MiVOLO - this test would hang/fail-to-download without that."""
    import demographics
    assert demographics._model is None
    estimate_demographics(None)
    assert demographics._model is None
