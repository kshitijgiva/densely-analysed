from validate_pipeline import suggest_threshold


def _repeat(values, times=1):
    return values * times


def test_clean_separation_returns_midpoint():
    same = _repeat([0.9, 0.92, 0.94, 0.96, 0.98], 5)   # 25 pairs, tight high similarity
    diff = _repeat([0.1, 0.15, 0.2, 0.25, 0.3], 5)      # 25 pairs, tight low similarity
    result = suggest_threshold(same, diff)
    assert result is not None
    assert 0.5 < result < 0.7


def test_overlapping_distributions_still_returns_clamped_midpoint():
    # Mirrors the real store1212354 job: same-person p5 (0.734) sits below
    # diff-person p95 (0.756) - no clean separation, but still a usable number.
    same = _repeat([0.73, 0.8, 0.85, 0.9, 0.95], 5)
    diff = _repeat([0.5, 0.6, 0.7, 0.75, 0.76], 5)
    result = suggest_threshold(same, diff)
    assert result is not None
    assert 0.5 <= result <= 0.95


def test_too_few_pairs_returns_none():
    same = [0.9, 0.92, 0.94]  # only 3 pairs, below default min_pairs=20
    diff = [0.2, 0.25, 0.3]
    assert suggest_threshold(same, diff) is None


def test_min_pairs_is_configurable():
    same = [0.9, 0.92, 0.94]
    diff = [0.2, 0.25, 0.3]
    assert suggest_threshold(same, diff, min_pairs=3) is not None


def test_clamp_caps_extreme_suggestions():
    same = _repeat([0.99], 25)
    diff = _repeat([0.98], 25)
    result = suggest_threshold(same, diff, clamp=(0.5, 0.95))
    assert result == 0.95

    same = _repeat([0.2], 25)
    diff = _repeat([0.1], 25)
    result = suggest_threshold(same, diff, clamp=(0.5, 0.95))
    assert result == 0.5
