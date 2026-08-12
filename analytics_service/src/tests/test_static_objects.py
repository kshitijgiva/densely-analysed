from static_objects import compute_static_identity_ids

FRAME_WIDTH, FRAME_HEIGHT, FPS = 640, 480, 25


def _detection(frame_idx, track_id, bbox):
    return (frame_idx, track_id, None, bbox)


def test_long_stationary_track_is_flagged_static():
    # Same bbox across 6 samples spanning 6000 frames at 25fps = 240s.
    detections_log = [
        _detection(i * 1200, 1, (300, 200, 340, 300)) for i in range(6)
    ]
    track_id_to_identity = {1: 10}

    static_ids = compute_static_identity_ids(
        track_id_to_identity, detections_log, FRAME_WIDTH, FRAME_HEIGHT, FPS
    )
    assert static_ids == {10}


def test_moving_track_is_not_flagged():
    # Same long span as the stationary test above, but the bbox walks
    # across the frame instead of staying put.
    detections_log = [
        _detection(i * 1200, 1, (i * 100, 200, i * 100 + 40, 300)) for i in range(6)
    ]
    track_id_to_identity = {1: 10}

    static_ids = compute_static_identity_ids(
        track_id_to_identity, detections_log, FRAME_WIDTH, FRAME_HEIGHT, FPS
    )
    assert static_ids == set()


def test_short_span_stationary_track_is_not_flagged():
    # Same bbox but only spans 8 frames (< 1s) - too short to judge.
    detections_log = [
        _detection(i, 1, (300, 200, 340, 300)) for i in range(8)
    ]
    track_id_to_identity = {1: 10}

    static_ids = compute_static_identity_ids(
        track_id_to_identity, detections_log, FRAME_WIDTH, FRAME_HEIGHT, FPS
    )
    assert static_ids == set()


def test_too_few_appearances_is_not_flagged():
    detections_log = [
        _detection(0, 1, (300, 200, 340, 300)),
        _detection(2000, 1, (300, 200, 340, 300)),
    ]
    track_id_to_identity = {1: 10}

    static_ids = compute_static_identity_ids(
        track_id_to_identity, detections_log, FRAME_WIDTH, FRAME_HEIGHT, FPS,
        min_appearances=4,
    )
    assert static_ids == set()


def test_fragmented_track_ids_resolving_to_same_identity_are_merged():
    # Simulates ByteTrack losing and reacquiring a static object across
    # sample-window gaps: two different raw track_ids, same identity, same
    # location - should still be judged as one static identity.
    detections_log = [
        _detection(0, 1, (300, 200, 340, 300)),
        _detection(1200, 1, (300, 200, 340, 300)),
        _detection(2400, 2, (300, 200, 340, 300)),
        _detection(3600, 2, (300, 200, 340, 300)),
    ]
    track_id_to_identity = {1: 10, 2: 10}

    static_ids = compute_static_identity_ids(
        track_id_to_identity, detections_log, FRAME_WIDTH, FRAME_HEIGHT, FPS
    )
    assert static_ids == {10}


def test_unresolved_track_id_is_ignored():
    detections_log = [_detection(0, 99, (300, 200, 340, 300))]
    static_ids = compute_static_identity_ids(
        {}, detections_log, FRAME_WIDTH, FRAME_HEIGHT, FPS
    )
    assert static_ids == set()
