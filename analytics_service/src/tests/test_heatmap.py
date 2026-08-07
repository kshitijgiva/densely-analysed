import numpy as np
import pytest

from heatmap import (
    HeatmapAccumulator,
    axial_round,
    build_accumulator_from_detections,
    foot_point,
    hex_to_pixel,
    point_to_hex,
    render_heatmap,
)


def test_axial_round_preserves_exact_integer_coords():
    assert axial_round(3.0, -2.0) == (3, -2)


def test_hex_center_round_trips_through_point_to_hex():
    hex_size = 40
    for q in range(-3, 4):
        for r in range(-3, 4):
            cx, cy = hex_to_pixel(q, r, hex_size)
            assert point_to_hex(cx, cy, hex_size) == (q, r)


def test_nearby_points_land_in_same_cell_as_their_center():
    hex_size = 40
    cx, cy = hex_to_pixel(2, -1, hex_size)
    # A small nudge should stay within the same cell.
    assert point_to_hex(cx + 3, cy - 2, hex_size) == (2, -1)


def test_foot_point_is_bottom_center_of_bbox():
    assert foot_point((10, 20, 30, 60)) == (20.0, 60)


def test_accumulator_rejects_non_positive_hex_size():
    with pytest.raises(ValueError):
        HeatmapAccumulator(100, 100, hex_size=0)


def test_accumulator_counts_points_in_same_cell_together():
    acc = HeatmapAccumulator(200, 200, hex_size=40)
    cell_a = acc.add_point(100, 100)
    cell_b = acc.add_point(101, 99)  # should land in the same cell
    assert cell_a == cell_b
    assert acc.counts[cell_a] == 2
    assert acc.total_visits() == 2
    assert acc.max_count() == 2


def test_accumulator_add_bbox_uses_foot_point():
    acc = HeatmapAccumulator(200, 200, hex_size=40)
    cell_from_bbox = acc.add_bbox((90, 10, 110, 100))
    cell_from_point = point_to_hex(100, 100, 40)
    assert cell_from_bbox == cell_from_point


def test_to_dict_normalizes_density_relative_to_busiest_cell():
    acc = HeatmapAccumulator(200, 200, hex_size=40)
    hot_cell = (0, 0)
    cold_cell = (5, 5)
    acc.counts[hot_cell] = 10
    acc.counts[cold_cell] = 5

    data = acc.to_dict()
    assert data["max_count"] == 10
    assert data["total_visits"] == 15
    assert data["occupied_cells"] == 2

    by_cell = {(c["q"], c["r"]): c for c in data["cells"]}
    assert by_cell[hot_cell]["density"] == 1.0
    assert by_cell[cold_cell]["density"] == 0.5


def test_to_dict_on_empty_accumulator_has_zero_max_and_no_divide_by_zero():
    acc = HeatmapAccumulator(200, 200, hex_size=40)
    data = acc.to_dict()
    assert data["max_count"] == 0
    assert data["cells"] == []


def test_build_accumulator_from_detections_bins_every_bbox():
    detections_log = [
        (0, 1, None, (90, 10, 110, 100)),
        (1, 1, None, (91, 11, 111, 101)),
        (1, 2, None, (400, 400, 420, 480)),
    ]
    acc = build_accumulator_from_detections(detections_log, 640, 480, hex_size=40)
    assert acc.total_visits() == 3
    assert acc.max_count() == 2  # the two near-identical bboxes share a cell


def test_render_heatmap_is_noop_on_empty_accumulator():
    background = np.zeros((100, 100, 3), dtype=np.uint8)
    acc = HeatmapAccumulator(100, 100, hex_size=20)
    out = render_heatmap(background, acc)
    assert out.shape == background.shape
    assert out.dtype == background.dtype
    assert np.array_equal(out, background)


def test_render_heatmap_colors_the_hot_cell_and_leaves_background_shape_intact():
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    acc = HeatmapAccumulator(200, 200, hex_size=30)
    for _ in range(5):
        acc.add_point(100, 100)

    out = render_heatmap(background, acc, alpha=1.0)
    assert out.shape == background.shape
    assert out.dtype == background.dtype
    # Fully opaque overlay (alpha=1.0) at the hot cell's own center should no
    # longer be pure black.
    assert out[100, 100].sum() > 0
    # A far corner outside any hex cell should be untouched.
    assert out[0, 0].sum() == 0


def test_render_heatmap_respects_min_count_threshold():
    background = np.zeros((200, 200, 3), dtype=np.uint8)
    acc = HeatmapAccumulator(200, 200, hex_size=30)
    acc.add_point(100, 100)  # single visit

    out = render_heatmap(background, acc, alpha=1.0, min_count=2)
    assert np.array_equal(out, background)
