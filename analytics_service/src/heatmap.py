"""
Hexagonal-bin foot-traffic heatmap.

Bins each person's floor position (bottom-center of their detection bbox -
the point where they're actually standing, not the bbox centroid) into a
flat-top hex grid using axial coordinates, then renders the per-cell density
as a semi-transparent color overlay on top of a video frame. Hex bins read
better than square bins for crowd density because they have no preferred
axis and every cell has 6 equidistant neighbors, so adjacent high-traffic
cells blend smoothly instead of showing grid artifacts.

Usage as a library (see render_tracked_video.py for the integration):
    acc = HeatmapAccumulator(frame_width, frame_height, hex_size=40)
    acc.add_point(x, y)               # once per detection, per frame
    overlay = render_heatmap(frame, acc)
    acc.to_dict()                     # for saving alongside the metrics JSON
"""
import json
import math

import cv2
import numpy as np

# Flat-top axial hex math: https://www.redblobgames.com/grids/hexagons/
_SQRT3 = math.sqrt(3)


def pixel_to_axial(x, y, hex_size):
    """Pixel coords -> fractional axial (q, r) for a flat-top hex grid of
    the given circumradius `hex_size`."""
    q = (2.0 / 3.0 * x) / hex_size
    r = (-1.0 / 3.0 * x + _SQRT3 / 3.0 * y) / hex_size
    return q, r


def axial_round(q, r):
    """Round fractional axial coords to the nearest hex cell (cube-round)."""
    x, z = q, r
    y = -x - z
    rx, ry, rz = round(x), round(y), round(z)

    dx, dy, dz = abs(rx - x), abs(ry - y), abs(rz - z)
    if dx > dy and dx > dz:
        rx = -ry - rz
    elif dy > dz:
        ry = -rx - rz
    else:
        rz = -rx - ry
    return int(rx), int(rz)


def point_to_hex(x, y, hex_size):
    """Pixel coords -> the (q, r) axial cell that contains them."""
    return axial_round(*pixel_to_axial(x, y, hex_size))


def hex_to_pixel(q, r, hex_size):
    """Axial (q, r) -> the pixel coords of that cell's center."""
    x = hex_size * (3.0 / 2.0 * q)
    y = hex_size * (_SQRT3 / 2.0 * q + _SQRT3 * r)
    return x, y


def hex_corners(cx, cy, hex_size):
    """The 6 corner points of a flat-top hex centered at (cx, cy)."""
    return [
        (
            cx + hex_size * math.cos(math.radians(60 * i)),
            cy + hex_size * math.sin(math.radians(60 * i)),
        )
        for i in range(6)
    ]


def foot_point(bbox):
    """A person's floor position: bottom-center of their bbox, i.e. where
    their feet are - the right point to bin spatially, unlike the bbox
    centroid which drifts upward as a person gets closer to the camera."""
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, y2


class HeatmapAccumulator:
    """Accumulates per-hex-cell visit counts over a sequence of frames."""

    def __init__(self, frame_width, frame_height, hex_size=40):
        if hex_size <= 0:
            raise ValueError("hex_size must be positive")
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.hex_size = hex_size
        self.counts = {}  # {(q, r): int}

    def add_point(self, x, y):
        """Record one visit at pixel coords (x, y)."""
        cell = point_to_hex(x, y, self.hex_size)
        self.counts[cell] = self.counts.get(cell, 0) + 1
        return cell

    def add_bbox(self, bbox):
        """Record one visit from a detection bbox (x1, y1, x2, y2), using
        the bbox's foot point rather than its centroid."""
        x, y = foot_point(bbox)
        return self.add_point(x, y)

    def total_visits(self):
        return sum(self.counts.values())

    def max_count(self):
        return max(self.counts.values()) if self.counts else 0

    def to_dict(self):
        """JSON-serializable summary: cell centers, counts, and normalized
        density, for a dashboard/external renderer to consume."""
        max_count = self.max_count()
        cells = []
        for (q, r), count in sorted(self.counts.items(), key=lambda kv: -kv[1]):
            cx, cy = hex_to_pixel(q, r, self.hex_size)
            cells.append({
                "q": q, "r": r,
                "center_x": round(cx, 1), "center_y": round(cy, 1),
                "count": count,
                "density": round(count / max_count, 4) if max_count else 0.0,
            })
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "hex_size": self.hex_size,
            "total_visits": self.total_visits(),
            "occupied_cells": len(self.counts),
            "max_count": max_count,
            "cells": cells,
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def render_heatmap(background, accumulator, alpha=0.55, colormap=cv2.COLORMAP_JET,
                    min_count=1):
    """Draw a hex-bin heatmap of `accumulator` as a color overlay on top of
    `background` (e.g. the last processed frame, or an all-black canvas of
    the same size). Cells below `min_count` visits are left untouched so
    single-frame noise doesn't paint the whole grid.

    Returns a new image the same shape/dtype as `background`.
    """
    out = background.copy()
    max_count = accumulator.max_count()
    if max_count == 0:
        return out

    overlay = background.copy()
    drawn = False
    for (q, r), count in accumulator.counts.items():
        if count < min_count:
            continue
        cx, cy = hex_to_pixel(q, r, accumulator.hex_size)
        if cx < -accumulator.hex_size or cx > accumulator.frame_width + accumulator.hex_size:
            continue
        if cy < -accumulator.hex_size or cy > accumulator.frame_height + accumulator.hex_size:
            continue

        density = count / max_count  # 0..1
        # Map density -> a BGR color via the requested OpenCV colormap.
        color_lut_idx = np.uint8([[int(density * 255)]])
        color = cv2.applyColorMap(color_lut_idx, colormap)[0, 0].tolist()

        corners = np.array([hex_corners(cx, cy, accumulator.hex_size)], dtype=np.int32)
        cv2.fillPoly(overlay, corners, color)
        drawn = True

    if not drawn:
        return out

    cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0, dst=out)
    return out


def build_accumulator_from_detections(detections_log, frame_width, frame_height, hex_size=40):
    """Convenience for render_tracked_video.py's `detections_log` list of
    (frame_idx, track_id, embedding, bbox) tuples - bins every detection's
    bbox without needing to touch the main per-frame loop."""
    acc = HeatmapAccumulator(frame_width, frame_height, hex_size=hex_size)
    for entry in detections_log:
        bbox = entry[3]
        acc.add_bbox(bbox)
    return acc
