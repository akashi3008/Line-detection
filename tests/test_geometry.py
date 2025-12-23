import numpy as np

from lane_perception.pipeline import (
    _auto_canny_thresholds,
    compute_steering_angle_deg,
    detect_lanes,
    generate_synthetic_lane,
)


def test_auto_canny_thresholds_produce_non_degenerate_range():
    gray = np.zeros((32, 32), dtype=np.uint8)
    low, high = _auto_canny_thresholds(gray)
    assert 0 <= low < high <= 255


def test_steering_angle_reflects_curve_direction():
    img_right = generate_synthetic_lane(width=320, height=240, curvature_px=24)
    res_right = detect_lanes(img_right, mask_mode="blue")
    angle_right = compute_steering_angle_deg(img_right, res_right.lane_lines)

    img_left = generate_synthetic_lane(width=320, height=240, curvature_px=-24)
    res_left = detect_lanes(img_left, mask_mode="blue")
    angle_left = compute_steering_angle_deg(img_left, res_left.lane_lines)

    assert angle_right is not None and angle_left is not None
    assert angle_left < 90.0 < angle_right
