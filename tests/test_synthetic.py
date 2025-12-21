from lane_perception.pipeline import detect_lanes, generate_synthetic_lane, compute_steering_angle_deg


def test_detects_two_lane_lines_on_synthetic():
    img = generate_synthetic_lane(width=320, height=240)
    res = detect_lanes(img, mask_mode="blue")
    assert len(res.lane_lines) == 2


def test_steering_is_near_straight_on_synthetic():
    img = generate_synthetic_lane(width=320, height=240, curvature_px=0)
    res = detect_lanes(img, mask_mode="blue")
    angle = compute_steering_angle_deg(img, res.lane_lines)
    assert angle is not None
    assert 70.0 <= angle <= 110.0
