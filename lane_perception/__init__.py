"""Lane perception utilities (edges → lines → steering angle)."""

from .pipeline import (
    LaneDetectionResult,
    detect_lanes,
    generate_synthetic_lane,
    compute_steering_angle_deg,
)

__all__ = [
    "LaneDetectionResult",
    "detect_lanes",
    "generate_synthetic_lane",
    "compute_steering_angle_deg",
]
