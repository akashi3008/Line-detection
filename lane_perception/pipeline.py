"""
Classic lane perception baseline built on OpenCV.

Pipeline:
  BGR image -> HSV color mask (optional) -> Canny -> ROI -> HoughLinesP
  -> average left/right lines -> overlay + steering angle
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Vertex = Tuple[int, int]
Line = Tuple[int, int, int, int]


@dataclass(frozen=True)
class LaneDetectionResult:
    edges: np.ndarray
    masked_edges: np.ndarray
    line_image: np.ndarray
    overlay: np.ndarray
    lane_lines: List[Line]


def _to_gray(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


def _blur(gray: np.ndarray, k: int) -> np.ndarray:
    k = int(k)
    if k <= 0 or k % 2 == 0:
        raise ValueError("gaussian_kernel must be a positive odd integer")
    return cv2.GaussianBlur(gray, (k, k), 0)


def _canny(gray: np.ndarray, low: int, high: int) -> np.ndarray:
    return cv2.Canny(gray, int(low), int(high))


def hsv_color_mask(bgr: np.ndarray, mode: str) -> np.ndarray:
    """
    Return a masked BGR image based on a simple HSV threshold.

    mode:
      - "none"   : no masking
      - "blue"   : good for synthetic demo (blue-ish lane markings)
      - "white"  : simplistic white lane threshold
      - "yellow" : simplistic yellow lane threshold
    """
    mode = (mode or "none").lower()
    if mode == "none":
        return bgr

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    if mode == "blue":
        lower, upper = (np.array([75, 0, 50]), np.array([130, 255, 255]))
    elif mode == "white":
        lower, upper = (np.array([0, 0, 200]), np.array([179, 40, 255]))
    elif mode == "yellow":
        lower, upper = (np.array([15, 80, 80]), np.array([40, 255, 255]))
    else:
        raise ValueError(f"Unknown mask mode: {mode}. Choose from none|blue|white|yellow")

    mask = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(bgr, bgr, mask=mask)


def region_of_interest(img: np.ndarray, vertices: Sequence[Vertex]) -> np.ndarray:
    mask = np.zeros_like(img)
    color = (255,) * img.shape[2] if img.ndim == 3 else 255
    cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], color)
    return cv2.bitwise_and(img, mask)


def _auto_roi(height: int, width: int) -> Tuple[Vertex, ...]:
    # trapezoid — ok for synthetic sample and many dashcam-like frames
    return (
        (int(width * 0.05), height),
        (int(width * 0.45), int(height * 0.60)),
        (int(width * 0.55), int(height * 0.60)),
        (int(width * 0.95), height),
    )


def _hough_lines(
    edges: np.ndarray,
    rho: float,
    theta: float,
    threshold: int,
    min_line_length: int,
    max_line_gap: int,
):
    return cv2.HoughLinesP(
        edges,
        rho=rho,
        theta=theta,
        threshold=int(threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )


def _slope_intercept(line: Line) -> Optional[Tuple[float, float]]:
    x1, y1, x2, y2 = line
    if x2 == x1:
        return None
    slope = (y2 - y1) / (x2 - x1)
    intercept = y1 - slope * x1
    return slope, intercept


def _make_line(y1: int, y2: int, slope: float, intercept: float) -> Line:
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return x1, y1, x2, y2


def average_slope_intercept(
    image: np.ndarray,
    line_segments: Iterable[Line],
    slope_threshold: float = 0.5,
) -> List[Line]:
    left: List[Tuple[float, float]] = []
    right: List[Tuple[float, float]] = []

    for seg in line_segments:
        si = _slope_intercept(seg)
        if si is None:
            continue
        slope, intercept = si

        # reject near-horizontal segments
        if abs(slope) < slope_threshold:
            continue

        (left if slope < 0 else right).append((slope, intercept))

    h = image.shape[0]
    y1 = h
    y2 = int(h * 0.60)

    lanes: List[Line] = []
    for group in (left, right):
        if not group:
            continue
        slope = float(np.mean([g[0] for g in group]))
        intercept = float(np.mean([g[1] for g in group]))
        lanes.append(_make_line(y1, y2, slope, intercept))

    return lanes


def draw_lines(image: np.ndarray, lines: Sequence[Line], color=(0, 255, 0), thickness: int = 10) -> np.ndarray:
    line_img = np.zeros_like(image)
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), color, int(thickness))
    return line_img


def combine_overlay(base: np.ndarray, line_img: np.ndarray) -> np.ndarray:
    return cv2.addWeighted(base, 0.8, line_img, 1.0, 1.0)


def compute_steering_angle_deg(image: np.ndarray, lane_lines: Sequence[Line]) -> Optional[float]:
    """
    Compute steering angle in degrees.

    Convention:
      - 90° means go straight
      - <90° steer left
      - >90° steer right

    If one lane line found, angle from that line.
    If two found, angle from midpoint at top of the lines.
    """
    if not lane_lines:
        return None

    h, w = image.shape[:2]

    if len(lane_lines) == 1:
        x1, y1, x2, y2 = lane_lines[0]
        x_offset = x2 - x1
    else:
        left, right = lane_lines[0], lane_lines[1]
        _, _, lx2, _ = left
        _, _, rx2, _ = right
        x_offset = (lx2 + rx2) / 2 - (w / 2)

    y_offset = h * 0.60  # look-ahead distance
    angle_rad = np.arctan2(y_offset, x_offset if x_offset != 0 else 1e-6)
    angle_deg = float(angle_rad * 180.0 / np.pi)

    steering = 90.0 - angle_deg
    return float(np.clip(steering, 0.0, 180.0))


def detect_lanes(
    image: np.ndarray,
    mask_mode: str = "blue",
    roi_vertices: Optional[Sequence[Vertex]] = None,
    canny_low: int = 50,
    canny_high: int = 150,
    gaussian_kernel: int = 5,
    hough_rho: float = 2.0,
    hough_theta: float = np.pi / 180.0,
    hough_threshold: int = 50,
    min_line_length: int = 40,
    max_line_gap: int = 100,
    slope_threshold: float = 0.5,
) -> LaneDetectionResult:
    if image is None or image.size == 0:
        raise ValueError("Empty image provided")

    masked_bgr = hsv_color_mask(image, mask_mode)
    gray = _to_gray(masked_bgr)
    blurred = _blur(gray, gaussian_kernel)
    edges = _canny(blurred, canny_low, canny_high)

    if roi_vertices is None:
        roi_vertices = _auto_roi(edges.shape[0], edges.shape[1])

    masked_edges = region_of_interest(edges, roi_vertices)
    segments = _hough_lines(masked_edges, hough_rho, hough_theta, hough_threshold, min_line_length, max_line_gap)

    if segments is None:
        lanes: List[Line] = []
    else:
        lanes = average_slope_intercept(image, segments.reshape((-1, 4)), slope_threshold=slope_threshold)

    line_img = draw_lines(image, lanes)
    overlay = combine_overlay(image, line_img)

    return LaneDetectionResult(
        edges=edges,
        masked_edges=masked_edges,
        line_image=line_img,
        overlay=overlay,
        lane_lines=lanes,
    )


def generate_synthetic_lane(
    width: int = 640,
    height: int = 360,
    lane_color_bgr=(255, 0, 0),  # blue in BGR
    background_bgr=(20, 20, 20),
    thickness: int = 12,
    curvature_px: int = 0,
) -> np.ndarray:
    """
    Generate a deterministic synthetic lane image for demos/tests.

    curvature_px shifts top points left/right to simulate a gentle curve.
    """
    img = np.full((height, width, 3), background_bgr, dtype=np.uint8)
    bottom_y = height - 20
    top_y = int(height * 0.60)

    left_bottom = (int(width * 0.20), bottom_y)
    right_bottom = (int(width * 0.80), bottom_y)
    left_top = (int(width * 0.45) + int(curvature_px), top_y)
    right_top = (int(width * 0.55) + int(curvature_px), top_y)

    cv2.line(img, left_bottom, left_top, lane_color_bgr, int(thickness))
    cv2.line(img, right_bottom, right_top, lane_color_bgr, int(thickness))
    return img
