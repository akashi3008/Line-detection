from pathlib import Path

import cv2

from .pipeline import compute_steering_angle_deg, detect_lanes, generate_synthetic_lane


def _cmd_generate(args: argparse.Namespace) -> int:
    img = generate_synthetic_lane(width=args.width, height=args.height, curvature_px=args.curvature)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), img)
    print(f"Wrote synthetic image to {args.output}")
    return 0


def _cmd_detect(args: argparse.Namespace) -> int:
    img = cv2.imread(str(args.image))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")

    res = detect_lanes(
        img,
        mask_mode=args.mask,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        auto_canny_sigma=args.auto_canny_sigma,
        gaussian_kernel=args.gaussian_kernel,
        hough_threshold=args.hough_threshold,
        min_line_length=args.min_line_length,
        max_line_gap=args.max_line_gap,
        slope_threshold=args.slope_threshold,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), res.overlay)
    print(f"Wrote overlay to {args.output}")

    if args.save_edges:
        args.save_edges.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_edges), res.edges)
        print(f"Wrote edges to {args.save_edges}")

    if args.save_masked:
        args.save_masked.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save_masked), res.masked_edges)
        print(f"Wrote masked edges to {args.save_masked}")

    if args.steering:
        angle = compute_steering_angle_deg(img, res.lane_lines)
        print("Steering angle:", "N/A" if angle is None else f"{angle:.2f}°")

    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Lane perception baseline (OpenCV)")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate a synthetic lane image")
    g.add_argument("--output", type=Path, required=True)
    g.add_argument("--width", type=int, default=640)
    g.add_argument("--height", type=int, default=360)
    g.add_argument("--curvature", type=int, default=0, help="pixels to shift top points (curve)")
    g.set_defaults(func=_cmd_generate)

    d = sub.add_parser("detect", help="Detect lane lines on an image")
    d.add_argument("--image", type=Path, required=True)
    d.add_argument("--output", type=Path, default=Path("outputs/overlay.png"))
    d.add_argument("--mask", choices=["none", "blue", "white", "yellow"], default="blue")
    d.add_argument("--canny-low", type=int, default=None, help="Lower Canny threshold (auto if omitted)")
    d.add_argument("--canny-high", type=int, default=None, help="Upper Canny threshold (auto if omitted)")
    d.add_argument("--auto-canny-sigma", type=float, default=0.33, help="Sigma used for auto Canny thresholds")
    d.add_argument("--gaussian-kernel", type=int, default=5)
    d.add_argument("--hough-threshold", type=int, default=15)
    d.add_argument("--min-line-length", type=int, default=20)
    d.add_argument("--max-line-gap", type=int, default=30)
    d.add_argument("--slope-threshold", type=float, default=0.4)
    d.add_argument("--save-edges", type=Path, help="Optional path to save raw Canny edges")
    d.add_argument("--save-masked", type=Path, help="Optional path to save ROI-masked edges")
    d.add_argument("--steering", action="store_true", help="Print steering angle estimate")
    d.set_defaults(func=_cmd_detect)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
