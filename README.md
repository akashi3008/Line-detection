# Lane Perception (OpenCV) — Edges, Lines, Steering Angle

A small, local-first toolkit that turns a road image into:
- edge map (Canny)
- lane line overlay (Hough + slope/intercept averaging)
- optional steering angle estimate (classic lane-following baseline)

**Why this exists:** the attached notebooks (`DetectEdgesDisplayLines.ipynb`, `future_mob.ipynb`) were Colab-specific and **incomplete** (code replaced with `...`). This repo rebuilds the missing pipeline into a reusable, testable, CI-friendly project.

## Quick start

```bash
pip install -r requirements.txt
python -m lane_perception.cli generate --output examples/synthetic_lane.png
python -m lane_perception.cli detect --image examples/synthetic_lane.png --output outputs/overlay.png --steering
pytest -q
```

CLI tips:
- Canny thresholds now default to **auto** (median-based). Override with `--canny-low/--canny-high` if needed.
- Save intermediates for debugging with `--save-edges outputs/edges.png --save-masked outputs/masked.png`.

## Notes (ruthless mentor edition)

- This is a **baseline**, not ADAS-grade lane detection. Real roads need temporal filtering, better segmentation, and camera calibration.
- What makes this portfolio-worthy is **engineering discipline**: packaging, CLI, deterministic synthetic test data, and tests.
