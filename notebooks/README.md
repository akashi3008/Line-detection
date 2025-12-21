# Notebooks (provenance only)

The original notebooks were Colab-oriented and include incomplete code (`...` placeholders).
They are kept here only to show the origin of the idea.

Use the package + CLI instead:

```bash
python -m lane_perception.cli generate --output examples/synthetic_lane.png
python -m lane_perception.cli detect --image examples/synthetic_lane.png --output outputs/overlay.png --steering
