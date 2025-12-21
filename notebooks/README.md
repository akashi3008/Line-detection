# Notebooks (provenance only)

The original notebooks were Colab-oriented and include incomplete code (`...` placeholders).
They are kept here only to show the origin of the idea.

Use the package + CLI instead:

```bash
python -m lane_perception.cli generate --output examples/synthetic_lane.png
python -m lane_perception.cli detect --image examples/synthetic_lane.png --output outputs/overlay.png --steering


---

## `notebooks/DetectEdgesDisplayLines_original.ipynb`
Put your uploaded notebook here unchanged.

## `notebooks/future_mob_original.ipynb`
Put your uploaded notebook here unchanged.

---

# How to create this repo locally (copy/paste commands)

```bash
mkdir -p lane-perception-opencv/{lane_perception,examples,outputs,tests,notebooks}

# Then create files with the contents above
# Add your notebooks into notebooks/ with the names mentioned

cd lane-perception-opencv
pip install -r requirements.txt
python -m lane_perception.cli generate --output examples/synthetic_lane.png
python -m lane_perception.cli detect --image examples/synthetic_lane.png --output outputs/overlay.png --steering
pytest -q
