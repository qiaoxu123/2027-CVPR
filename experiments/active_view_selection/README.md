# P3.5 causal active view selection

No training. UAV XYZ/yaw and camera yaw/tilt are separate. `planner.py` receives
only frozen public Geometry logits/grid/bounds, public object geometry, current
pose, and causal previously observed ROI coverage. It cannot read point clouds,
image files, simulator render results for candidate views, or target labels.

`run.py` separates the simulator and privileged offline evaluator. The simulator
calls the unchanged P3 measured-point renderer; original RGB, axial depth,
visibility, original point winners and weak instance masks remain local in runs.
Policies select a view before receiving that exposure, even when the simulator
can reuse an identical view generated during the offline Oracle scan. No Oracle
frame is added to normal history until independently selected and reached.

The finite-library Oracle is exact only over its evaluated legal trace catalogue,
not a global upper bound on all continuous two-step actions. `oracle_extras.py`
adds explicitly privileged GT-directed camera/approach witnesses to separate
belief mistakes from missing source detail. None of these views/actions is
allowed to enter a normal policy. Both Oracle traces and normal traces have
at most two exposures and obey the same per-step public action constraints.

From the repo root, existing remoteclip venv, set `OMP_NUM_THREADS=1` and
`MKL_NUM_THREADS=1`:

```bash
python -m pytest -q tests/test_active_view_selection.py tests/test_multiview_rgb_grounding.py
python -m experiments.active_view_selection.run
python -m experiments.active_view_selection.oracle_extras
python -m experiments.active_view_selection.verify
python -m experiments.active_view_selection.geometry_check
python -m pytest -q
python -m experiments.active_view_selection.deliver
```

Output: `runs/active_view_selection/` (ignored). A completed run is never
re-executed or overwritten. Source DATA, old runs and Checkpoints stay unchanged.
The static middle-prefix cohort is map-stratified, target-independent, and uses
public bounds/AGL/building-risk legality. Results apply to this legal-start
cohort, not invalid poses excluded from it. Public flat ground/building proxies
cannot certify real-world collision-free flight over unannotated structures.

Stage A must pass on internal Dev before frozen model inference is considered.
No Seen/Unseen, Goal residual, Gain or controller changes. Fewer than50 independent
credible-visible static-error families does not establish Stage B feasibility.
