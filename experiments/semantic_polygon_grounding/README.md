# P4 audit and identifiable name-set Oracle

This stage stops before RGB binding/model development when the required
independently verifiable ambiguous reference-instance labels do not exist.
Do not substitute target-nearest polygons, synthetic colors or a forced choice
among parts of one named road for missing reference identity truth.

Only the existing named `cityrefer/landmarks.json` catalogue is whitelisted as
public inference geometry. The other `objects.json` polygons are audited
offline, not assumed publicly available. Description annotations and target
keys select offline audit records; candidate resolution sees public names and
public polygons only. Weak processed reference-name annotations are additional
dataset inputs already used by frozen A3, not a deployed instruction extractor.

Run from the repository root with the existing scientific Python environment:

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m experiments.semantic_polygon_grounding.audit
python -m experiments.semantic_polygon_grounding.verify
python -m experiments.semantic_polygon_grounding.eligibility
python -m pytest -q
python -m experiments.semantic_polygon_grounding.deliver
```

`audit` refuses to overwrite its run directory. The maximum observed cutoff
per family is selected using public timestamps before inspecting target labels.
All-prefix outputs stay separate. The original map/family splits, 784 grid
candidates, 20m XY localization radius and caches are unchanged. No training,
GPU encoding, new observation, Seen tuning, closed-loop or Unseen evaluation.

`A1_known_name_set` is the exact identity intervention after verifying the
existing A3 reference contours/aliases against the public catalogue. It is
**not** a correct homonymous-instance Oracle. That stronger Oracle is null when
there are no independently labeled reference instances. Eligibility-only
ceilings assume every error in a flagged subset is recoverable without damage;
they are not attainable metrics or a global visual-grounding upper bound.

Local `runs/semantic_polygon_grounding/` retains all raw audits, candidates,
uniform uninformative name-set probabilities, unknown visual/instance
probabilities, Goal rankings and confidence intervals. Large raw dataset audit
tables stay local; the report packages compact metrics and representative cases.
