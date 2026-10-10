# Observed-only visual grounding benchmark

This benchmark runs actual pinned RemoteCLIP L/14 and B/32, RegionCLIP RN50 and
SigLIP2 Large/384 on the same CityNav pose observations. It does not modify A3,
Trajectory/Stop, running experiment checkpoints or shared Python environments.
Development starts from GitHub main `96bc2288c274cae3a4f5fc68f985510a66b8cb4b`.
The isolated main worktree lives under `Ablations/GraphNav/visual-grounding-main`.

Read `protocol.json` before running. CityRefer processed **target** phrases can
supply conservative language weak labels, but are not independent image-verified
attribute ground truth. Other objects become negatives only when that same
attribute has a supported different label. All equally matching labels are
positives; unknowns are excluded. Geometry is a weak ROI, not segmentation.
Objects/IDs/GT are evaluation linkage only; encoders receive pixels and strings.

The original preserved stride-8/max-step64 observation audit is retained. The
full benchmark additionally renders every original episode at fixed trajectory
fractions 0, .5 and .8, independent of target visibility. These are human-prefix
simulator observations, not new policy rollouts or perspective photographs.
Future RGB and full orthophotos are absent from all model forward interfaces.
The environment-private renderer reads the original raster only to render the
camera's current field of view. Every candidate comes from the same episode and
a step no later than the declared cutoff. Train object identities stay together
in 200m spatial-cell-hash partitions; map and CityNav split definitions remain.
The final Train audit finds no object or cell overlap, but 5,328 source frames
contain different objects from both partitions. ROI context can therefore be
shared. This is not a frame-family-independent supervised development split;
no semantic adapter is trained on it in this run.

Actual commands use isolated assets in `../visual-grounding-assets/`. No weights,
shared data or large image/feature caches are committed. Asset environments reuse
an existing torch 2.8/CUDA12.8 environment read-only and add dependencies locally.
RegionCLIP uses pinned official Python modules plus supported axis-aligned ops;
legacy rotated ops fail if called. Exact official region-head equivalence is
recorded in the model results, along with official RPN proposal coverage.

```bash
# CPU preparation. The parallel worker conservatively resumes completed map shards.
PYTHONPATH=. python -m experiments.visual_grounding.data.render_full_observations
PYTHONPATH=. python -m experiments.visual_grounding.data.build_parallel
# Freeze a completed split's entire eligible candidate population and an
# object-balanced attribute/retrieval index. A frozen output cannot be overwritten.
python -m experiments.visual_grounding.freeze_dataset \
  --data runs/visual_grounding_benchmark/data_audit/full/val_seen \
  --output runs/visual_grounding_benchmark/data_audit/frozen_seen
# Same --data for ALL encoders. Use the RemoteCLIP independent Python environment.
python -m experiments.visual_grounding.evaluate_zero_shot \
  --model remoteclip_l14 \
  --checkpoint ../visual-grounding-assets/remoteclip/weights/RemoteCLIP-ViT-L-14.pt \
  --data runs/visual_grounding_benchmark/data_audit/frozen_seen \
  --output runs/visual_grounding_benchmark/remoteclip_vitl14/seen
# Other --model values: remoteclip_b32, siglip2, regionclip.
# RegionCLIP additionally requires --checkpoint and --official-repo.
# Frozen Unseen uses --split unseen exactly once after the protocol/config lock.
```

`evaluate_zero_shot.py` saves normalized image/text features, every prediction,
separate native language grounding and identity-retrieval metrics, attribute and
map breakdowns, IID and map-cluster confidence intervals, and actual telemetry.
`evaluate_counterfactual.py` adds fixed Seen-only occluded/invalid-pixel probes.
`evaluate_retrieval.py` compares native mean patches and the existing identity
adapter; that 256D adapter is **not** a trained 1024D text-space semantic encoder.
`evaluate_region_proposals.py` reports official RPN recall and the restricted
covered subset. Best-IoU annotation-to-proposal association is an explicitly
annotation-assisted diagnostic upper bound, not an unsupervised proposal policy.
`benchmark_runtime.py` times warm loaded models with feature caches cleared.
`evaluate_a3_reranking.py` preserves original static rankings in ties and leaves
all Unknown scores unchanged; trusted and weak evidence results are separate.
`compare.py` computes paired differences and exports CSV/PNG comparisons.
`audit_metrics.py` adds object-balanced confidence intervals, independent
category counts, attribute Macro-F1 and the multi-positive random baseline.
`benchmark_loading.py` separates warm-cache PNG decoding from CUDA latency.
`visualize_predictions.py` renders Seen-only agreement and apparent-failure
cases from cached predictions, without model inference or Unseen inspection.

The post-freeze binding audit flags compound colors such as "gray and brown
roof" being compressed to one weak color label. These errors remain visible in
the primary frozen results; they are never relabeled using model predictions.
The 150 projection cases and 12 model comparison cases support later review,
not an assertion that all labels were independently verified.

Coverage denominators include all raw episodes, unique map+instructions, unique
objects, actual eligible ROIs and reliable independent attribute instances.
Large multi-positive galleries make R@5 an easy metric; the random R@1 expectation
is reported too. Simulated cross-view retrieval may retain raster texture identity
without language understanding. Weak labels cannot establish which model truly
understands a roof color or justify training with mislabeled negatives. The
finetuning gate stops Adapter/LoRA when independently verified Train labels cannot
support disjoint training/development sets. A stopped pilot has no invented scores.

The final report is `docs/VISUAL_FOUNDATION_BENCHMARK_20261010.md`, with local
artifacts under `runs/visual_grounding_benchmark/`. Downloads follow direct-first,
then proxy on demonstrated direct failure; partial data is resumed, not redownloaded.
