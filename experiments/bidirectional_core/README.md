# Bidirectional core test

Actual CityFlight measured positions supply forward supervision and an offline
reverse traversal. This is a pilot over a fixed 2,048 Train and 128 Seen trajectory
families, not a claim that new human return flights were recorded. Original map
splits stay unchanged; internal Train/Dev separation covers complete families,
object identities and all visited 200 m regions.

E0 repeats forward states to match E1/E2's total number of optimization steps.
E1 uses forward and reverse states; E2 adds conservative multi-positive region/text
supervision. The same lightweight head, initialization, optimizer, controller and
three-epoch budget apply to every condition. A3 Epoch14, BERT and SigLIP2 remain
frozen. Epoch 3 is evaluated regardless of Seen results.

Destination coordinates, future waypoints and target IDs belong to labels or
evaluation metadata. `CoreModel` rejects private fields in its public inputs.
Public map polygons supply region proposals and geometry. Each native SigLIP2
transformer sees only the actual current 384px observed crop; the official
attention-pooling head pools supported patches in each projected ROI. Contextual
frame-patch ROI features are not independent high-resolution crop encodings.
The frozen image transformer runs in FP16 and the native masked attention head
in FP32. A 16-frame actual-ROI FP32 comparison must pass the fixed cosine/error
tolerances before training. The initial BF16 cache failed the small-ROI check
and remains as an audit artifact; its image vectors are excluded from training
and formal evaluation. The text tower's BF16 features passed their FP32 check.
Nonpositive camera height produces Unknown visual evidence. Unobserved grid
scores remain exactly unchanged under visual reranking.

Return language uses auditable map/anchor geometry. A launch point without a
reliable corresponding object receives trajectory supervision, not fabricated
visual identity or color supervision. Unverified linguistic attributes are weak
labels. Unknown objects are excluded from contrastive negatives; multiple judged
positives are supported. A usable contrastive row requires an observed positive
and a judged negative. Effective pair counts must be reported, not inferred from
trajectory counts.

```sh
python -m experiments.bidirectional_core.data --help
python -m experiments.bidirectional_core.data_pilot --help
python -m experiments.bidirectional_core.cache --help
python -m experiments.bidirectional_core.reencode --help
python -m experiments.bidirectional_core.precision --cache runs/bidirectional_core_test/cache_fp16/val_seen --output runs/bidirectional_core_test/smoke/native_roi_precision_fp16.json
python -m experiments.bidirectional_core.train --cache runs/bidirectional_core_test/cache_fp16/train_seen --condition E0 --output runs/bidirectional_core_test/E0 --epochs 3 --batch-size 32 --seed 0
python -m experiments.bidirectional_core.runtime --episodes runs/bidirectional_core_test/data/val_seen/pilot_episodes.jsonl --checkpoint runs/bidirectional_core_test/E0/epoch03.pt --static-cache runs/bidirectional_core_test/cache_fp16/val_seen --output runs/bidirectional_core_test/navigation/E0
python -m experiments.bidirectional_core.evaluate --help
python -m experiments.bidirectional_core.run_evaluation
```

Repeat the fixed training/evaluation commands for E1 and E2. Exact executed
commands, resource hashes and timings live beside local results. Existing runs
and checkpoints are never overwritten; model weights and large caches stay local.

The shared controller uses 50 m horizontal/10 m vertical bounds, 20 decisions,
10–100 m above-ground altitude limits, a 0.5 calibrated Stop threshold and a
20 m guard around the selected predicted candidate. SR/SPL use the released XY
20 m criterion after rollout. Policy rollouts cannot access target coordinates or
teacher paths. Reverse starts at the measured forward endpoint; an endpoint
outside the annotated forward goal radius is retained and reported.

Evaluation separates navigation, weak region/text selection and frozen A3
reranking. Same-type real RGB feature swaps preserve public positions, language,
quality and candidate availability. Original native minimum-side bins are 8–23,
24–47 and at least 48 pixels; these observational bins do not establish a causal
resolution effect. Paired trajectory-family bootstrap and map-cluster intervals
are required. Unseen is held out until the fixed Seen comparison has an actual
benefit; it is never used to tune prompts, thresholds or checkpoints. No complex
GAT or broad joint fine-tuning follows an E2 comparison without clear benefit.
