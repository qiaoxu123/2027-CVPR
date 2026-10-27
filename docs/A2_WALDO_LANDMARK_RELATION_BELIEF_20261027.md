# A2: Language-Landmark-Conditioned WALDO Box Relations for Belief Refinement

**Status (2026-10-27):** design and standalone, testable pair-relation/fusion primitives; **no fully integrated CUDA experiment or claimed navigation gain**. This is a separate branch off main. The frozen A0 formal GraphNav-V1 checkpoint, legacy evaluator, and A1 experiments must not be changed.

## Research question

Can language-grounded object proposals provide a *causal, spatially localized* correction to A0's global goal belief by scoring the relationship between every WALDO box and **each landmark actually referred to in the instruction**?

Earlier `research/local-semantic-geometry-refinement-20261027` already used BERT landmark **names** and box/reference relative geometry, but had only 30 informative train views (46 with verified positives), 5 informative dev views, and an ineligible correction gate. A2 must explicitly add **instruction-specific mention and relation-context encoding**, not merely repeat name embeddings or claim the old branch improved navigation.

## Architecture

```
instruction + public CityRefer linked references
    |                                        |
frozen BERT full contextual tokens      validate map-linked reference identity
    |                                        |
named landmark mention spans           landmark world polygons/centroids
    |                                        |
separate mention + name + sentence        |
context encoder per landmark             |
    |                                        |
    +-----------------------+----------------+
                            |
current observed 384x384 RGB -> frozen WALDO boxes -> box polygon image-to-world
                            |
                 frozen SigLIP2 box appearance
                            |
              each valid box x each valid named landmark
                            |
       geometry: signed dx/dy (east/north), distance, bearing,
       signed polygon-boundary separation, containment, area ratio,
       confidence; explicit validity and projection uncertainty
                            |
     language-conditioned pair-relation attention / candidate ranking
                            |
    trusted-match gate + local support -> bounded mass-conserving fusion
                            |
      updated original A0 B_t -> original trajectory/Stop controller
```

## Landmark language encoding (NEW and essential)

A *reference landmark* is an entity used in the instruction, e.g., `the library` or `the church`, and is **not** itself the destination box.

1. Use the same frozen full-sentence BERT contextual token states as A0; preserve all token positions and their mask.
2. For each verified, uniquely linked public CityRefer reference, find its explicit mention span using source-linked names/aliases plus tokenizer offset mappings; never infer a unique map identity from the GT target or fuzzy name similarity alone. Treat duplicates/ambiguous or missing alignments as unresolved.
3. Encode **separately per reference**:
   - phrase/span token pooling (instruction-contextual representation);
   - frozen BERT reference name tokens (name identity);
   - phrase-to-full-instruction attention to capture role/nearby relational predicates (e.g., `east of the library`, `between the church and the station`).
4. The learned projection/attention is trainable, while BERT stays frozen. Use explicit mention/link-validity masks. Failed linkage **does not create a false landmark**; no reference evidence => A0 fallback.
5. Do not claim generic open-vocabulary reference detection or role parsing unless separately demonstrated. For multi-reference `between` instructions, an explicit **ordered-pair/triad** feature is planned after basic per-pair validation.

## Every box x landmark pair (not only nearest/top reference)

For WALDO boxes `i=1..N` and linked landmarks `j=1..L` create `R_ij`, with:
- `dx,dy` in world-map metres (declare east = positive x, north = positive y only after verifying this transform for each map);
- normalized distance, sin/cos bearing, log distance;
- signed point-to-landmark-polygon boundary distance and inside/outside flag;
- box/landmark polygon area ratio and WALDO confidence;
- optional observed-surface height differences only with valid masks (DO NOT label invalid depth as zero height);
- polygon vs rectangular projection and coordinates must be audited; world polygon is a geometric approximation to the observed box rather than an actual object segmentation.

Do NOT use the private GT target's coordinates/identity during inference or to select a box. Unobserved boxes and missing references are `unknown`, not negative evidence. WALDO box confidence is not a calibrated target-match probability.

The pair tensor must have shape `[B,N,L,F]` with independently verified `box_valid`, `reference_valid`, `pair_valid`. Query all references, not only the closest one. Variable N/L have correct masks and numerical stability.

## Local belief update

Start from the **actual A0 dynamic belief** `B_t^{A0}`, not merely the static `B_0`. Form `Q_t` from trusted candidate scores convolved with validated, observed-support grid masks and candidate-local kernels. Define `S` as the union of supported cells and `m_S=sum_{q in S} B_t^{A0}(q)`.

- Inside `S`: `B_t^{A2}(q)=(1-alpha) B_t^{A0}(q)+alpha m_S Q_t(q)`.
- Outside `S`: `B_t^{A2}(q)=B_t^{A0}(q)`.
- `0 <= alpha <= .5`, calibrated **only** with internal TRAIN calibration groups, and forced **0** when reference linkage, true box visibility, ranked candidates, projected support, or checkpoint eligibility is insufficient.
- Keep the original 28x28 A0 controller. A coarse grid cannot distinguish boxes mapping to the same cell, even if the pair scorer can.

An uncalibrated scorer does not authorize nonzero correction: default should be `alpha=0` until risk/benefit is verified.

## Training order and controls

1. **Data first:** run a funnel audit of WALDO visibility, true-target detection recall, world projection, public map reference linking, token-span quality, and trusted target/other-instance box association. Fix the previous severe scarcity of reliable positive/negative contrasts. Never treat unassociated boxes as hard negatives.
2. Freeze A0, WALDO, BERT, SigLIP2, trajectory/Stop; train only mention/name fusion, pair relation model, candidate ranking, matchability and optional correction confidence.
3. Train with group-disjoint target instances and matched observation time; positive box vs same-category spatial/visual confusing negatives; use labels only for loss. Predict relation direction/distance as meaningful auxiliary losses rather than solely candidate rank. Where no positive/negative evidence, mask ranking loss.
4. Check instance-level dev for newly broken correct predictions; select/freeze checkpoint using internal TRAIN-dev, calibrate alpha on separate TRAIN-calibration. No Val-Unseen tuning.
5. First do a **small paired A0 vs A2 pilot**; only after safe nonzero correction and coverage evidence run the complete **Val-Seen 2470 + Val-Unseen 2697 episodes** with the same frozen A0 checkpoint and controller. Report full-mix and supported-subset metrics separately (SR/SPL/NE/R@1, newly corrected vs broken, proposal recall, unchanged fallback fraction, compute latency), paired per map.

Mandatory diagnostic controls: A0, A2, A2 with text-landmark evidence suppressed, A2 with box-landmark geometry suppressed, link permutation/ambiguous reference, image-relative vs world-coordinate direction, empty/failed detection, and counterfactual language constraints **with valid corresponding targets**. These controls are planned, not run.

## Minimal code milestones

- `experiments/a2_waldo_landmark_belief/core.py`: standalone phrase/name encoding, box–landmark geometry, pair scoring and gated conservative fusion; **no A0 policy override**.
- `tests/test_a2_landmark_relations.py`: CPU shape, invalid span, orientation, gradients, fallback and probability conservation.
- Next: adapter to original A0 `stage1_runtime.py` via an explicit opt-in model identifier; frozen input hashes and causal current/history; detector and SigLIP encoder from validated earlier branch with no private-label imports.
- Finally: reliable labels, training, calibration, full benchmark. Until then A2 is a research prototype, not trained policy.

## Comparison identity

A0 = archived original model and original visual residual. A1 = integrated RGB fusion; it is a separate architecture study. **A2 = the same frozen A0 plus an observed, language-landmark-conditioned WALDO instance correction.** A2 must not silently absorb A1 or retrain A0.
