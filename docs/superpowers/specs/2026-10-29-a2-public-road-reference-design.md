# A2 public road reference repair — proposed next revision

## Purpose and authority

The user authorizes improvement and iterative testing toward A2 exceeding A0. This design addresses a verified missing-input failure; it does not assert navigation improvement. Preserve the running V3 experiment, all original A0/A1 results, immutable data partitions and independent confirmation records. No production change is deployed by this document.

## Evidence

Frozen epoch04 DEV observations contain 94 A0-error cases with a nearby category-compatible candidate but no usable landmark relationship. Of these, 86 involve ambiguous/unverified public names and eight involve names absent from the instruction. A CPU-only topology audit finds that 70 observations across 16 target identities could regain at least one literal instruction mention under the grouping rule below. These repeated observations are neither independent successes nor proof of correct object identity. Latest completed V3 epoch22 still fails admission: 26 corrections and 212 harmful changes in the internal real-observation diagnostic.

## Scope and exclusions

Repair public-map reference resolution and the downstream representation of its geometry. Keep A0, its original RGB path, controller, WALDO, BERT, SigLIP2, 28x28 grid, seed and train/dev target assignments frozen. Do not use destination identity, target coordinates, private target category, benchmark outcomes or calibration results to resolve names. Do not relax literal instruction matching, introduce guessed aliases, choose the largest object, choose the object nearest a target, or force belief correction. Missing candidates and harmful updates remain separate problems; measure them rather than claiming this repair solves them.

## Resolution rule (implementation choice)

Add an explicit versioned resolver mode; legacy is the default for existing V3. Consider all public exact-normalized matches of one query on one map, not just a selected convenient subset. Multiple matches may form one road reference only if every geometry is finite, nonempty, valid and positive-area; all public types are TrafficRoad/Footpath; at least one is TrafficRoad; and the exact polygon-intersection graph is connected. No learned or tuned distance tolerance. Disconnected homonyms, building homonyms and malformed records remain unresolved. This is a conservative grouping hypothesis, not external confirmation of entity identity.

Use deterministic member ordering and a public membership-derived identifier. Store original member IDs, original contours, canonical public name, type membership, resolver version and union geometry. Deduplicate query aliases consistently; normalization or ordering must not change results. Reject alias collisions across distinct resulting entities. Repeated aliases of one entity must not create duplicate attention evidence.

Deduplicate normalized queries before resolution; entity identity is (map, sorted member IDs), not alias spelling. Canonical spelling/type text is deterministic from sorted public members. Union accepted public aliases once. If distinct entities claim one exact alias or overlapping instruction mention, mask all conflicting entities. Reject partially overlapping membership groups instead of silently enlarging them. These choices were checked independently against the current resolver/mention consumers.

## Exact geometry representation

Existing code assumes one Nx2 contour in core.box_landmark_geometry and perception.extract. Do not pass concatenated rings, a bounding rectangle or a convex hull into that API. Introduce a separately versioned public region representation containing polygon exterior/interior rings; retain the original constituent shapes for auditing. Build the exact geometric union, supporting holes and multiple components. A connected intersection graph can still yield a MultiPolygon when members touch at a point.

For merged references, compute area and centroid from the exact union; point containment uses the union; signed distance measures distance to the union boundary, positive inside and negative outside, zero on the boundary. Holes count as outside. Preserve the original twelve relationship features, scaling, and missing-height mask. Existing singleton geometry must retain its legacy code path so prior inputs remain reproducible.

Builder inspection confirms that teacher/V1 caches lack native RGB/depth, and V2 retains an RGB-valid bitmap but no native RGB/depth. Consequently this revision explicitly disables height evidence for newly grouped references in BOTH reconstructed data and live inference: height difference=0, height-valid=0, metadata states missing by policy. Never synthesize merged height from candidate surfaces or old single-reference summaries. Singleton height sampling remains unchanged. This intentionally loses one source of evidence; the model must learn with the same missing input it sees at inference. Exact region height rasterization is deferred rather than silently approximated.

The private training angle/distance labels also currently use ref['contour'] and core._center in relation_v3_data.build_training_row (and the legacy private_labels.label_view). New-version labels must use the same exact union centroid as public pair geometry; old labels and artifacts remain immutable. A versioned region accessor should own this agreement.

## New cache and execution boundary

Never flip valid-reference flags in old caches: excluded BERT name features and pair geometry are absent. Inventory which TRAIN and DEV records require reconstruction, including legacy teacher and closed-loop source schemas. Recover instruction text only from the identity-checked bound annotation. Reuse verified unchanged image crops/detections where their public provenance suffices; regenerate landmark names, mentions, relationships and height metadata from the original public observation. If required observation data is absent, explicitly mark that record unreconstructable rather than inventing geometry or silently dropping it.

Write new versioned caches and manifests with parent raw SHA, source annotation SHA, public object SHA, resolver/geometry code SHA, roles SHA and feature schema. Existing V3 cache bytes and running source inventory remain untouched. Before any next GPU extraction/training, verify the current GPU job is terminal and use the shared serial lock. This repair does not authorize shortening the required 20+5 training or bypassing calibration.

## Required verification

Unit tests: same-name touching road/footpath; disconnected roads; building homonyms; footpath-only sets; malformed/nonfinite/zero-area polygons; ordering and alias invariance; point-touching MultiPolygon; concave unions and holes; union area/centroid/boundary distances; grouped height explicitly missing in cache AND live inference; private angle/distance agrees with public union centroid; instruction absent/truncated/overlapping spans; exact legacy singleton parity; no private target input accepted.

Data audit: complete TRAIN and DEV inventories, reconstructable counts, recovered mention counts and valid-pair counts; finite/aligned tensors and unchanged partitions; source and parent hashes; no confirmation/benchmark records opened. Diagnose target-level clustering and report both observations and distinct targets.

Before full training: CPU tests, actual CUDA forward/backward smoke test, frozen-parameter checks, serial GPU check, and A0/OFF parity. Full training remains 20 ranking plus five refinement epochs with complete histories and TensorBoard. Select solely on internal validation, calibrate on independently reserved TRAIN identities, freeze once, then run the mandated complete Seen2470 and Unseen2697 navigation evaluations. If safety fails, do not enable the new correction. Goal completion requires verified A2>A0, not improved mention counts.

## Remaining builder checks

Production resolver entry is policy.set_episode; audit_source is another call. Define the region schema once and share it between cache reconstruction, label rebuilding and live evaluation. Validate all source schemas before assuming every record can be reconstructed. Missing grouped height has an explicit shared policy above; any other unavailable required input must be reported rather than silently invented.

CPU-only schema inventory completed on one TRAIN and one DEV example per available source kind (five files total), with hashes/role checks through read_training_source. It confirms public network/spatial tensors, candidate geometry/surface summaries and camera footprints are available in these examples; this is a schema feasibility check, not proof that all 39,087 rows can be rebuilt. Detailed local inventory: revision_v3/remote_migration/next_revision_schema_inventory.json. No GPU context was initialized and no source tensor was modified.
