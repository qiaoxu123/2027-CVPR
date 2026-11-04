# Structured TRAIN language and PGVB implementation plan

Goal: preserve original checkpoint; parse all official TRAIN texts without target GT; train standalone polygon-guided landmark visual belief.

1. Audit original TRAIN count, public catalog, parser resources, RGB/feature hashes and CUDA. Do not change default inference or controller.
2. Reuse dependency parser and public-name binding. Export text spans, entity IDs, directed edges, explicit unresolved reference frames and ambiguity. Never read target object IDs/category/coordinates to infer language. Count direct and indirect paths independently. Inspect a stratified sample with AI review explicitly distinguished from human verification.
3. Prepare current real cached RGB features plus public Polygon supervision. Use city-disjoint partitions if available (stronger than adjacent map chunks). Same RGB queried with multiple public names, matching weak positive/negative region labels. Unobserved/ambiguous windows excluded; proxy labels are not human pixel GT.
4. Train standalone PGVB: frozen SigLIP2 shared image/text vectors, centered polygon-shape tokens, cross-attention over actual observed image tokens, per-patch belief head. Absolute polygon/world center excluded from scorer. World projection follows existing Footprint mapping. Original A3 logits absent. GPU smoke, finite gradients, resume, fixed seed0 pilot budget with best/last/config.
5. Freeze training plan before held-out scoring. Compare PGVB, observation coverage, zero RGB, shuffled RGB, wrong Polygon and same-image query swaps. Report Building/Road R20/Polygon NE/probability mass and paired same-image discrimination. No navigation or direct-relation network in this round.
6. Save raw per-sample metrics, original RGB cases, training curves and concise Chinese report; tests and checkpoint SHA verification; commit/push independent branch. Preserve negative results; no repeated held-out tuning.
