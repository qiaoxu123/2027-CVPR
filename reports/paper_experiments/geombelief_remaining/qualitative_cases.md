# GeomBelief qualitative cases

All images are rendered from real CityNav RGB-D at actual executed poses. Map polygons, language, goal annotations and trajectories come from recorded public inputs and evaluation-only GT. English figure labels; no invented paths.

## visual_improved

Episode `['birmingham_block_5', 128, 2]`; val_unseen; NE 19.59 m. Source trace and SHA: qualitative_manifest.json.

![visual_improved](figures/visual_improved.png)

## visual_degraded

Episode `['birmingham_block_5', 106, 2]`; val_unseen; NE 57.02 m. Source trace and SHA: qualitative_manifest.json.

![visual_degraded](figures/visual_degraded.png)

## false_stop

Episode `['birmingham_block_5', 1, 1]`; val_unseen; NE 110.04 m. Source trace and SHA: qualitative_manifest.json.

![false_stop](figures/false_stop.png)

## Correct versus shuffled geometry belief

Pending all static models being frozen. The heatmap uses predeclared Seed0, never a selected favorable seed. No fabricated probability field is substituted for missing logits.
