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

Actual normalized model probabilities, predeclared Seed0 and first Val-Unseen query with changed public node geometry; no result-based selection. Cyan outlines are the geometry supplied to each model. Recipient language and landmark names are unchanged. Separately budget-matched trained models; not a same-weight feature intervention. Source IDs, donor and checkpoint SHAs: belief_figure_manifest.json.

![Actual goal belief](figures/correct_vs_shuffled_belief.png)
