# GeomBelief map geometry input audit

Baseline: 34d297425ac5c48c036055af62f325f20126964b. Archived Polygon experiments remain read-only. Model capacity, frozen BERT, loss, seeds 0/1/2, initialization, Train-Fit/Dev object split, 15 epochs and per-epoch sample order match the archived Full Geometry runs exactly. Correct Geometry reuses those three runs; no checkpoint is overwritten.

|Path|Correct Geometry|Name/Type Only|Distance / Direction / Both|Shuffled Geometry|
|---|---|---|---|---|
|Candidate–landmark 10-D|Original|All zero|Only channel 2 / 3,4 / 2,3,4|Donor spatial bundle|
|Physical AND relation node geometry|Original|All zero|All zero|Donor spatial bundle|
|prior / prior_buildings|Original|Zero|Zero|Donor spatial bundle|
|rules / relation_fields|Original|Zero|Zero|Donor spatial bundle|
|Edges / adjacency|Zero (geometry variant, no GAT)|Zero|Zero|Zero|
|Candidate positional encoding|Absolute fixed grid|Same absolute grid|Same absolute grid|Same absolute grid|
|Language, landmark names/types, masks|Original|Original|Original|Recipient original|

The architecture has relative_position=False. Thus grid positional features cannot recover landmark centers after erasing node geometry. Grid coordinates remain necessary output coordinates and permit an unconditional spatial prior; Name/Type Only means no supplied landmark spatial geometry, not no spatial output representation. Learned name-location memorization on Seen maps remains possible and is evaluated on held-out maps.

All relative XY magnitudes, boundary distance, Inside and Extent are removed from the simple distance/direction arms. Distance is unsigned center distance; Direction is the normalized vector. This avoids signed boundary leaking inside and raw XY leaking both distance and direction. Distances to several landmarks can themselves constrain a position; that is legitimate distance information.

Shuffle uses a fixed seeded permutation of COMPLETE geometry tensor bundles (nodes, pair descriptors, priors and rule fields), stratified by map, exact node type/validity/entity schema and Train-Fit/Dev partition. It retains every spatial tensor's empirical joint distribution and landmark count. Language/name embeddings stay with the recipient. Rule fields deliberately become spatial context from a different instruction; this is an all-spatial-grounding intervention, not a pure within-query name-binding test. Singleton strata cannot be shuffled without breaking matching constraints and remain unchanged, with exact counts in manifests. This limitation reduces intervention strength and must be reported. No GT coordinate enters partition, shuffle or network.

Correct retains all geometry paths; simple arms clear other paths. Therefore Correct-minus-simple combines descriptor and rule/prior contributions, while Distance+Direction-minus-Name isolates adding those simple descriptors. It must not be interpreted as a fair test of Polygon boundary shape.

The fixed controller is the archived pretrained trajectory/Stop head. Newly trained locator contexts are a transfer distribution; navigation cannot isolate localization independently of that shift. Formal-model controller attribution uses the complete frozen C checkpoint separately, including its original visual module and trajectory weights.

Data: 2426/2679 unique Seen/Unseen queries; navigation 2470/2697. No Test inference, training or tuning. Model selection only fixed Train-Dev R@1@20m, then NE, then earliest tied epoch. Paired 10,000-draw map bootstrap and seed means use all seeds; four Unseen map blocks do not establish broad cross-city generalization.

Navigation donor cache access is whitelisted to nodes/prior/prior_buildings/rules/relation_fields/pair_geometry before constructing policy inputs. Cached target labels are never exposed through the donor policy dictionary. Dataset targets and expert trajectories remain in the outer evaluator solely for post-rollout scoring. Oracle coordinates are an explicit exception only in separately labeled controller diagnostics.
