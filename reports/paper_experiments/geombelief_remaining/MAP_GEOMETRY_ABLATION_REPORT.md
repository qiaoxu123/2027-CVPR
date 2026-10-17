# GeomBelief: Language-Grounded Map Geometry


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

## Per-seed results

|endpoint|split|arm|seed|status|R@1|R@4|R@8|SR|SPL|OSR|NE|False_Stop_Rate|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|localization|val_seen|Language + Correct Geometry|0|Done|0.42003297609233303|0.6343775762572135|0.738252267106348||||31.27512907537172||
|localization|val_seen|Language + Correct Geometry|1|Done|0.40189612530915086|0.6281945589447651|0.7374278647980215||||31.84552330830229||
|localization|val_seen|Language + Correct Geometry|2|Done|0.40272052761747734|0.6240725474031328|0.7440230832646332||||31.408823947201853||
|localization|val_seen|Language + Name/Type Only|0|Pending|||||||||
|localization|val_seen|Language + Name/Type Only|1|Pending|||||||||
|localization|val_seen|Language + Name/Type Only|2|Pending|||||||||
|localization|val_seen|Language + Shuffled Geometry|0|Pending|||||||||
|localization|val_seen|Language + Shuffled Geometry|1|Pending|||||||||
|localization|val_seen|Language + Shuffled Geometry|2|Pending|||||||||
|localization|val_seen|Language + Distance Only|0|Pending|||||||||
|localization|val_seen|Language + Distance Only|1|Pending|||||||||
|localization|val_seen|Language + Distance Only|2|Pending|||||||||
|localization|val_seen|Language + Direction Only|0|Pending|||||||||
|localization|val_seen|Language + Direction Only|1|Pending|||||||||
|localization|val_seen|Language + Direction Only|2|Pending|||||||||
|localization|val_seen|Language + Distance + Direction|0|Pending|||||||||
|localization|val_seen|Language + Distance + Direction|1|Pending|||||||||
|localization|val_seen|Language + Distance + Direction|2|Pending|||||||||
|localization|val_unseen|Language + Correct Geometry|0|Done|0.1862635311683464|0.3445315416200075|0.4494214259051885||||53.380361738868864||
|localization|val_unseen|Language + Correct Geometry|1|Done|0.20119447555057857|0.3684210526315789|0.48114968271743186||||53.745322871848956||
|localization|val_unseen|Language + Correct Geometry|2|Done|0.2123926838372527|0.3602090332213512|0.46061963419186264||||51.84313869600912||
|localization|val_unseen|Language + Name/Type Only|0|Pending|||||||||
|localization|val_unseen|Language + Name/Type Only|1|Pending|||||||||
|localization|val_unseen|Language + Name/Type Only|2|Pending|||||||||
|localization|val_unseen|Language + Shuffled Geometry|0|Pending|||||||||
|localization|val_unseen|Language + Shuffled Geometry|1|Pending|||||||||
|localization|val_unseen|Language + Shuffled Geometry|2|Pending|||||||||
|localization|val_unseen|Language + Distance Only|0|Pending|||||||||
|localization|val_unseen|Language + Distance Only|1|Pending|||||||||
|localization|val_unseen|Language + Distance Only|2|Pending|||||||||
|localization|val_unseen|Language + Direction Only|0|Pending|||||||||
|localization|val_unseen|Language + Direction Only|1|Pending|||||||||
|localization|val_unseen|Language + Direction Only|2|Pending|||||||||
|localization|val_unseen|Language + Distance + Direction|0|Pending|||||||||
|localization|val_unseen|Language + Distance + Direction|1|Pending|||||||||
|localization|val_unseen|Language + Distance + Direction|2|Pending|||||||||
|navigation|val_seen|Language + Correct Geometry|0|Done||||0.4202429149797571|0.40695244965783456|0.5016194331983805|31.231796742570143|0.5797570850202429|
|navigation|val_seen|Language + Correct Geometry|1|Done||||0.40850202429149796|0.39571671287508237|0.5020242914979757|31.623739089877162|0.5912408759124088|
|navigation|val_seen|Language + Correct Geometry|2|Done||||0.4093117408906883|0.39785699943384206|0.4983805668016194|31.301594577689816|0.5904298459042985|
|navigation|val_seen|Language + Name/Type Only|0|Pending|||||||||
|navigation|val_seen|Language + Name/Type Only|1|Pending|||||||||
|navigation|val_seen|Language + Name/Type Only|2|Pending|||||||||
|navigation|val_seen|Language + Shuffled Geometry|0|Pending|||||||||
|navigation|val_seen|Language + Shuffled Geometry|1|Pending|||||||||
|navigation|val_seen|Language + Shuffled Geometry|2|Pending|||||||||
|navigation|val_seen|Language + Distance Only|0|Pending|||||||||
|navigation|val_seen|Language + Distance Only|1|Pending|||||||||
|navigation|val_seen|Language + Distance Only|2|Pending|||||||||
|navigation|val_seen|Language + Direction Only|0|Pending|||||||||
|navigation|val_seen|Language + Direction Only|1|Pending|||||||||
|navigation|val_seen|Language + Direction Only|2|Pending|||||||||
|navigation|val_seen|Language + Distance + Direction|0|Pending|||||||||
|navigation|val_seen|Language + Distance + Direction|1|Pending|||||||||
|navigation|val_seen|Language + Distance + Direction|2|Pending|||||||||
|navigation|val_unseen|Language + Correct Geometry|0|Done||||0.1846496106785317|0.17902334271543008|0.2810530218761587|53.61244037808063|0.8153503893214683|
|navigation|val_unseen|Language + Correct Geometry|1|Done||||0.19799777530589543|0.19199550414963018|0.28846866889136075|53.95016535406546|0.8017817371937639|
|navigation|val_unseen|Language + Correct Geometry|2|Done||||0.20986281053021877|0.20444530777231737|0.310715609936967|51.99855859739478|0.7887271369914147|
|navigation|val_unseen|Language + Name/Type Only|0|Pending|||||||||
|navigation|val_unseen|Language + Name/Type Only|1|Pending|||||||||
|navigation|val_unseen|Language + Name/Type Only|2|Pending|||||||||
|navigation|val_unseen|Language + Shuffled Geometry|0|Pending|||||||||
|navigation|val_unseen|Language + Shuffled Geometry|1|Pending|||||||||
|navigation|val_unseen|Language + Shuffled Geometry|2|Pending|||||||||
|navigation|val_unseen|Language + Distance Only|0|Pending|||||||||
|navigation|val_unseen|Language + Distance Only|1|Pending|||||||||
|navigation|val_unseen|Language + Distance Only|2|Pending|||||||||
|navigation|val_unseen|Language + Direction Only|0|Pending|||||||||
|navigation|val_unseen|Language + Direction Only|1|Pending|||||||||
|navigation|val_unseen|Language + Direction Only|2|Pending|||||||||
|navigation|val_unseen|Language + Distance + Direction|0|Pending|||||||||
|navigation|val_unseen|Language + Distance + Direction|1|Pending|||||||||
|navigation|val_unseen|Language + Distance + Direction|2|Pending|||||||||

## Three-seed means and sample standard deviations

|endpoint|split|arm|R@1_mean|R@1_std|SR_mean|SR_std|SPL_mean|SPL_std|NE_mean|
|---|---|---|---|---|---|---|---|---|---|
|localization|val_seen|Language + Correct Geometry|0.40821654300632043|0.010241629652972898|||||31.50982544362529|
|localization|val_unseen|Language + Correct Geometry|0.19995023018539257|0.013108938351191661|||||52.98960776890899|
|navigation|val_seen|Language + Correct Geometry|||0.4126855600539811|0.006557371451150517|0.40017538732225305|0.0059658726992511775|31.385710136712373|
|navigation|val_unseen|Language + Correct Geometry|||0.19750339883821533|0.012613868072844474|0.1918213848791259|0.012711876925878031|53.18705477651363|

## Primary map-cluster paired intervals

|comparison|metric|delta_pp|map_CI95_pp|maps|
|---|---|---|---|---|

Confidence intervals are pointwise, not multiplicity-adjusted. Episode bootstrap, equal-map estimates, every map and leave-one-map-out are in paired_statistics.json. Four Unseen maps are map blocks, not four independent cities. No missing or partial result is treated as zero.

## Conclusions

- correct − name_type, Unseen R@1: Pending complete three-seed paired results.
- correct − name_type, Unseen SR: Pending complete three-seed paired results.
- correct − shuffled, Unseen R@1: Pending complete three-seed paired results.
- correct − shuffled, Unseen SR: Pending complete three-seed paired results.
- distance_direction − name_type, Unseen R@1: Pending complete three-seed paired results.
- distance_direction − name_type, Unseen SR: Pending complete three-seed paired results.

Correct-minus-Name tests all map geometry together, not uniquely language–geometry interaction. A language-shuffle factorial would be needed to isolate that interaction; none is claimed here. Fixed-head transfer and possibly memorized Seen map/name priors limit causal interpretation of navigation.
