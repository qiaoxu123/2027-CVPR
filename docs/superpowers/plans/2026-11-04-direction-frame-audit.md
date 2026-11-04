# Direction frame and 34-case audit

Base fef6c37. Read-only diagnostics; no retraining, weights, inference-coordinate changes or GT-fitted direction rules.

- [ ] Search primary CityNav/CityRefer/SensatUrban sources for explicit compass/georeferencing evidence, distinguish code conventions from author-verified geographic axes.
- [ ] Recheck all34 raw instructions, target identities/coordinates, public name/Polygon bindings, grammatical relation subject/object and source direction.
- [ ] Compare Anchor grid mean, continuous centroid, all fragments, projection intervals and nearest-boundary facts only as offline diagnostics, never GT-selected inference inputs.
- [ ] Render original human start/end poses and full real orthophoto map for all34; overlay +X/+Y, Footprint, public Anchor, GT and immutable prior Target Belief. Do not claim final trajectory yaw is language-annotation reference.
- [ ] Current assistant reviews instructions/RGB, labels uncertain frame/instance cases unknown; no Qwen and no automatic claimed human gold.
- [ ] Source/weight/prediction SHA, tests, per-case JSON/CSV, HTML/figures and Chinese report, commit/push.

If no independently verifiable mapping is available, state that explicitly rather than choose the best rotation from GT. Sources are reference material, not additional train labels.
