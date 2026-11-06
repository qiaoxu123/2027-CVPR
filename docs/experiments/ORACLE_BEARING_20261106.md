# Oracle bearing diagnostic, frozen before scoring

No training. Baseline: matched public-bound full instruction checkpoint 9ac51ee253eed46acbf66e6afb47888857b70258f5fcd195721723d5112dc387, full Val-Seen SR1170/2470. Public bindings unchanged. Existing PR26 uses whole contour vertex mean (not area centroid); retain that anchor. World-map XY angle atan2(dY,dX), no geographic-north claim.

A unchanged. B mean cosine angular consistency over all bound valid Polygons. C negative mean perpendicular distance to forward rays divided by one world grid-cell diagonal; exclude candidates behind ANY ray. If no forward-support candidate exists, abstain and retain A. D original logits + lambda * 4 * mean cosine, lambda fixed at 0.5,1,2,4. B/C ties break by original logits only. No range from GT; GT used only once to compute angle from each already bound anchor. Exactly coincident Anchor and GT (within numerical 1e-9m) undefined: abstain on whole task, never omit an ambiguous anchor based on GT. No landmarks/invalid Polygon: retain A. Single unique valid Polygon is primary subgroup; multiple distinct names and same-name multiPolygon reported separately, all bound instances retained equally.

Offline all2470 using cached baseline model logits and the exact784 grid locations. R@1/5/10 means target within20m among topK; nearest-grid error diagnoses grid support, separately from topK ranking. Also report nearest-grid rank, first20m-hit rank and sourceTop20 recoverability. No instance matching claims.

Before reading offline outcomes, fix closed-loop arms B,C,D(lambda1); all2470 each, same controller/Stop/Horizon and initialization. Do not choose weights using Val. Other fixed D weights are offline sensitivity diagnostics only. No Oracle target range, candidate index or GT target vector supplied to runtime; it receives only frozen angles indexed by PUBLIC task text plus start pose. Rebuild candidate IDs/XY/context from corrected full784 logits before unchanged stableGoal/controller. Isolated tools only; normal evaluator/model code untouched.

Workflow: CPU geometry and leakage contract tests -> cache/GT-angle generation and offline -> real GPU forward+full closedloop in tmux -> paired results/heatmaps/report/push. No Test, no main changes. Persist SHA/config/result per phase.

Numerical tie policy: pure B/C add <=1e-4 times normalized original rank, to resolve FP32 near-ties without any GT. This tolerance is fixed for all tasks. It can affect cosine differences below1e-4 and ray scores below1e-4 cell diagonal; report accordingly. No weight selection after scoring.
