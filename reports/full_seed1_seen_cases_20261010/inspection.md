# Full seed1 Seen trajectory inspection

Twelve post-rollout illustrative comparisons against the frozen local SBF checkpoint. The deterministic first three examples in each outcome group all happen to lie in `birmingham_block_1`; they are not a random, map-balanced sample or a failure-rate estimate. Whole source orthophoto, reference trajectory and GT target are diagnostic-only, never policy inputs. Aggregate rates use all2470 Episodes separately.

Visually inspected `gained_0.png`: Graph executes11 bounded moves and stops14.52m from target; SBF finishes55.10m away after19 moves/stagnation. This supports the concrete observed outcome, not a causal attribution to GAT or RGB.

Visually inspected `lost_8.png`: instruction describes a red car in a driveway across from Willmore Road. SBF finishes11.45m from target; Graph stops62.86m away at a different location after9 moves. The wrong location and direct route are consistent with a goal-grounding/false-Stop failure. The plot alone cannot distinguish language, appearance, map binding or trained head contributions; controlled visual-vector tests are recorded independently.

The same source map provides both gains and losses. Stronger geometry can improve one description while missing another. A query-conditioned appearance mechanism is a next hypothesis, not an improvement established by these selected pictures. No model/hyperparameter change is made from this inspection.
