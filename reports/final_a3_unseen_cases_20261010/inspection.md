# Source-backed post-rollout trajectory inspection

Twelve deterministic cases: the first three in each gained/lost/both-success/both-failed outcome group. All happen to be birmingham_block_5. This is neither random nor map-balanced sampling and cannot estimate failure prevalence or map-wide gains. Full map/episode statistics are supplied separately. Source orthophoto, target star,20m circle and human reference path are offline diagnostic overlays; target coordinates and expert paths did not enter policy inference. These are not perspective-camera flight recordings or collision checks.

Visually inspected `gained_41.png`: Graph reaches11.2495m from the target and Stops after8 motions. SBF passes through the success region(OSR1) but finishes59.7049m away after20 motions/stagnation. The plot shows Graph's shorter route ending inside the20m circle and SBF's late departure. The brown-roof/landmark instruction is preserved, but this case alone cannot attribute the outcome to geometry, appearance or Stop.

Visually inspected `lost_11.png`: Graph stops44.4055m from the target after10 motions, outside the circle; SBF finishes16.9812m away at its20-decision horizon. The instruction describes several nearby buildings and appearance details. Graph's wrong endpoint is consistent with goal ambiguity; it is not proof that a particular phrase or missing visual feature caused the error. A3 is the geometry-only control, not the full visual model.

The manifest also retains both-failed0, where both OSR1 outcomes end far from the target, and all other selected successes/failures. Negative cases have not been dropped. Mean outcomes remain those of the full2697-episode evaluation, not these selected pictures.
