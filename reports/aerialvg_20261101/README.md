# AerialVG native visual branch — synchronization snapshot

This snapshot contains the current relation/native-query implementation, its tests, completed diagnostic evidence and frozen evaluation protocol. Full2470 Seen evaluation is now complete and verified: native SR50.73% versus original47.61% (+3.12pp, whole-map CI[+1.43,+4.76]); previous calibrated residual48.70% (+2.02pp, map-cluster CI reaches0). Independent full2697 Val-Unseen confirmation is in progress; goal completion and geographic robustness over the stronger reference remain unproven.

Start with [navigation results](../../AERIALVG_NATIVE_NAVIGATION_RESULTS_20261101.md), [official implementation tests](official_validation/UNDERSTANDING_AND_TESTS.md) and [source/evidence hashes](SOURCE_SNAPSHOT.json). Negative relation, on-policy retraining and arrival-rule experiments are retained.

The published native entry points use `scripts.evaluate_native_shared_closedloop`; `scripts.evaluate_closedloop` retains the older experiment branch's resolution/residual/resume interface. Native online inference is launched with `scripts.evaluate_native_object_closedloop`, and actual Train rollouts with `scripts.collect_native_onpolicy`. This separation preserves old experiment protocols.

The current experiment expects the reconstructed `2027-CVPR` workspace layout, its sibling DATA directory, frozen GraphNav checkpoint and SBFNav source, and `third_party/AerialVG`. Official source: https://github.com/Ideal-ljl/AerialVG at `db33e1dc733911853c5e132394d2e00b024334db`. Download the pretrained weights through the official repository and verify SHA256 `f563437fd0b6f3a2e8e1b037ce15ef6ad40c81aead385fb81dae01c7953cc812`. Python package versions are in `official_validation/environment.json`. Weights, raw observations, feature caches and full per-step rollout traces remain in the local experiment workspace; this Git snapshot includes lightweight completed reports and their provenance.

Publication validation: all19 experiment test files ran in the isolated checkout:45 passed,3 skipped. Skips require native observation caches or completed rollout traces that are intentionally external assets. The live workspace separately verified exact native online/cache tensors, completed128 score recomputation and interrupted2+resumed2 trajectory equality. A source AST parse and `git diff --check` are also required before committing.

Additional completed fullSeen pair/map-cluster reports and per-decision scoring verification are included in this update. No model, controller, window or threshold was retuned.
