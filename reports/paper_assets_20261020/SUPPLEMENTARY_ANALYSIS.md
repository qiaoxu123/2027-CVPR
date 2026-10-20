# Offline subgroups and profiling

All 10448 archived formal episodes were rescored in XY; per-episode and aggregate NE/SR/OSR/SPL match original records. No navigation replay, checkpoint selection or training was run.

Subgroup SR is actual final navigation success. R@1 columns are initial, episode-weighted localization; they are not deduplicated-query R@1 or closed-loop success. Distance bins are half-open and cover the entire split. Relation tags are lexical, overlap, and do not establish correct language attachment or a geographic reference frame.

Saved strict projection visibility is reused for Seen2470: 332 visible / 1860 outside / 278 uncertain. Unseen/Test visibility remains not evaluated; no guessed visibility labels were created. Weak projected polygon identity is not human verification.

Same GPU model whole-episode evidence exists for both methods, with 2697 common IDs. GeomBelief mean rollout/model time is 396.68/199.78ms per episode; historical HETT local teacherfix epoch9 is 200.82/148.16ms. Native policy/step counts and training identity differ; these are not official controlled speedup estimates. The separate 22,144 GeomBelief decision timings are not mixed into per-episode values. Model/vision/environment and first-episode warmup are separated; model-load startup is not included. A folder named profile_hett actually identifies its model as GraphNav-V1 and is explicitly excluded from HETT evidence.
