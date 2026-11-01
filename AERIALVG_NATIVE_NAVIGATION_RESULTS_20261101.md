# AerialVG native features: navigation validation

Status: improvement remains unproven. The frozen temporal candidate is undergoing full Seen evaluation.

The encoder uses the official checkpoint's complete Swin, HCA and decoder. Geometry-guided selection retains15 of900 native object queries with actual current RGB/depth support. The trained ranker updates a bounded global belief residual. Native evidence replaces the older SigLIP branch in this experiment. The original trajectory controller, Stop threshold0.3, altitude guard and exact predicted-goal arrival guard are preserved.

## Completed paired Seen128

Identical episode manifests, base checkpoint, dataset hashes and navigation/controller protocol. Values below are percentages except NE.

| Configuration | SR | OSR | SPL | NE (m) |
|---|---:|---:|---:|---:|
| Original GraphNav Stage1 |50.00|59.38|49.14|28.00|
| Native AerialVG queries, current view |50.00|75.00|40.60|28.66|
| Native queries, teacher plus on-policy retraining |49.22|71.88|41.73|28.16|
| Native queries, causal8-view residual window |52.34|70.31|49.40|28.61|

The temporal candidate helps13 and harms10 episodes versus the original baseline. SR difference+2.34375 percentage points, grouped bootstrap95% interval[-4.6875,+9.375]; this is insufficient evidence for a navigation improvement. Its policy stops124/128 episodes versus93/128 for native current-only. Native inference costs approximately156ms per decision versus45.5ms for the original on this pilot.

The history window was selected on1511 actual Train states using state R1, then NE, then shorter window. W1/2/4/8 R1=64.86/66.51/67.11/68.03%; predicted-goal jumps>20m=143/115/105/75. This offline ranking was followed by prospective closed-loop validation; no future frames enter inference.

On-policy retraining adds1511 states to teacher prefixes, yielding2658 states after duplicate initial observations are removed. All episode/pose/step/hash checks pass. It does not improve navigation and is not promoted. It uses20 epochs on enlarged data; no positive causal claim about distribution coverage or equal compute is made.

## Relation module and arrival diagnostics

The actual official pair transformer was tested and fine-tuned on native queries. Prepared Dev R1 plain69.92%, relation69.76%; no independent relation benefit established. The navigation temporal candidate uses the plain native-query ranker. Its connection to AerialVG is the verified official encoder/HCA/native decoder representation; no navigation gain is attributed to the relation module.

Simple arrival-radius relaxation was rejected:20m guard lowers native SR to46.875% on the exact-prefix128 counterfactual. A public18-feature arrival probe fitted/calibrated on disjoint Train instruction groups confirms10/10 Dev states but covers only9.35% of positive arrival states. Sample support is inadequate; the probe remains offline and does not alter Stop.

## Frozen full Seen confirmation

Candidate: `aerialvg_native_plain_sigma8_s0_20261101/best.pt`, real native evidence, history8, original control. Freeze record: `aerialvg_native_temporal_frozen_seen_candidate_20261101.json`. Evaluate all2470 Seen episodes without interim tuning or subsampling. Output: `aerialvg_native_temporal_navigation_full_seen_20261101`; paired report generated only after completion.

Historical full baseline uses the same checkpoint and identical annotation/object/description content hashes. All128 historical selected-baseline pose traces reproduce exactly in the fresh environment (maximum pose error0). Comparisons normalize only migrated dataset roots and preserve relative paths/content hashes; latency across historical environments is not directly comparable. Full visual-asset equality outside the reproduced subset is not independently established by annotation hashes.

Completion still requires a meaningful full paired navigation improvement and independent held-out confirmation after freezing. No new native candidate has been evaluated on Test.
