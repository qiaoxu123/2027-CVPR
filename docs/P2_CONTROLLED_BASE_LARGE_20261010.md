# P2 controlled frozen Base/Large comparison

Same trusted P0 source,837 training/94 internal-dev identities,836 eligible hard-negative pairs/epoch,5 epochs,seed0,batch32,microbatch16,workers4,hidden256,8 attention heads. Epoch-indexed view selection is worker/model independent. Both pretrained vision towers are frozen; no LoRA is applied. Head checkpoint and backbone choice use internal Train Seen dev R@1 only.

| Encoder | Dev R@1 | Dev R@5 | Unseen R@1 | Unseen R@5 | Same-city/type R@1 | R@5 |
|---|---:|---:|---:|---:|---:|---:|
| SigLIP2 Base224 |82.98%|94.68%|72.17%|92.17%|78.07%|92.98%|
| SigLIP2 Large384 |86.17%|97.87%|73.04%|93.91%|77.19%|96.49%|

Large is retained by the predeclared dev criterion. The115 Unseen queries span4 maps; Large R@1 improves only1 net identity over Base. Paired identity-bootstrap95% interval for R@1 difference is−7.83..+9.57 percentage points; R@5−4.35..+7.83pp. No stable cross-map or navigation advantage is demonstrated. Large improves Building R@1 (87.18% vs74.36%) but declines on Footpath (40.00% vs60.00%) and TrafficRoad (67.24% vs72.41%). Rare categories have insufficient support.

Five-epoch training took50.50s Base and56.28s Large; peak allocated memory1.94GiB and4.10GiB. These are cached head-training pipeline measurements, not pure backbone benchmarks:Base encoded6363 images with0 disk hits,Large encoded3046 with3317 preexisting frozen-feature hits. Native224 vs384 resolutions also confound a pure model-size comparison. Revisions/config/source SHA/epoch timings/coverage are in `reports/p2_controlled_20261010/`.

Checkpoints remain local:`runs/p2_controlled_base224/visual_best.pt` and `runs/p2_controlled_large384/visual_best.pt`; full histories,initial-head and frozen-mean controls and failures are retained. The Large trained head improves over frozen mean patches by6.09pp R@1 and12.17pp R@5 on this visual retrieval test. This is cross-frame landmark matching, not CityNav target localization,SR,SPL or a closed-loop RGB gain. Retrieval evaluation uses an explicit gallery only for that task; this gallery must never initialize an online Unseen episode with future human images.

Online integration will use only the currently rendered384 RGB/vertical-depth proxy and public named polygons, with pixel-keyed frozen features and episode-local history. Polygon registration remains weak supervision; source support and boundary robustness are quality indicators, not semantic segmentation truth. A zero-gated visual residual must preserve A3 initially, and a navigation contribution must be measured separately on the unified protocol before it is considered effective.
