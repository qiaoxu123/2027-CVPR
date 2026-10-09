# Shared closed-loop evaluation protocol

This protocol preserves the existing local SBFNav benchmark. It does not substitute offline localization for navigation. GraphNav development branch: `2027-CVPR/graphnav-closedloop-sbf-parity` from `11e0409b322f08db4459888c6d38ba08547da730`.

## Inputs and checkpoint

Use unchanged Revised CityNav under `/home/rental/20260922_1/Workspace/DATA/sbfnav-data`. Raw source episodes: 2470 Val Seen, 2697 Val Unseen. The complete source sequence/order is retained; no filtering by landmark availability, target visibility or model success. Static query deduplication applies only to separate localization reports.

SBFNav is a local paper-based reproduction, not the authors' official software. Selected existing checkpoint: `SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/training/checkpoints/best_val_unseen.pt`, epoch14, SHA256 `e991b4e46ebc20b9cf7a1e350a0993c75028e863b10b55df166acd4e59510885`. Frozen source/config snapshot lives in the same run's `source/`. Current SBF worktree has unrelated existing changes and is not used as the model/renderer source. Historical checkpoint selection used Val Unseen128 proxy; this limitation is disclosed. No new checkpoint selection uses Unseen results.

Both receive the same immutable source objects and supplied CityRefer reference names. SBF retains the largest homonymous polygon; verified A3 retains matching source fragments up to16 nodes. These native preprocessing differences are preserved and explicitly audited rather than silently dropping A3 road regions. IDs are preprocessing/metric metadata; the policy receives map reference names/polygons, instruction and observed state. The policy never receives the label record, true destination, annotation IDs or expert future trajectory. SBF oracle target injection is not called. PublicCatalog allows the scene builder to operate without a label-bearing record.

## Observations and dynamics

Both policies use a single shared rollout, source rasters, start poses and world bounds. SBF state is north-up9 channels, containing RGB/height only inside visited camera footprints, reference masks/contours and visited pose history. Footprint half-width is altitude above known map ground and yaw rotates the square footprint. This is the existing orthographic simulator proxy, not a photorealistic camera or full3D occlusion simulator. Source PNG/TIF dimensions and georeferencing are audited. GraphNav x/y normalization uses East/North rectangular bounds; SBF's internal square canvas uses image row/column with inverted y. Conversions return common world coordinates; normalized coordinates cannot be reused across these models.

Max20 decisions; max50m horizontal and10m vertical displacement per decision; waypoint clipped to public XY map bounds and ground+10..100m altitude. SBF ends at horizon or5 stagnating movements <=1e-4m. A learned policy may Stop before movement; the evaluator uses its actual final position. No target-distance Stop or oracle waypoint correction. The existing controller is a macro-action simulator without swept-volume collision checking. Terrain collision/unsafe initial-state diagnostics must be disclosed; no extra safety advantage is claimed for either method.

Native384 RGB-D is available as an additional current-view renderer using the same source/footprint. Its depth is vertical camera-minus-surface height, with invalid source pixels masked. It is not perspective ray depth. The SBF baseline's learned preprocessing remains its native224 state/crops, while visual module inputs follow their declared native resolutions. An unobserved region is unknown and cannot provide negative evidence.

## Metrics

Benchmark XY Euclidean NE; SR=final distance<=20m; OSR=minimum visited distance<=20m; SPL=SR×straight-line initial-to-goal distance / max(executed XY path length, straight-line distance,1e-9). For zero initial goal distance, this retains the local benchmark's SPL convention. nDTW=exp(-DTW(executed XY path, entire source human XY path)/(20m×reference point count)); sDTW=SR×nDTW. SR/OSR/SPL/nDTW/sDTW are JSON fractions and displayed as percentages. NE is meters.

Mean steps count actual executed movements; a Stop decision adds no movement. Mean policy inference latency synchronizes CUDA and includes uncached policy preprocessing, excluding environment state construction and metric calculation. False-Stop Rate=failed policy Stop decisions / all policy Stop decisions; it is null if no policy Stop exists. Also report false stops / all episodes. Stagnation/horizon are not silently classified as learned Stop.

128 quick episodes are evenly spaced over the full source split using the frozen reproduction's deterministic selector. Model comparisons must share the exact episode manifest SHA256. Full rows use every source episode. Map/episode paired comparisons and uncertainty will be generated for final frozen candidates; train/internal dev and Val Seen determine new model/hyperparameter selection.

## Verification and measured status

Shared evaluator: `scripts/evaluate_closedloop.py`; GT-free interface/control: `graphnav/closedloop.py`. Tests cover label isolation, Stop before movement, XY boundary success, path inefficiency, OSR/SR distinction and no-Stop denominator. SBF Seen128 replay exactly matches all128 historical pose arrays and path lengths (maximum absolute discrepancy0).

Fresh full baseline completed and matched every historical trajectory exactly:2470 Seen and2697 Unseen. Metrics below use percentages except NE, steps and latency.

| Split | SR | SPL | OSR | NE(m) | nDTW | sDTW | Steps | Policy ms | Eval minutes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Seen |43.52|36.12|60.40|31.52|28.28|14.74|15.77|47.97|32.27|
| Unseen |21.51|15.89|38.75|47.57|23.51|7.59|16.65|47.76|36.96|

GPU allocated peaks:1.028GiB Seen/1.023GiB Unseen. This checkpoint never emits a policy Stop, so False-Stop Rate is null; stagnation/horizon termination cannot be reported as learned Stop calibration. Episode-level results remain in `runs/closedloop_baseline/`; report copies and exact replay checks are in `reports/sbf_protocol/`.

The fresh Level2 Unseen minimums are SR20.4301075% and SPL14.3016446%; Level3 requires SR21.5053763% and SPL15.8907162%. Level1 and Seen/Stop/safety checks are additional conditions; no GraphNav acceptance level is claimed. These targets derive from this local reproduction, not paper results. The legacy SBF checkpoint had historical Val Unseen proxy selection, explicitly disclosed; new GraphNav selection uses internal training dev/Seen.

Seen128 A3 bounded Top1 smoke:SR38.28%,SPL37.61%,OSR44.53%,NE29.76m;128 identical public episodes, no trained trajectory/Stop or RGB/history. Versus SBF Seen128, SR−6.25pp,SPL+0.72pp; this is preliminary Seen evidence, not an Unseen gain or completed system.

Original checkpoints, source inputs and previous results remain unchanged. The initial full Seen wrapper wrote current Git at report finish; its loaded wrapper90b302f SHA is recorded separately in `historical_replay_full_seen.json`. Subsequent launches record startup code/diff/hash. All full Seen and Unseen paths have maximum absolute historical discrepancy0.

## Reproduction

From the new worktree, using existing SBFNav Python for the baseline:

```bash
PYTHONPATH=. HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
TOKENIZERS_PARALLELISM=false \
../SBFNav-CityNav-Reproduction/.venv/bin/python scripts/evaluate_closedloop.py \
  --sbf-source ../SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/source \
  --config ../SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/source/configs/sbfnav/main.yaml \
  --checkpoint ../SBFNav-CityNav-Reproduction/runs/sbfnav_main_e20_bs16_s0_20260929/training/checkpoints/best_val_unseen.pt \
  --data-root ../DATA/sbfnav-data --split val_seen \
  --output runs/reproduce_sbf_val_seen
```

Use `--split val_unseen` and a new output for the full Unseen replay; add `--max-episodes 128` for the quick set. Existing output directories are never overwritten. Check `reports/sbf_protocol/phase1_plan.json` for exact next-stage commands and predeclared selection rules.

The full input audit found SBF-vs-static binding-set differences in1470/2470 Seen and1208/2697 Unseen episodes, commonly multiple true road fragments rather than wrong object identity. Initial altitude exceeds ground+100m in1020/2470 Seen and1261/2697 Unseen episodes; starting poses are preserved and controller descent is gradual.50 Seen and2 Unseen supplied reference polygons are invalid. These episodes remain in the denominator and require robust model-side quality handling.
