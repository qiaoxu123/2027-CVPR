# Core Pair Only: Original Residual vs. Joint RGB–Geometry (Seed 0)

**Code status:** committed. **Training status:** not started by this GitHub-only connection. No claimed GPU result.

## Scope (only one paired comparison)

- **A0:** Original formal GraphNav-V1 architecture: static geometric localization + separate bounded visual logit residual.
- **A1:** Same map/text/ROI and controller, but observed RGB enters candidate queries **before** language attention and landmark relation scoring. The independent visual logit residual is removed.
- Seed **0**, same original A3 Epoch14 and pretrained Trajectory/Stop initialization, same **A3/B3/C5** schedule, same existing complete Stage1 expert-prefix cache (~87,512 states), original 28×28 goal grid, frozen SigLIP2/BERT, same current/history memory and original 20-step guarded-blend/Stop protocol.
- **Run only A0 vs A1.** Zero, Shuffle, Current-only, multi-seed, WALDO local boxes, depth architecture changes, 384×384 goal-grid changes, Test-Unseen, and extra scenarios are NOT in this run.

## One command on the CUDA/data host

From the root of `2027-CVPR` checked out at `research/joint-rgb-geometry-retrain-20261027`, in the existing GraphNav environment, set original paths once:

```bash
export SBF_SOURCE=/absolute/path/to/verified/SBFNav/source
export SBF_CONFIG=/absolute/path/to/original/sbf_config.yaml
export CITYNAV_DATA=/absolute/path/to/sbfnav-data

python -m scripts.run_joint_rgb_geometry_core \
  --cache data/graphnav_stage1_frames \
  --sources runs/graphnav_v1_stage1/initial_sources.json \
  --sbf-source "$SBF_SOURCE" \
  --sbf-config "$SBF_CONFIG" \
  --data-root "$CITYNAV_DATA" \
  --output runs/joint_rgb_geometry_core_20261027 \
  --seed 0 --batch-size 16 --workers 4
```

Replace the three illustrative absolute paths with the **actual paths** recorded in the archived formal model's `runs/graphnav_v1_stage1/baseline_seen128/metrics.json`. Do not invent, copy or substitute a different dataset/config/checkpoint. Run on the host with the archived checkpoint and frozen encoder caches; the GitHub repo alone is insufficient.

The runner verifies cache completion and A3/head/reference source files, runs A and B stages (3 epochs each) plus C (5 epochs) for each arm, chooses between B-best and C-best by the **original Seen128 SR, SPL and internal-dev R@1 order**, and repeats the matched Seen128 closed-loop evaluation at the selected checkpoints. Training is sequential on the single GPU. Logs/status/checkpoints are isolated per arm and resumable; pre-existing outputs with a different protocol are rejected.

## Outputs

```
runs/joint_rgb_geometry_core_20261027/
  paired_protocol.json
  A0_original_residual/
    stages/A/{best.pt,last.pt,complete.json,...}
    stages/B/{best.pt,last.pt,complete.json,...}
    stages/C/{best.pt,last.pt,complete.json,...}
    seen128_B_candidate/metrics.json
    seen128_C_candidate/metrics.json
  A1_joint_rgb_geometry/
    stages/A/...
    stages/B/...
    stages/C/...
    seen128_B_candidate/metrics.json
    seen128_C_candidate/metrics.json
  core_pair_results.json
```

The result contains per-arm SR, SPL, OSR, NE, nDTW and sDTW (where present), paired episode manifest hash, selected stage and A1 minus A0 differences. It is **single-seed Seen128** development evidence only, **not** a held-out or full split paper result. Any later Val-Unseen should use a frozen selected checkpoint and a manifest created after Seen selection; no tuning by Unseen results.

## Safety/validation before the full run

```bash
python -m pytest -q tests/test_joint_rgb_geometry.py tests/test_stage1.py tests/test_full_token_rgb.py tests/test_stage1_final_evaluation.py
python -m compileall -q graphnav scripts
```

This runs only focused CPU tests (automated by the branch-only GitHub Actions workflow). They cannot replace the needed real CityNav CUDA smoke, memory profiling and full-training integrity checks.

The formal paper's archived checkpoint/results are a descriptive external reference only. The causal architecture comparison is **A0 vs A1 after matched retraining**, not A1 vs the original archived result, because stochastic optimization and model selection can differ.

## Local execution extension (2026-10-27)

The local runner now completes **both** seed-0 arms before final selection/freezing,
then evaluates every episode of Val-Seen (2470) and Val-Unseen (2697), in that order,
using the same frozen checkpoints. Seen128 remains development selection only.
No Test-Unseen or additional ablation is scheduled.

Server checkout: `/home/20261024_1/Workspace/.workspace/joint-rgb-geometry-retrain-20261027`.
Run `bash scripts/run_core_pair_local.sh`; `--smoke-only` executes only the CPU tests
and real CUDA A/B/C smoke for both architectures. The entrypoint uses the existing
`graphnav` Conda Python directly and takes an exclusive execution lock.

The migrated original frame cache is incomplete. The runner instead uses the
already fully reconstructed original source-data cache at
`GraphNav/data/rebuilt_stage1_20261025/frames` (87512 states, 21878 episodes), with
all chunk hashes checked on every stage entry. Reconstruction provenance and
exact source-weight checks are saved in `reports/core_pair_20261027/input_verification.json`.
Original A3/head/reference files retain the required SHA values. The preserved
epoch-zero snapshot (SHA `b044909700e1912a49d69551abe95b6fad22b3d0c0e8fd162d230ee6272c4f45`)
provides a strict recovery path when the historical head-cache manifest is absent;
its A3 and trajectory tensors must exactly equal the separately verified source
weights. Both arms receive all 112 identical common initial tensors. A1 adds only
its RGB query adapter. The regenerated frozen BERT cache is fingerprinted and
must match the original model revision/token protocol.

Each stage writes TensorBoard events, training.csv, history.json, console logs,
last/best checkpoints and complete.json. A single `frozen_pair.json` fixes both
selected checkpoint hashes and controls before any full evaluation starts.
Per-episode predictions preserve goal retrieval, public-query hash and source
identity for paired statistics. `scripts.summarize_core_pair` validates complete
counts, checkpoint hashes and paired manifests before producing results.json,
metrics.csv, by_map.csv and RESULTS.md. It reports paired episode bootstrap CIs,
exact SR McNemar, query-cluster SR bootstrap and query-level retrieval CIs.
All rates are fractions; NE is meters. A single seed does not measure
training-seed uncertainty. Large files remain in ignored server `runs/`.
