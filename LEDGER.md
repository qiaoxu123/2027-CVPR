# Execution ledger

- 2026-10-27: User approved design implementation. Independent directory protects original experiments; native inline execution.
- Resource probe: generic detector cannot parse installed nvidia-smi shim; direct PyTorch confirms CUDA, RTX PRO 6000, ~94 GiB free. CPU 24 threads, RAM ~118 GiB available, disk ~996 GiB free. Use PyTorch as authoritative GPU resource probe.
- Task 1: starting RED tests for geometry/labels/fusion and neural invariants.
- Tasks1/2: geometry, unknown-label ranking masks, local mass-conserving fusion and 4,621,548-parameter relation network implemented; empty/permutation/depth-fill/distance/language tests GREEN.
- Task3: first18 real views extracted; real forward/backward and 4-update smoke completed. Two full zero-adapter trajectories match original and archived paper; one-controller forced-routing check passed. These are implementation checks, not efficacy evidence.
- Ruling: batch8 uses observation-wise gradient accumulation rather than padded multi-observation attention — variable candidate/reference counts remain exact; no cross-observation normalization — cost: some batching throughput.
- Independent whole-branch review:3 material provenance issues. RED tests reproduced partial-resume memory omission, unverified WALDO/annotation hashes, checkpoint/new-label-cache mismatch. Fixed causal replay, input hashes, calibration cache binding and eval annotation/map checks.
- Model invalid-height-fill regression RED→GREEN. Visualization renderer-constructor and NumPy export integration failures corrected; PNG+JSON export completed.
- User additionally requested full remote sync. Remote qiaoxu123/2027-CVPR, planned independent branch research/local-semantic-geometry-refinement-20261027; original GraphNav dirty worktree is untouched.
- Ruling: remote bundle includes current feature cache, final verification weights, runtime source and original/WALDO weights; external raw RGBD/CityNav datasets and generic HF caches remain external — required for observation regeneration/navigation; cached-only training does not open raw annotations.
- Archived pre-fix cache in data_archive/before_integrity_fix; rebuilding all860 views with final extractor/provenance and row content hashes. Old smoke artifacts intentionally stale and retained locally; only final matching artifacts will be published.
- Actual adversarial-review plugin invocation failed with workspace routing discovery timeout; independent reviewer findings above are separate and not misrepresented as plugin output.
