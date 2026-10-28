# A2 remote execution migration

The user requested preparing the new RTX5090 host and training A2 there, then requested maximum existing-data reuse and direct remote code retrieval. This steers execution location; the goal remains a real A2 improvement over the same frozen A0, with complete20+5 training and final Seen2470/Unseen2697 evaluation without Unseen selection.

## Preserved experiment

- Local training/confirmation waiters were verified live, with no training children and0/25epochs, then stopped explicitly for migration. Their state and the original untrained proofs/initial tensors are preserved under `revision_v3/remote_migration/`.
- A0/A1 completed results, all prior A2 revisions, data roles/query identities, source bytes, model architecture, seed0, optimizer schedule, frozen foundation weights/controller and28grid remain unchanged.
- The new host uses an isolated user-owned directory on the large data disk. A user-space path adapter preserves legacy absolute paths, allowing original manifests and data SHA checks to remain meaningful without rewriting dataset lineage. No system profile, unrelated environment or other user's service is modified.
- The committed GPU policy remains exclusive. Only the separate explicitly approved host policy may exempt the two existing speech services; all other processes, including another A2 phase, still block. Both policy code and actual policy JSON are included in training provenance.

## Transfer/reuse procedure

1. Clone/pull the specified A2 branch on the destination, using its already configured proxy only for commands that require it. Verify the resulting commit and source hashes. Frozen external source snapshots/private A0/WALDO files are transferred separately when missing.
2. Compare existing CityNav/CityRefer files and frozen public-model files by exact size+SHA. Reuse only identical files. Retrieve missing public encoder files at the exact revisions and verify original hashes.
3. Reuse exact-version CUDA libraries already installed on the destination; compare actual installed files and Python ABI before reuse. Transfer only the unmatched Python3.10-specific files, not a different-version Torch environment. Do not modify existing environments.
4. Transfer the complete original snapshot and required older-revision context/seals, preserving bytes and heldout roles. Compression/transport may change, source records may not. Final archive sizes and checksums are recorded only after pack completion. Retain partial transfers for recovery.
5. On the destination, archive old untrained proofs before rebuilding. Run complete CPU tests and compare the new initial-weight fingerprint to the preserved old seed0 weights; then genuine CUDA backward and the32-target A0/OFF/ON checks. New source/policy fingerprints require new proofs; old proofs cannot authorize training.
6. Execute full20+5 once, verify all epochs/curves, then independent calibration and real navigation in the original strict order. Source-bound final freezing/reporting remain mandatory.

## Current evidence / remaining work

494 local CPU regressions passed after the resource-policy feature; independent read-only review found no unresolved important/critical issue. The old429/456/485 proofs are historical after this versioned execution change. No remote CUDA proof or formal epoch is claimed yet.

Actual reuse already found seven exact dataset files (343,729,234bytes) and the original BERT files, including440,449,768-byte weights. An existing environment has matching CUDA12.8 library versions; its Python3.9 Torch extension cannot be substituted for Python3.10. File-level comparison/reconstruction is required. Public code/model retrieval and private-data transfers are in progress; Git/HTTP stalls must be assessed against actual processes before retries. No password is stored in scripts, reports or Git.

The original SSH relay is slow; a request for an optional direct route/shared store is pending. Continue safe reuse/transfer rather than treating silence as approval or restarting a live job. Final paired statistics/reporting and automatic navigation handoff also remain to be completed after migration.


## Resumable host preparation and launch protection

The host-side `scripts/a2_remote_bootstrap.py` waits for the existing exact archives, verifies their complete hashes, extracts only the current transfer plan, then checks all data, original frozen sources/weights and reconstructed compute files. It preserves old zero-epoch proofs and refuses to replace proofs if **any** formal training file is present, including an orphan epoch, protocol or history. Once a current launch has been certified, a restart retains matching proofs rather than resetting trained state.

A nonblocking `bootstrap.lock` covers preparation, service handoff and monitoring. A second supervisor cannot overwrite the first process's status. The existing scientific pipeline and GPU locks continue to enforce serial A2 phases. Missing `execution_target.json` leaves the host prepared without starting training; only an explicit `{"target":"remote"}` arms the remote handoff. The user's original remote-training authorization remains the selected deployment unless the user changes it; no parallel local training is armed.

Before launch, create `launch_manifest.json` **on the reviewed source machine**, after committing code. Bind the exact `git rev-parse HEAD`, branch, `source_inventory(project)` and SHA256 of all four `INPUT_MANIFESTS` imported from the bootstrap module. Transfer that exact manifest, then pass its independently recorded SHA with `--launch-sha256`. The destination checks the reviewed commit, tracked executable/test inventory, running bootstrap and input/policy manifest hashes before and after waiting, before checks and after recertification. An existing ready marker must bind the same launch. The actual machine policy is the only excluded default source config and is independently bound by `approved_gpu_policy.json` plus exact installed-policy equality.

Hardware checks use the trusted `/usr/bin/nvidia-smi`; a PATH wrapper cannot hide existing GPU work. An optional approved-host hash restricts background-service exceptions to that host. On a different host the identical policy acts as exclusive, retaining portable source fingerprints without granting exceptions there. The machine-specific policy stays outside the generic committed default.

Local verification: 518 CPU regressions pass, including missing-target, singleton ownership, orphan-checkpoint preservation, wrong Git/source/manifest and changed-source-after-check cases. Original seed0 initial tensors were rebuilt on CPU and exactly match fingerprint `dd47da218294c3f18fc6793c463db624c69650b8da91a220507fbfed48b9c2bf`. On the destination, all 11,976 original compute files (6,641,176,280 bytes) were reconstructed and individually checked. A real remote CUDA matrix backward check passed with original Torch2.8.0/CUDA12.8 on RTX5090; this is environment verification, **not** the real A2 CUDA/model gate or a formal epoch. Full dataset/snapshot transfers, model checks,20+5 training, independent confirmation and full benchmarks remain pending.


## Complete serial service handoff

The updated bootstrap additionally certifies and starts the navigation handoff supervisor. All three original scientific modules run through `scripts/a2_guest_service.py`, which commits their actual guest PID and finished-status SHA to the current systemd invocation/launcher identity. `A2_EXECUTION_HOST_ROOT` records only the approved task location and is passed explicitly to each service; it does not change model/data/optimizer/controller settings. The handoff independently waits for actual confirmation completion before using any GPU, then performs the fixed pilot/freeze/full-benchmark/report sequence. Missing/rejected evidence does not force enablement. Bootstrap cannot turn a stale terminal status into a full-evaluation claim: a final CPU-only verifier checks the real NAV service identity plus the current freeze and report.

At this preparation checkpoint, the entire missing-data archive was transferred, whole-file checked and extracted. The11GB artifact archive is still transferring. Environment-only CUDA backward passed earlier, but actual A2 CUDA/model gates, formal epochs and full benchmarks remain pending. The561 local CPU checks include45 independently reviewed handoff/bootstrap regressions. Source/policy launch rebinding is performed only after verifying the old owned preparation process is still waiting with no scientific children; live data senders are retained.
