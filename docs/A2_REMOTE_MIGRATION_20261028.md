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
