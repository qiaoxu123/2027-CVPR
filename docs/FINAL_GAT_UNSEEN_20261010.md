# Frozen GAT control: Seen benefit does not transfer

The full2697-episode Unseen GAT-only residual control is complete. It uses the same historical A3 and trajectory/Stop head as its A3-only comparator, the same protocol and zero visual evidence. The checkpoint/config were frozen before Unseen; results do not alter the queued models.

| Model | SR (%) | SPL (%) | OSR (%) | NE (m) | nDTW (%) | sDTW (%) |
|---|---:|---:|---:|---:|---:|---:|
| A3 + guarded trajectory/Stop | 23.3222 | 22.6725 | 32.3693 | 48.7174 | 26.2260 | 8.8368 |
| + GAT residual | 21.1717 | 20.5367 | 30.9974 | 51.1479 | 25.3990 | 7.9877 |

Paired episode-bootstrap95% intervals(2000 draws,seed20261010): SR-2.1505percentage points,CI[-3.1516,-1.1123]; SPL-2.1358points,CI[-3.1127,-1.1533]; OSR-1.3719points,CI[-2.4472,-0.4079]; NE+2.4305m,CI[1.8029,3.0888]. There are67 gained and125 lost successes. This is a negative generalization result despite the earlier Seen SR+1.2146point gain. A single Seen gain is insufficient evidence that GAT improves the strong geometry baseline.

All21490 movements and2697 episode traces pass step/horizon/bounds/nonworsening-start checks, with legal final altitude;1261 source-invalid starts remain. False-Stop Rate is78.8025%, with2119 failed policy Stops and265 visited-success-but-final-failed episodes. No physical swept-collision claim follows from the public action audit.

Exact paired episode/map uncertainty, all failures, original metrics, CSV and frozen checkpoint bindings are in `reports/final_gat_unseen_20261010`; action audit is in `reports/final_gat_constraints_unseen_20261010`. This report does not establish why GAT fails: edge construction, relation-only training objective, message propagation and map shift remain hypotheses. No Unseen-directed gate tuning or removal changes the frozen primary candidate.
