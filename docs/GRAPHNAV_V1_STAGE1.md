# GraphNav-V1 Stage1

Current reporting convention: GraphNav-V1 is the sole main experiment. Historical
initialization checks and module controls remain archived; do not present A3 as a
separate main comparison model. The frozen complete-system Test-Unseen evaluation
uses `scripts.evaluate_closedloop --model graphnav-v1 --split test_unseen`, the
existing Seen-selected frozen manifest, and all 5281 refined test episodes.
It cannot subsample episodes, alter visual/history controls, or change weights.
`scripts.summarize_graphnav_v1_results` verifies all three full splits and writes
the GraphNav-V1-only `GRAPHNAV_V1_RESULTS.md` and JSON/CSV metrics after completion.

Stage1 reuses verified A3 Seed1 Epoch14, the original pretrained Trajectory/Stop,
and its original A3 controller reference. Required SHA256 values are enforced by
`graphnav/stage1_checkpoint.py`. Historical JointActive Epoch1 weights are not
used. Active observation, new GAT, and relation auxiliary losses are disabled.

The independent strong baseline lives in `runs/graphnav_v1_stage1/baseline_*`.
Seen128 has SR48.4375%, SPL47.4129893%, OSR56.25%. Its 128 trajectories,
candidate/mode decisions and Stop match the historical E1 Seed1 evaluation
exactly. Full Seen2470 B0 has SR48.5020%, SPL47.0162%, OSR56.4372%.

The complete initial model also reproduces B0's 128 trajectories, candidate/mode
and Stop probabilities exactly. These are compatibility results, not evidence
that visual training improves navigation.

## Training and delivery

`scripts.prepare_graphnav_stage1` prepares actual current frames and strictly
earlier expert-prefix observations using frozen Large/384 native patch features.
Depth supplies support validity, not a separate trained depth tower. ROI scales
reuse contextual patch tokens from the currently observed frame. Candidate
memory uses geographic grid alignment, quality weights, and pose deduplication.
Unknown observations cannot act as negatives. Private target/teacher fields are
accepted only by losses. The internal split holds out complete map+instruction
groups:83140 train states and4372 development states from87512 expert-prefix
states. No on-policy augmentation is used.

`scripts.train_graphnav_stage1` implements A3/Trajectory-frozen A3epochs,
visual/history B3epochs, and joint C5epochs initialized from prior stage Best.
C uses A3=1e-5, visual/history=1e-4, trajectory/stop=2e-5, 5% warmup, cosine
decay and gradient clipping. BERT/SigLIP2 remain frozen. Each epoch has complete
Train Dev and identical Seen128 validation. AdamW/scheduler/RNG and source/cache
SHA values are restored on resume; pending validation can resume independently.

`scripts.run_graphnav_stage1 --resume` preserves finished stages, resumes the
interrupted epoch from Last, selects complete B/C candidates using Seen only,
runs frozen full Seen/Unseen, and retains B0 as an independent release candidate.
`scripts.audit_graphnav_stage1` checks exact initial Seen128 parity and additional
A Real/Zero/Shuffle/No-Belief and B Current/History controls. Diagnostic controls
do not retune weights. Supplementary controls can overlap training; timings are
explicitly marked. Full final evaluations are serial.

`scripts.summarize_graphnav_stage1` requires all planned epochs, controls, and
full evaluations. It produces all requested CSVs, curves, source/input manifests,
parameters, paired episode/map bootstrap and a Chinese report. Checkpoints stay
under ignored `runs/`; compact reports are copied into `reports/graphnav_v1_stage1`.
Backbone weights are resolved by pinned model revisions in the offline cache;
the small Best/Last files contain all trainable modules and recovery state, not
duplicated frozen backbone tensors.

```bash
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
../2027-CVPR-static-geometry/.venv/bin/python -m scripts.run_graphnav_stage1 --resume
```

The task is complete only when `workflow_status.json` reports `complete` and
the full Chinese report and measured Seen/Unseen artifacts exist. Loss decreases,
nonzero gradients, architecture diagrams and a running queue are not acceptance.
