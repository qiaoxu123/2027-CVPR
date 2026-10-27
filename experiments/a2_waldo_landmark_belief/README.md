# A2 WALDO Box <-> Language Landmark Relations

A2 keeps the archived formal GraphNav-V1 (A0) goal belief and controller frozen.

This directory currently contains **standalone public-input primitives**, **not a fully connected runtime or trainable checkpoint**:

- `mentions.py`: match uniquely linked, explicitly mentioned CityRefer landmarks to frozen BERT instruction-token offsets. Ambiguous/unmatched references are masked.
- `core.py / InstructionLandmarkEncoder`: separately encode each landmark's mention tokens, its named BERT tokens and full-instruction contextual role; BERT backbone itself stays frozen.
- `core.py / box_landmark_geometry`: explicit every-Box × every-Landmark world-XY relative geometry; optionally masked observed height. Projection and coordinate-frame correctness must be verified.
- `core.py / BoxLandmarkRelationScorer`: language-conditioned pair scoring and candidate ranking, masked for unavailable evidence.
- `core.py / conservative_belief_mix`: opt-in calibrated redistribution of **A0 dynamic** probability only within validated observed candidate support; exact fallback when alpha=0 or unsupported.

Run CPU tests:

```bash
python -m pytest -q tests/test_a2_landmark_relations.py tests/test_a2_landmark_mentions.py
```

Remaining integration milestones: offline WALDO detection and current-view image-to-world projection; unique public CityRefer linkage and BERT offset extraction; SigLIP2 candidate crops; object-level training/labels with trusted positives and hard negatives; train-only dev/calibration; A0 policy hook; paired Seen/Unseen navigation. Do **not** start purported full A2 training before these steps and input provenance tests are complete.

Full protocol: [docs/A2_WALDO_LANDMARK_RELATION_BELIEF_20261027.md](../../docs/A2_WALDO_LANDMARK_RELATION_BELIEF_20261027.md).
