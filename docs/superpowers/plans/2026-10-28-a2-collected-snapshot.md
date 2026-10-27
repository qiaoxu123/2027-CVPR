# A2 user-requested collected-data training

User explicitly stops further collection and authorizes training on completed data.

1. Stop CPU queue before GPU cache, preserve raw files/progress/logs. Never mark the original full cache complete.
2. Freeze only episode completion records; exclude interrupted episode rows. Derive an immutable hard-linked snapshot. Secondary seed0 target-group SHA split 80/10/10 across collected original TRAIN targets; no original benchmark or calibration inputs.
3. Keep raw extraction provenance. At load time mask labels for candidate identities assigned to other derived partitions. Retain all positive observations and unknown masks; no supervised heldout negatives. Calibration is teacher-view only and cannot establish on-policy safety.
4. Test partition/negative isolation, incomplete episode exclusion and tamper rejection. Run whole CPU suite, independent adversarial review and real updated-model CUDA backward.
5. Train full20+5 on the frozen snapshot, dev-only selection, preserve checkpoints/TensorBoard. Default gate OFF pending independent calibration and actual closed-loop checks. Commit/push code and audit summary; keep raw data on server.
