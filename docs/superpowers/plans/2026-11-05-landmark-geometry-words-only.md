# PR22 minimal name + geometric words experiment

1. Preserve unrelated untracked files; use PR22 head73ff64e, never main. Full Revised21878 source, public cityrefer_original objects only. Audit extracted spans/phrases and forbidden text; no GT reference lists.
2. Fix demonstrated invalid Polygon handling, phrase scope limitations and scrambled-control batching if needed with tests. No new network. Original instruction is extractor-only; no original pretrained Geometry/RGB.
3. CPU tests and real FP32 GPU Smoke. Frozen BERT name/phrase CLS caches only. All Cambridge14259 fit, Birmingham7619 final Dev, seed42,3epochs. Uniform fallback tasks retained. No Dev selection or Val/Test/navigation.
4. Same final model compare real phrases, name-only, deterministic global shuffled phrases. Save whole distribution/predictions/GT scoring, subgroup coverages and corrections/damages versus name-only. Document ablation rather than independently retrained name-only.
5. Save curves, 12+ genuine Belief heatmaps, extraction examples, Chinese report and resumable checkpoints server-side. Commit/push PR22 only, update PR summary, no merge.
