# CityNav parser comparison implementation plan

Goal: run A(sm+current rules), B(trf+same rules), C(GLiNER2), D(local Qwen3-14B) on 300 targeted + independent random TRAIN language samples, with no target metadata in parser input.
Architecture: separate public parser inputs and private sampling/diagnostic metadata. All outputs use exact-text spans and directed triples; derived chains preserve relation subject/object. Independent human graph labels are absent, so precision/recall/F1 remain unavailable until blind human adjudication.
Tech: existing Python/torch, isolated parser dependency environments, local HuggingFace weights, HTML+JSON review/export.

- [ ] Protect current files; independent worktree/branch, fetch current remote main and prior reports.
- [ ] Seed42 sample 100 large-error/context-absent,100 large-error/context-present,100 correct-direction queries; deduplicate query and record selected-anchor strata. Add100 disjoint random full-TRAIN queries with population coverage caveat.
- [ ] Public records contain only index/text/public name catalog; no map positions/GT/old parses. Models receive the same name list. Sampling outcomes remain private.
- [ ] Write failing tests for exact spans/directed chains, blind inputs, review gating, error resume; implement shared JSON schema and adapters.
- [ ] Install isolated model dependencies, pin versions/revisions; 2-example smoke, then persistent resumable full inference, GPU synchronized timing and peak allocated VRAM, separate model loading.
- [ ] Provide blank blind human graph editor first, candidate outputs hidden until requested; store reviewer/source/version, explicit reviewed status, gold export. Never automatically promote model consensus or Agent judgments into independent gold.
- [ ] Produce descriptive output coverage/disagreement, independently marked official-name lexical audit, case reviews and error-associated hypotheses. No true parsing accuracy or causal percentage without human labels.
- [ ] Full tests, artifact inspection, report limitations, commit/push; no direction training/RGB/navigation/Val/Test.

Potential blockers: downloads/dependency incompatibility; remedy with local isolated venv and official model repos. Missing human gold is not permission to invent accuracy. Record any model that cannot actually execute instead of substituting another size/model silently.

Engineering smoke decisions before final comparison: preserve conflicting GLiNER role predictions (do not silently drop landmark when the model also labels it target); exact unique-literal alignment allowed but recorded; unknown type fields use a shared lexical span heuristic, never GT categories. A/B keep identical rules and public gazetteer. C uses the same catalog through literal-name schema conditioning; D receives complete names. GLiNER threshold .5, full20 relation list fixed. D non-thinking deterministic BF16, compact JSON, max1600 new tokens, batch8. Preliminary smoke outputs are retained separately and are not final comparison data.
