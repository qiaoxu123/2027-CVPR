# Goal: AerialVG → demonstrable navigation improvement

Status: active. Completion requires verified paired closed-loop navigation improvement, not successful module integration or offline R@1 alone.

## Phases
1. [complete] Inspect official implementation/checkpoint and test full demo plus original modules.
2. [complete] Test official pair-token adapter on pooled ROI: no clear Dev/audit benefit; do not promote.
3. [complete] Extract actual object queries from causal CityNav observations; verify boxes/world association and feature discrimination.
4. [in_progress] Train region/context plus object-query visual evidence, isolating representation and relation effects under grouped splits. First matched native trials complete; further decisions depend on rollouts.
5. [in_progress] Integrate causal online inference/history, validate budgets/stop, run paired closed-loop pilots. Seen128 complete: no SR gain, OSR improves, SPL degrades. Train-only on-policy collection in progress; history remains to assess.
6. [pending] Full held-out navigation confirmation, multiple seeds where needed, confidence intervals and failures/latency; independent final confirmation after design freeze.

## Gates
- No target GT or future observations in forward, proposal selection or inference.
- Original official modules verified; HCA requires real spatial pyramid, not pooled physical ROI scales.
- Preserve original main checkpoints/results; distinguish every experimental configuration.
- Navigation success: meaningful paired SR gain with positive confidence-interval lower bound on a full predefined evaluation, supported by SPL/NE and held-out confirmation; report costs and robustness. A nominal gain or smoke pass is insufficient.
- Previously inspected audit is regression evidence, not untouched final confirmation. Freeze final protocol before evaluation; never tune on Test.

## Next action
Finish frozen window8 fullSeen2470; verify complete manifest/trace/protocol and paired SR/SPL/NE with uncertainty. Only then decide held-out frozen validation or further training. Do not tune on interim fullSeen results, deploy the under-supported arrival probe, or relax the arrival guard.

## Current execution
Both Seen128 pilots complete. On-policy SR49.22%,SPL41.73% (not adopted). Window8 SR52.34%,SPL49.40% vs original50%,49.14%, but SR CI[-4.69,+9.38]pp spans zero. Old full run544395/audit548331 stopped at2047/2470, cause undetermined; old exec13191 is missing. Frozen native resume validated and exact2+2 episode smoke passed. Detached parent749619/child749620 live; append423 remaining episodes, then paired summary and full verification automatically. Monitor JSON contains current timestamp/count. Original2047 prefix preserved with SHA and backup. Goal remains unachieved pending full and held-out evidence.
