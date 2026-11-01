# Goal: AerialVG → demonstrable navigation improvement

Status: active. Completion requires verified paired closed-loop navigation improvement, not successful module integration or offline R@1 alone.

## Phases
1. [complete] Inspect official implementation/checkpoint and test full demo plus original modules.
2. [complete] Test official pair-token adapter on pooled ROI: no clear Dev/audit benefit; do not promote.
3. [complete] Extract actual object queries from causal CityNav observations; verify boxes/world association and feature discrimination.
4. [in_progress] Train region/context plus object-query visual evidence, isolating representation and relation effects under grouped splits. First matched native trials complete; further decisions depend on rollouts.
5. [in_progress] Integrate causal online inference/history, validate budgets/stop, run paired closed-loop pilots. Seen128 complete: no SR gain, OSR improves, SPL degrades. Train-only on-policy collection in progress; history remains to assess.
6. [in_progress] Full held-out navigation confirmation, multiple seeds where needed, confidence intervals and failures/latency; independent final confirmation after design freeze.

## Gates
- No target GT or future observations in forward, proposal selection or inference.
- Original official modules verified; HCA requires real spatial pyramid, not pooled physical ROI scales.
- Preserve original main checkpoints/results; distinguish every experimental configuration.
- Navigation success: meaningful paired SR gain with positive confidence-interval lower bound on a full predefined evaluation, supported by SPL/NE and held-out confirmation; report costs and robustness. A nominal gain or smoke pass is insufficient.
- Previously inspected audit is regression evidence, not untouched final confirmation. Freeze final protocol before evaluation; never tune on Test.

## Next action
Finish frozen fullValUnseen2697 baseline/candidate pair, verify and compare both original and previous calibrated residual. Include whole-map robustness and compute cost. No held-out retuning or premature goal completion; Test remains untouched by this candidate.

## Current execution
FullSeen2470 complete+verified: SR50.7287 vs original47.6113 (+3.1174pp, groupedCI[1.5815,4.7446], wholemapCI[1.4275,4.7561]); SPL47.0115 vs46.0028 (gain uncertain); NE28.2716 vs29.1192. Compared with prior calibrated residual SR48.7045, gain2.0243pp, but wholemapCI lower0. Frozen candidate selected for independent fullValUnseen2697. Detached coordinator751047 running original baseline child757617 before native candidate; actual progress must be revalidated from processes/files. Publication commitfcfde0b is pushed. Goal remains unachieved pending held-out and stronger-reference evidence.
