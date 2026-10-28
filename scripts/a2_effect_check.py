"""Read-only post-calibration effect audit. No fitting, reselection or GPU."""
from pathlib import Path
from collections import Counter,defaultdict
import json,math
import torch
from experiments.a2_waldo_landmark_belief.calibration_fusion import calibrated_logits
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold,group_summary
from experiments.a2_waldo_landmark_belief.paths import sha,save

torch.set_num_threads(2)
b=Path('runs/a2_waldo_landmark_20261027');c=b/'calibration_collected'
candidate=json.loads((c/'candidate_calibration.json').read_text())
parameters=dict(candidate['selected_fit_parameters'],temperature=candidate['temperature'],enabled=True)
manifest=json.loads((b/'collected_snapshot/manifest.json').read_text())
expected={e['filename']:e['sha256'] for e in manifest['entries'] if e['split']=='calibration'}
pred_signature=sha(c/'predictions/signature.json')
rows=[];counts=Counter();map_counts=defaultdict(Counter)
for name,digest in expected.items():
 p=torch.load(c/'predictions'/name,map_location='cpu',weights_only=False)
 assert p['source_row_sha256']==digest and p['prediction_signature_sha256']==pred_signature
 base=p['spatial']['base_logits']
 off=calibrated_logits(p,candidate)
 assert torch.equal(base,off)
 counts['all_calibration_off_parity_views']+=1
 if calibration_fold(*p['group'])!='verify':continue
 mixed=calibrated_logits(p,parameters);grid=p['spatial']['grid_world'];target=p['target_xy']
 before=float((grid[base.argmax()]-target).norm());after=float((grid[mixed.argmax()]-target).norm())
 row=dict(group=p['group'],a0_correct=before<=20,a2_correct=after<=20,a0_ne=before,a2_ne=after,changed=not torch.equal(base,mixed))
 rows.append(row)
 values=dict(views=1,a0_correct=int(before<=20),a2_correct=int(after<=20),
             corrected=int(before>20 and after<=20),broken=int(before<=20 and after>20),
             changed_belief=int(row['changed']),changed_argmax=int(base.argmax()!=mixed.argmax()))
 counts.update(values);map_counts[p['group'][0]].update(values)
summary=group_summary(rows)
for key,value in candidate['verify_summary'].items():
 actual=summary[key]
 if isinstance(value,float):assert math.isclose(actual,value,rel_tol=1e-8,abs_tol=1e-10),(key,actual,value)
 else:assert actual==value,(key,actual,value)
summary['metric']='teacher-trajectory cached dense-grid argmax R@1 within20m; equal target-instance weights; not closed-loop SR'
report=dict(status='independent_cpu_effect_recheck_passed_no_gpu',calibration_seconds=candidate['seconds'],
            checkpoint_sha256=candidate['checkpoint_sha256'],evaluated_choice=parameters,choice_scope='Original fit-selected choice only; no refitting or alternative selection',
            deployed_gate=dict(enabled=candidate['enabled'],alpha=candidate['alpha']),
            counts=dict(counts),verify_summary=summary,by_map={m:dict(v) for m,v in map_counts.items()},
            calibrated_gate_passed=candidate['verify_gate']['passed'],checks=candidate['verify_gate']['checks'],
            provenance_erratum='candidate_calibration.verify_summary.metric contains obsolete A0-onpolicy wording. Actual frozen snapshot contains teacher-trajectory observations, not A0-on-policy. Numerical formulas unchanged; original outputs preserved.',
            conclusions=['No demonstrated held-out localization improvement for the fixed choice','Gate OFF reproduces base logits exactly in all4226 cached views','No full benchmark evaluation started; no navigation SR conclusion available'])
save(report,b/'effect_check.json');save(report,Path('reports/a2_execution_20261027/EFFECT_CHECK.json'))
print(json.dumps(report,ensure_ascii=False),flush=True)
