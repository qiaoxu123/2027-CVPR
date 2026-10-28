"""Post-hoc CPU diagnostics of a frozen model/config; never changes decisions."""
from collections import Counter,defaultdict
from pathlib import Path
import json,math
import torch
from experiments.a2_waldo_landmark_belief.paths import sha,save
from experiments.a2_waldo_landmark_belief.collected_snapshot import load_cache_row
from experiments.a2_waldo_landmark_belief.calibration_protocol import calibration_fold
from experiments.a2_waldo_landmark_belief.calibration_fusion import calibrated_logits

torch.set_num_threads(2)
b=Path('runs/a2_waldo_landmark_20261027');cache=b/'collected_snapshot';cal=b/'calibration_collected'
params=json.loads((cal/'candidate_calibration.json').read_text());choice=dict(params['selected_fit_parameters'],temperature=params['temperature'],enabled=True)
manifest=json.loads((cache/'manifest.json').read_text());signature=sha(cal/'predictions/signature.json')
c=Counter();negs=Counter();measurements=defaultdict(list);examples=defaultdict(list)
for entry in manifest['entries']:
 if entry['split']!='calibration':continue
 p=torch.load(cal/'predictions'/entry['filename'],map_location='cpu',weights_only=False)
 if calibration_fold(*p['group'])!='verify':continue
 assert p['prediction_signature_sha256']==signature and p['source_row_sha256']==entry['sha256']
 row=load_cache_row(cache,entry,'calibration');out=p['output'];s=p['spatial'];y=row['labels']
 grid=s['grid_world'];base=s['base_logits'];target=y['target_xy'];reliable=out['reliable'].bool()
 positive=(y['ranking']>0) & reliable;reachable=y['offset_valid'].bool() & reliable
 link=bool(row['public']['network']['link_valid'].any());visible=bool(row['private']['target_visible'])
 winner=int(out['scores'].argmax());trust=torch.sigmoid(out['candidate_match_logits']/params['temperature'])
 global_pass=bool(torch.sigmoid(out['match_logit']/params['temperature'])>=choice['threshold'])
 eligible=reliable & (trust>=choice['threshold']) & s['candidate_support'].any(-1)
 mixed=calibrated_logits(p,choice)
 a0_ne=float((grid[base.argmax()]-target).norm());a2_ne=float((grid[mixed.argmax()]-target).norm())
 failure=a0_ne>20
 c.update(dict(views=1,visible=int(visible),with_landmark=int(link),no_detections=int(not row['meta']['candidates']),
               with_reliable_candidate=int(reliable.any()),with_associated_positive=int((y['ranking']>0).any()),
               with_reliable_positive=int(positive.any()),with_reachable_positive=int(reachable.any()),
               global_gate_pass=int(global_pass),with_eligible_candidate=int(eligible.any()),
               changed_belief=int(not torch.equal(base,mixed)),changed_argmax=int(base.argmax()!=mixed.argmax()),
               a0_failure=int(failure),corrected=int(failure and a2_ne<=20),broken=int(not failure and a2_ne>20)))
 if positive.any():
  c['positive_views_winner_positive']+=int(positive[winner]);c['positive_views_global_gate_pass']+=int(global_pass)
  c['positive_views_winner_known_negative']+=int(p['known'][winner] and not positive[winner])
  c['positive_views_winner_outside_known_mask']+=int(not p['known'][winner])
  if failure:c['a0_failure_with_reliable_positive']+=1
 if reachable.any():
  c['reachable_views_winner_reachable']+=int(reachable[winner])
  measurements['positive_trust'].extend(trust[reachable].tolist())
  xy=s['candidate_xy'][reachable];refined=xy+out['offset'][reachable]
  measurements['positive_center_error_m'].extend((xy-target).norm(dim=-1).tolist())
  measurements['positive_refined_error_m'].extend((refined-target).norm(dim=-1).tolist())
  before_cells=torch.cdist(xy,grid).argmin(-1);after_cells=torch.cdist(refined,grid).argmin(-1)
  c['positive_box_count']+=int(reachable.sum());c['positive_offset_changes_nearest_cell']+=int((before_cells!=after_cells).sum())
 if p['known'].any():
  known=p['known'];positive_known=known & p['trusted_label'];negative_known=known & ~p['trusted_label']
  negs['positive_labels']+=int(positive_known.sum());negs['negative_labels']+=int(negative_known.sum())
  negs['positive_pass_threshold']+=int((positive_known & (trust>=choice['threshold'])).sum())
  negs['negative_pass_threshold']+=int((negative_known & (trust>=choice['threshold'])).sum())
  measurements['known_negative_trust'].extend(trust[negative_known].tolist())
 # Structural oracle bound: even a perfectly chosen trusted-positive box only
 # redistributes its ORIGINAL support mass; it cannot move outside peak mass.
 if positive.any() and failure:
  union=s['candidate_support'][positive].any(0)
  if union.any() and (~union).any():
   inside=base[union];outside=base[~union].max();log_mass=inside.logsumexp(0)
   best_possible=torch.logaddexp(inside.max()+math.log1p(-choice['alpha']),log_mass+math.log(choice['alpha']))
   c['failed_positive_structural_bound_population']+=1
   c['failed_positive_impossible_any_mass_preserving_alpha']+=int(log_mass<=outside)
   c['failed_positive_impossible_at_fixed_alpha']+=int(best_possible<=outside)
   if len(examples['mass_bound_failures'])<4 and log_mass<=outside:
    examples['mass_bound_failures'].append(dict(group=p['group'],episode=p['episode'],step=p['step'],a0_ne_m=a0_ne,
       positive_support_cells=int(union.sum()),log_positive_support_mass=float(log_mass),outside_max_logit=float(outside)))
 if global_pass and eligible.any():
  union=s['candidate_support'][eligible].any(0);inside=base[union];outside=base[~union].max() if (~union).any() else torch.tensor(-torch.inf)
  log_mass=inside.logsumexp(0);upper=torch.logaddexp(inside.max()+math.log1p(-choice['alpha']),log_mass+math.log(choice['alpha']))
  c['gated_support_bound_population']+=1;c['gated_single_cell_union']+=int(union.sum()==1)
  c['gated_outside_peak_cannot_be_beaten_at_fixed_alpha']+=int(upper<=outside)
  measurements['gated_support_probability_mass'].append(float((log_mass-base.logsumexp(0)).exp()))
  measurements['gated_support_cells'].append(int(union.sum()))
 measurements['grid_pitch_m'].append(float((grid[1]-grid[0]).norm()))
 if failure and len(examples['baseline_failure_candidates'])<4:
  examples['baseline_failure_candidates'].append(dict(group=p['group'],episode=p['episode'],step=p['step'],a0_ne_m=a0_ne,
    positive_available=bool(positive.any()),winner_positive=bool(positive[winner]),winner_known=bool(p['known'][winner]),
    global_gate_pass=global_pass,winner_trust=float(trust[winner]),candidates=len(row['meta']['candidates'])))
 if c['views']%400==0:print('Diagnosed',c['views'],flush=True)

def summarize(values):
 if not values:return None
 x=torch.tensor(values,dtype=torch.float64)
 return dict(n=len(values),mean=float(x.mean()),median=float(x.median()),p10=float(x.quantile(.1)),p90=float(x.quantile(.9)))
report=dict(status='posthoc_frozen_verify_cache_diagnosis_no_gpu',scope='137 heldout calibration targets, teacher trajectory observations only; exploratory causes, not navigation SR',
 checkpoint_sha256=params['checkpoint_sha256'],parameters=choice,counts=dict(c),trust_label_counts=dict(negs),
 measurements={k:summarize(v) for k,v in measurements.items()},examples=dict(examples),
 structural_bound='For positive-support union U, outside logits stay fixed and updated U mass equals original sum(U). If sum exp(base[U]) <= max exp(base[outside]), no mass-preserving local proposal can beat the outside peak, even with a perfect scorer and alpha=1. At fixed alpha, upper=(1-alpha)*max_inside+alpha*sum_inside.',
 no_refitting=True,no_new_parameter_selection=True,no_benchmark_data_used=True)
save(report,b/'no_gain_diagnosis.json');save(report,Path('reports/a2_execution_20261027/NO_GAIN_DIAGNOSIS.json'))
print(json.dumps(report,ensure_ascii=False),flush=True)
