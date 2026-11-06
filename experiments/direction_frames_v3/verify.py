"""Artifact and anti-leakage checks; never reads Val/Test or recovers labels from GT."""
from common import *
from infer import generate
from embed import encoder_text
from evaluate import candidate_weights,local_score,uniform_orientation_score
from PIL import Image
import re

def run():
 out=BASE/'results';checks=[]
 def check(name,condition):
  assert condition,name
  checks.append(name)
 D=readj(out/'dataset.json');X=readlines(out/'public_inputs.jsonl.gz');P=readlines(out/'inference_predictions.jsonl.gz');E=readlines(out/'evaluation_gt.jsonl.gz');R=readlines(out/'relations.jsonl.gz');Q=readlines(out/'evaluation_predictions.jsonl.gz');byX={r['occurrence_id']:r for r in X};byP={r['occurrence_id']:r for r in P};byE={r['occurrence_id']:r for r in E}
 check('TRAIN and object source SHA256 unchanged',all(sha(r['path'])==r['sha256'] for r in D['source_files']))
 check('Only TRAIN source declared; 22002 episodes',D['train_episodes']==22002 and 'train' in D['source_files'][0]['path'].lower())
 check('Full named-landmark candidate ledger and output population',len(R)==4659 and len(readlines(out/'reference_frame_outputs.jsonl.gz'))==4659)
 old=readlines(V2/'results/geometry_pairs.jsonl.gz');check('All 1042 v2 bindings preserved',len(old)==1042 and all(o['occurrence_id'] in byE for o in old))
 check('v2 coordinates, named references, scope confidence and geometry unchanged',all(byE[o['occurrence_id']]['reference_ids']==o['reference_ids'] and byX[o['occurrence_id']]['parse_tier']==o['confidence_tier'] and byE[o['occurrence_id']]['geometry_status']==o['geometry_status'] and np.allclose(byE[o['occurrence_id']]['subject_xy'],o['target_xy']) and np.allclose(byE[o['occurrence_id']]['reference_xy'],o['reference_xy']) for o in old))
 check('423/250 old valid pairs preserved',sum(o['geometry_status']=='disjoint' for o in old)==423 and sum(o['geometry_status']=='disjoint' and o['confidence_tier']=='high' for o in old)==250)
 check('569/317 expanded disjoint pairs',sum(e['geometry_status']=='disjoint' for e in E)==569 and sum(e['geometry_status']=='disjoint' and e['parse_tier']=='high' for e in E)==317)
 check('Subject geometry and future trajectories absent from public inputs',all(not(set(x)&{'subject_xy','target_xy','target_position','target_polygon','GT','future_trajectory','trajectory','yaw'}) and not(set(x['subject_ids'])&set(x['scene'])) for x in X))
 ledger={r['occurrence_id']:r for r in R};check('Navigation target also absent from every public scene',all(ledger[x['occurrence_id']]['target_id'] not in x['scene'] for x in X))
 check('Every saved inference prediction exactly reproduced without GT',all(generate(x)==p for x,p in zip(X,P)))
 check('All candidate directions explicitly marked no GT selection',all(not c['GT_used_to_construct_or_select'] for p in P for c in p['candidates']))
 new=[x for x in X if x['origin']=='typed_road_extension'];check('All 307 additions independently typed TrafficRoad references',len(new)==307 and all(all(x['scene'][i]['category']=='TrafficRoad' for i in x['reference_ids']) for x in new))
 O=readj(D['source_files'][1]['path']);check('Typed-road additions retain all matching named road segments',all(set(x['reference_ids'])=={i for i,o in O[x['map']].items() if name_key(o['name'])==name_key(x['scene'][x['reference_ids'][0]]['name']) and o['object_type']=='TrafficRoad'} for x in new))
 B=np.load(out/'bert_features.npz');T=readlines(out/'encoder_texts.jsonl.gz');check('Frozen 1349x768 BERT features aligned with public text',list(B['ids'])==list(byX) and B['features'].shape==(1349,768) and all(t['encoder_text']==encoder_text(x) for t,x in zip(T,X)))
 bm=readj(out/'bert_feature_manifest.json');check('BERT frozen and public-only provenance',bm['frozen'] and not bm['fine_tuned'] and bm['public_input_sha256']==sha(out/'public_inputs.jsonl.gz'))
 folds=readlines(out/'folds.jsonl.gz');core=[x for x in X if x['word'] in MAIN]
 for f in folds:
  te=[x for x in core if (str(fold_id(x['cluster'])) if f['scheme']=='object_5fold' else x['map'])==f['fold']];raw={text_key(x['instruction']) for x in te};tr=[x for x in core if x not in te and text_key(x['instruction']) not in raw]
  assert not(set(f['train_clusters'])&set(f['test_clusters']))
  assert len(te)==f['test_n'] and len(tr)==f['train_n']
  assert {x['cluster'] for x in te}==set(f['test_clusters']) and {x['cluster'] for x in tr}==set(f['train_clusters'])
  if f['scheme']=='leave_map_out':assert all(x['map']!=f['fold'] for x in tr)
 check('All object/map folds disjoint; exact English duplicates purged',True)
 for r in Q:
  p=byP[r['occurrence_id']];e=byE[r['occurrence_id']];w=r['weights'];metric='roadside' if r['metric']=='signed_roadside_halfplane' else 'sector'
  if r['uniform_orientation_control'] and r['local_score'] is not None:
   assert np.isclose(r['local_score'],uniform_orientation_score(e,metric))
  elif w:
   assert w==candidate_weights(p,list(w)) and np.isclose(sum(w.values()),1)
   s,err,_=local_score(p,w,e,metric);assert np.isclose(s,r['local_score']) and np.isclose(err,r['local_error'])
  if r['local_score'] is None:assert r['hybrid_score']==r['baseline_score']
  else:assert r['hybrid_score']==r['local_score']
 check('Every evaluated candidate score is fixed-weight average, never best-GT choice',True)
 grouped=collections.defaultdict(list)
 for r in Q:grouped[r['scheme'],r['cohort'],r['model']].append(r)
 for s in readj(out/'summary.json'):
  rr=grouped[s['scheme'],s['cohort'],s['model']];covered=[r for r in rr if r['local_score'] is not None];vals=[r['local_score'] for r in covered] if s['mode']=='conditional_local' else [r['hybrid_score'] for r in rr]
  assert s['n']==len(rr) and s['covered']==len(covered)
  if vals:assert np.isclose(s['consistency'],np.mean(vals))
  assert np.isclose(s['net_improvement_all'],np.mean([r['hybrid_score']-r['baseline_score'] for r in rr]))
 check('All summary denominators, coverage, compatibility and paired net differences recomputed',True)
 check('All randomization controls retain support and 300 replicates',all(n['coverage_preserved'] and n['B']==300 for n in readj(out/'negative_controls.json')))
 check('Known boathouse excluded sensitivity cohort has 424 observations',len(grouped['object_5fold','expanded_excluding_known_boat','strict_rules'])==424)
 figs=list((BASE/'figures').glob('*.png'));check('15 real-statistics/map PNG and matching PDFs',len(figs)==15 and all(f.with_suffix('.pdf').exists() for f in figs))
 check('Every PNG is high resolution 320dpi',all(min(Image.open(f).info.get('dpi',(0,0)))>=319 for f in figs))
 report=(BASE/'REPORT.md').read_text();check('All report local image and document links exist',all((BASE/t).exists() for t in re.findall(r'\]\(([^)]+)\)',report) if not t.startswith('http')))
 manifest=readj(out/'evaluation_manifest.json');check('Evaluation refers to exact frozen public inputs, predictions and features',manifest['public_input_sha256']==sha(out/'public_inputs.jsonl.gz') and manifest['frozen_inference_sha256']==sha(out/'inference_predictions.jsonl.gz') and manifest['BERT_features_sha256']==sha(out/'bert_features.npz'))
 writej(out/'verification.json',{'status':'passed','checks':checks,'n_checks':len(checks),'evaluation_rows_checked':len(Q),'limitations':'Local reproducibility/invariant checks, not an independent review or proof of annotator reference frames','report_sha256':sha(BASE/'REPORT.md')})
 print(f'PASS: {len(checks)} checks, {len(Q)} evaluated rows; all 1349 frame predictions independently reproduced')
if __name__=='__main__':run()
