"""Reproduce public inference, anti-leakage invariants and all unique-field checks."""
from shared import *
from frames import derive
from evaluate import field_geometry
from PIL import Image
import ast

def run():
 checks=[]
 def check(name,b):assert b,name;checks.append(name)
 D=readj(OUT/'dataset.json');F=readj(OUT/'freeze_manifest.json');GM=readj(OUT/'GT_phase_manifest.json');EM=readj(OUT/'evaluation_manifest.json');X=readlines(OUT/'public_inputs.jsonl.gz');P=readlines(OUT/'frozen_frames.jsonl.gz');E=readlines(OUT/'evaluation_gt.jsonl.gz');R=readlines(OUT/'relations.jsonl.gz');Q=readlines(OUT/'geometry_checks.jsonl.gz');fields=readlines(OUT/'road_fields.jsonl.gz');bx={r['occurrence_id']:r for r in X};bp={r['occurrence_id']:r for r in P};be={r['occurrence_id']:r for r in E};br={r['occurrence_id']:r for r in R};bf={r['key']:r for r in fields}
 check('TRAIN and objects source SHA unchanged',all(sha(z['path'])==z['sha256'] for z in D['source_files']))
 check('Full TRAIN 22002/24 maps scanned, no Val/Test source declared',D['train_episodes']==22002 and D['train_maps']==24 and all('val' not in z['path'].lower() and 'test' not in z['path'].lower() for z in D['source_files']))
 old=readlines(V3/'results/relations.jsonl.gz');oldx=readlines(V3/'results/public_inputs.jsonl.gz');check('All v3 ledger IDs preserved; 42 new relation IDs',set(r['occurrence_id'] for r in old)<=set(br) and len(R)==4701 and len(R)-len(old)==42)
 check('All 1349 original bound public inputs unchanged',all(bx[x['occurrence_id']]==x for x in oldx))
 check('Original 425 core-disjoint support preserved',sum(e['geometry_status']=='disjoint' and e['word'] in MAIN and not e['origin'].startswith('v4') for e in E)==425)
 check('Every candidate gets frame and geometry output',len(P)==len(Q)==len(R)==len(bp)==len(br))
 check('GT materialization population exactly matches public bound input IDs',set(be)==set(bx) and len(E)==1398)
 O=readj(DATA/'cityrefer_original/objects.json');new=[r for r in R if r['origin'].startswith('v4')];check('All49 new bindings aligned to original object annotation text',len(new)==49 and all(text_key(O[r['map']][r['target_id']]['descriptions'][r['ann_id']])==text_key(r['instruction']) for r in new))
 check('Subject and navigation target geometry absent from all public scenes',all(not(set(x['subject_ids'])&set(x['scene'])) and br[x['occurrence_id']]['target_id'] not in x['scene'] and not(set(x)&{'subject_xy','target_xy','target_positions','target_polygon','GT','trajectory','future_trajectory','yaw'}) for x in X))
 check('Freeze code and data SHA match current sources',all(sha(BASE4/f)==h for f,h in F['code_sha256'].items()) and sha(OUT/'public_inputs.jsonl.gz')==F['inputs_sha256'] and sha(OUT/'frozen_frames.jsonl.gz')==F['frame_predictions_sha256'] and sha(OUT/'road_fields.jsonl.gz')==F['road_fields_sha256'])
 check('GT and evaluator refer to the same frozen predictions',GM['frozen_predictions_sha256_before_GT']==EM['frozen_prediction_sha256']==F['frame_predictions_sha256'] and GM['GT_sha256']==EM['GT_sha256']==sha(OUT/'evaluation_gt.jsonl.gz'))
 check('Evaluator code/check SHA match',EM['code_sha256']==sha(BASE4/'evaluate.py') and EM['checks_sha256']==sha(OUT/'geometry_checks.jsonl.gz'))
 for filename in ['frames.py','roads.py','road_language.py']:
  tree=ast.parse((BASE4/filename).read_text());keys=[n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)]
  assert not {'subject_xy','target_xy','target_positions','target_polygon','trajectory','future_trajectory','yaw'}&set(keys)
 check('Pure frame/road/language modules never request private GT or trajectory keys',True)
 cache={}
 for x in X:
  p=derive(x,cache);saved=bp[x['occurrence_id']];assert all(saved[k]==v for k,v in p.items()),x['occurrence_id']
 check('All 1398 public-only frame predictions recomputed exactly, road fields rebuilt',True)
 check('All 37 road field definitions reproduced without GT',cache=={k:{a:b for a,b in v.items() if a!='key'} for k,v in bf.items()})
 check('Every unique road field passes independent multi-resolution geometry gate',all(f['state']!='unique_field' or (f['endpoints']==2 and f['branch_nodes']==0 and f['component_fraction']>=.95 and len(f['geometry_sensitivity'])==2 and all(z['state']=='unique_field' and z['heading_error_p90_deg']<=15 for z in f['geometry_sensitivity'])) for f in fields))
 for q in Q:
  if q['state']=='unique_field':
   p=bp[q['occurrence_id']];g=be[q['occurrence_id']];z=field_geometry(bf[p['road_field_key']],g['target_polygon'],q['word']);assert z['result']==q['result']
 check('All 44 unique-field full-contour results independently recomputed',True)
 check('No scalar observer heading silently promoted from candidates',sum(p['state']=='unique' for p in P)==0 and sum(p['state']=='unique_field' for p in P)==44)
 controls=readj(OUT/'negative_controls.json');check('Negative controls retain the original 44-relation denominator',all(c['support_n']==44 for c in controls) and next(c for c in controls if c['control']=='independent_random_rotation')['B']==300)
 details=readlines(OUT/'negative_control_details.jsonl.gz');reverse=[r for r in details if r['control']=='opposite_word_and_reverse_positive_direction_identity'];check('Opposite side plus reversed heading identity holds for all 44',len(reverse)==44 and all(r['identical'] for r in reverse))
 ints=readj(OUT/'fixed_geometry_language_interventions.json');inform=[r for r in ints if 'expected_180_error_deg' in r];check('Fixed-map role interventions rotate informative ranges 180deg without GT',len(inform)>0 and all(r['expected_180_error_deg']<1e-7 and r['real_geometry_unchanged'] and not r['GT_read_for_prediction'] for r in inform))
 fixed=readj(OUT/'fixed_origin_view_interventions.json');check('Fixed origin preserved while only named aim destination changes',len(fixed)>0 and all(r['origin_output_unchanged'] and r['real_geometry_unchanged'] and not r['GT_used'] for r in fixed) and any((r['support_center_change_deg'] or 0)>1 for r in fixed))
 for s in readj(OUT/'summary.json'):assert s['unique']+s['multiple']+s['unknown']==s['n'] and s['consistent']+s['conflict']==s['geometry_scorable'] and s['geometry_scorable']+s['unique_geometry_ambiguous_or_unavailable']==s['unique']
 check('Summary state/geometry denominators conserve every relation',True)
 figs=list((BASE4/'figures').glob('*.png'));check('14 high-resolution real-data PNG/PDF pairs present',len(figs)==14 and all(f.with_suffix('.pdf').exists() and min(Image.open(f).info['dpi'])>=319 for f in figs))
 text=(BASE4/'REPORT.md').read_text();check('All report relative links exist',all((BASE4/t).exists() for t in re.findall(r'\]\(([^)]+)\)',text) if not t.startswith('http')))
 rasters=readj(OUT/'raster_geometry_manifest.json');check('24 TRAIN map affine calibrations independent of GT; true north uncalibrated',len(rasters)==24 and all(r['image_right_is_positive_X'] and r['image_up_is_positive_Y'] and r['crs'] is None and not r['geographic_north_calibrated'] for r in rasters))
 diag=readj(OUT/'road_field_diagnostics.json');check('Road diagnostics cover the exact 37 final frozen configurations',len(diag)==len(fields)==37 and {d['key']:d['state'] for d in diag}=={f['key']:f['state'] for f in fields})
 check('Rule tests passed',(OUT/'test_results.txt').read_text().count('10 passed')==1)
 writej(OUT/'verification.json',{'status':'passed','n_checks':len(checks),'checks':checks,'relation_count':len(R),'recomputed_predictions':len(X),'recomputed_road_fields':len(cache),'report_sha256':sha(BASE4/'REPORT.md'),'no_independent_author_pose_labels':True,'limitations':'Local artifact/implementation reproducibility; does not establish true annotator frames or substitute independent scientific review'})
 print('PASS',len(checks),'checks;',len(X),'predictions,',len(cache),'fields rebuilt')
if __name__=='__main__':run()
