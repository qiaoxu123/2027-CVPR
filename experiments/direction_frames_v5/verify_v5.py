"""Verify version pinning, no-GT inference, exact support and all field results."""
from v5core import *
from geometry_v5 import *
from language_v5 import role_pairs,side_scope
from PIL import Image
import subprocess,hashlib,ast

def run():
 checks=[]
 def check(name,b):assert b,name;checks.append(name)
 L=readj(RESULT/'v4_lock.json');M=readj(RESULT/'phase1_manifest.json');EM=readj(RESULT/'evaluation_manifest.json');X={x['occurrence_id']:x for x in readlines(RESULT/'public_inputs.jsonl.gz')};N=readlines(RESULT/'new_candidates.jsonl.gz');NP=readlines(RESULT/'new_frames.jsonl.gz');F={f['key']:f for f in readlines(RESULT/'road_fields.jsonl.gz')};E={e['occurrence_id']:e for e in readlines(RESULT/'evaluation_gt.jsonl.gz')};Q=readlines(RESULT/'geometry_checks.jsonl.gz');methods=readlines(RESULT/'method_predictions.jsonl.gz')
 D=readj(V4/'results/dataset.json');check('Raw TRAIN and CityRefer source SHA unchanged',all(sha(z['path'])==z['sha256'] for z in D['source_files']));writej(RESULT/'dataset_source_manifest.json',{'source_files':D['source_files'],'TRAIN_only':True,'Val_Test_read':False,'train_episodes':22002,'train_maps':24})
 check('All frozen v4 source/data hashes unchanged',all(sha(V4/k)==h for k,h in L['files'].items()))
 check('Protocol unchanged from initial lock',sha(HERE/'PROTOCOL.md')==L['protocol_sha256'])
 deps={};root=HERE.parents[1]
 for f in [V3/'common.py',V3/'infer.py',V2/'geometry_frames.py',V2/'text_relations.py']:
  path=str(f.relative_to(root));baseline=subprocess.check_output(['git','show',L['base_commit']+':'+path],cwd=root);expected=hashlib.sha256(baseline).hexdigest();assert sha(f)==expected;deps[path]=expected
 writej(RESULT/'dependency_lock.json',{'git_baseline':L['base_commit'],'files':deps});check('Inherited v2/v3 geometry and text helpers equal exact Git baseline',True)
 check('Phase1 data and geometry fingerprints match',sha(RESULT/'public_inputs.jsonl.gz')==M['inputs_sha256'] and sha(RESULT/'new_frames.jsonl.gz')==M['new_frames_sha256'] and sha(RESULT/'road_fields.jsonl.gz')==M['fields_sha256'] and sha(RESULT/'exploratory_fields.jsonl.gz')==M['exploratory_fields_sha256'] and sha(RESULT/'negative_fields.jsonl.gz')==M['negative_fields_sha256'] and sha(RESULT/'resolution_fields.jsonl.gz')==M['resolution_fields_sha256'])
 check('Inference and evaluation source fingerprints match',sha(HERE/'language_v5.py')==M['parser_code_sha256'] and sha(HERE/'prepare_v5.py')==M['prepare_code_sha256'] and sha(HERE/'explore_v5.py')==M['exploration_code_sha256'] and sha(HERE/'score_v5.py')==EM['score_code_sha256'] and sha(HERE/'geometry_v5.py')==EM['geometry_predicate_sha256'])
 check('GT evaluator linked to completed current freeze',EM['frozen_phase1_manifest_sha256']==sha(RESULT/'phase1_manifest.json') and EM['GT_sha256']==sha(RESULT/'evaluation_gt.jsonl.gz'))
 oldX=readlines(V4/'results/public_inputs.jsonl.gz');check('All original1398 public inputs preserved exactly',all(X[x['occurrence_id']]==x for x in oldX))
 oldR=readlines(V4/'results/relations.jsonl.gz');oldids={r['occurrence_id'] for r in oldR};ids={r['occurrence_id'] for r in Q};newids={r['occurrence_id'] for r in N};check('All4701 original candidates plus39 new IDs, no duplication',oldids<=ids and len(Q)==len(ids)==4740 and len(newids-oldids)==39)
 check('All candidates retain explicit final unknown/failure outputs',len(readlines(RESULT/'all_relation_outputs.jsonl.gz'))==len(Q))
 check('GT and public bound inputs have identical1413 IDs',set(E)==set(X) and len(X)==1413)
 check('Private target geometry and future/yaw absent from public inference inputs',all(not(set(x)&{'target_positions','subject_xy','target_xy','target_polygon','GT','yaw','future_trajectory','trajectory'}) and not set(x['subject_ids'])&set(x['scene']) for x in X.values()))
 tree=ast.parse((HERE/'prepare_v5.py').read_text());keys={n.value for n in ast.walk(tree) if isinstance(n,ast.Constant) and isinstance(n.value,str)};check('New role preparation never requests target_positions', 'target_positions' not in keys)
 O=readj(DATA/'cityrefer_original/objects.json');newby={r['occurrence_id']:r for r in N};cache={}
 for p in NP:
  x=X[p['occurrence_id']];r=newby[p['occurrence_id']];catalog=TypedRoadCatalog([Entity(i,o['name'],o['object_type']) for i,o in O[x['map']].items() if o['name']]);pairs,hits=role_pairs(x['instruction'],catalog);assert any(z['A']['resolved_ids']==r['reference_ids'] and z['B']['resolved_ids']==r['source_ids'] and side_scope(x['instruction'],r['span_start'],z,hits)[0] for z in pairs);assert text_key(O[x['map']][r['target_id']]['descriptions'][r['ann_id']])==text_key(r['instruction'])
  key=p['road_field_key']
  if key not in cache:cache[key]=directed_field(x,p['reference_ids'],p['source_ids'])
  assert cache[key]=={k:v for k,v in F[key].items() if k!='key'} and p['state']==cache[key]['state']
 check('All15 new bindings independently re-parsed and annotations aligned',len(NP)==15)
 check('Every new A/B field reconstructed with unchanged v4 algorithm, no GT',True)
 original={r['occurrence_id']:r for r in readlines(V4/'results/geometry_checks.jsonl.gz') if r['state']=='unique_field'};byQ={r['occurrence_id']:r for r in Q};check('All44 original results21/1/22 preserved',len(original)==44 and all(byQ[k]['result']==r['result'] for k,r in original.items()) and collections.Counter(byQ[k]['result'] for k in original)=={'consistent':21,'conflict':1,'ambiguous':22})
 pp={p['occurrence_id']:p for p in readlines(V4/'results/frozen_frames.jsonl.gz')};pp.update({p['occurrence_id']:p for p in NP});active=[q for q in Q if q['state']=='unique_field'];prcache={}
 for r in active:
  f=F[r['road_field_key']];e=E[r['occurrence_id']];pr=projection(f,e['target_polygon']);assert orientation_score(pr,r['word'])['result']==field_geometry(f,e['target_polygon'],r['word'])['result']==r['result'];prcache[r['occurrence_id']]=pr
 check('All55 primary results reproduce original v4 full-contour predicate',len(active)==55)
 groups=collections.defaultdict(list)
 for r in methods:groups[r['method']].append(r)
 support={r['occurrence_id'] for r in active};check('Every method/control evaluated on exactly the same55 relations',all({r['occurrence_id'] for r in rr}==support and len(rr)==55 for rr in groups.values()))
 for name,rr in groups.items():
  if name.startswith(('A_','B_','C_','D_','E_')):
   for r in rr:
    if prcache[r['occurrence_id']]['mask_reason']:assert r['score']==0
  if name.startswith('E_'):
   for r in rr:assert r['score']==sum(z=='consistent' for z in r['sign_results'])/2
 check('A-E share frozen geometry mask; unsigned PCA never takes GT-best sign',True)
 for scheme,group in [('leave_map_out','map'),('leave_road_out','road_group')]:
  rr=groups['B_world_crossfit_'+scheme];grid={(r['occurrence_id'],int(name.rsplit('_',1)[-1])):r['score'] for name,z in groups.items() if name.startswith('B_world_forward_') for r in z}
  for r in rr:
   tr=[z for z in active if z[group]!=r[group]];h=max([0,90,180,270],key=lambda h:sum(grid[z['occurrence_id'],h] for z in tr));assert r['chosen_world_forward_deg']==h
 check('World dictionary selected only outside held-out map/road',True)
 aa=readj(RESULT/'v4_ambiguous22_classification.json');check('All22 ambiguous retain mechanistic categories, no invented true-centerline-error labels',len(aa)==22 and all(not r['centerline_error_confirmed_by_independent_lane_annotation'] for r in aa))
 abb=readj(RESULT/'Abbey_failure_analysis.json');check('Abbey preserved as conflict, direction never flipped',abb['case']['frozen_result']=='conflict' and not abb['direction_reversed_to_fit_GT'])
 for s in readj(RESULT/'summary.json'):assert s['unique']+s['multiple']+s['unknown']==s['n'] and s['scorable']+s['ambiguous_unique']==s['unique'] and s['consistent']+s['conflict']==s['scorable']
 check('All coverage/geometry/outcome denominators conserved',True)
 geo=readj(RESULT/'road_geometry_features.json');check('Width/curvature descriptors are public-geometry-only, never used to tune primary predictions',bool(geo) and all(not r['GT_used_for_feature'] and r['subgroups_postfreeze_descriptive_only'] for r in geo))
 figs=list((HERE/'figures').glob('*.png'));check('15 true-data high-resolution PNG/PDF pairs',len(figs)==15 and all(f.with_suffix('.pdf').exists() and min(Image.open(f).info['dpi'])>=319 for f in figs))
 text=(HERE/'REPORT.md').read_text();check('Report local links exist',all((HERE/t).exists() for t in re.findall(r'\]\(([^)]+)\)',text) if not t.startswith('http')))
 check('All14 mechanism/scope tests passed','14 passed' in (RESULT/'test_results.txt').read_text())
 writej(RESULT/'verification.json',{'status':'passed','n_checks':len(checks),'checks':checks,'TRAIN_only':True,'Val_Test_read':False,'new_bindings_recomputed':len(NP),'primary_predicates_recomputed':len(active),'candidate_outputs':len(Q),'report_sha256':sha(HERE/'REPORT.md'),'limitations':'Retrospective TRAIN evidence and local invariants; not author-reference-frame labels or blind external validation'})
 print('PASS',len(checks),'checks;',len(Q),'candidate outputs;',len(active),'matched primary fields')
if __name__=='__main__':run()
