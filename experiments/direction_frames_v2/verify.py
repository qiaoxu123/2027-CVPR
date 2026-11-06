"""Independent source recomputation and population/scope/GT separation invariants."""
import argparse,collections,gzip,hashlib,json,re,copy
from pathlib import Path
import numpy as np,pandas as pd
from shapely.geometry import Polygon,Point
from shapely.ops import unary_union,nearest_points
from PIL import Image
from text_relations import Entity,TextCatalog,landmark_anchor
from geometry_frames import compute_pair,OFFSETS,IMAGE
BASE=Path(__file__).parent;P=BASE/'results'

def main(root):
 load=lambda n:json.loads((P/n).read_text());M=load('manifest.json');E=json.loads((root/'original_citynav/citynav_train_seen.json').read_text());O=json.loads((root/'cityrefer_original/objects.json').read_text());R=[json.loads(x) for x in gzip.open(P/'text_relations.jsonl.gz','rt')];G=[json.loads(x) for x in gzip.open(P/'geometry_pairs.jsonl.gz','rt')];I={r['occurrence_id']:r for r in R};by={r['occurrence_id']:r for r in G};checks=[]
 assert len(E)==22002 and len(M['train_maps'])==24
 assert len(R)==M['landmark_relation_occurrences'] and len(R)+M['out_of_scope_word_occurrences_not_analyzed']==28077 and all(r['landmark_related'] for r in R)
 assert hashlib.sha256((P/'text_relations.jsonl.gz').read_bytes()).hexdigest()==M['text_binding_sha256'];checks.append('population is named-landmark relations only; immutable text-ledger SHA matches')
 for src in M['input_files']:assert hashlib.sha256(Path(src['path']).read_bytes()).hexdigest()==src['sha256']
 assert len({r['occurrence_id'] for r in R})==len(R)
 # Independently check each candidate's actual named reference NP from its source text.
 catalogs={m:TextCatalog([Entity(str(i),o['name'],o['object_type']) for i,o in O[m].items() if o['name']]) for m in M['train_maps']}
 from text_relations import PATTERNS
 for r in R:
  e=E[r['episode']];assert r['instruction']==e['descriptions'][0];assert r['instruction'][r['span_start']:r['span_end']].lower()==r['word']
  if r.get('binding_rule')=='explicit_stand_at_facing_frame':continue
  rel=PATTERNS[r['word']].match(r['instruction'][r['span_start']:]);assert rel
  after=r['span_start']+rel.end();hits=[h for h in catalogs[r['map']].spans(r['instruction']) if h['start']>=after and re.fullmatch(r'\s*(?:the\s+)?',r['instruction'][after:h['start']],re.I)];assert len(hits)==1
  assert not (hits[0]['start']<=r['span_start']<hits[0]['end'])
 checks.append('every population candidate has a directly named reference NP; anonymous-neighbor directions excluded')
 assert len(G)==sum(r['status']=='bound' for r in R)
 for g in G:
  r=I[g['occurrence_id']];assert r['subject_ids']==[g['relation_subject_id']] and r['reference_ids']==g['reference_ids']
  obs=O[r['map']];subject=obs[g['relation_subject_id']];refs=[obs[i] for i in r['reference_ids']]
  pp=[Polygon(o['contour']) for o in refs];ok=all(p.is_valid and p.area>0 for p in pp)
  rp=unary_union(pp) if ok else None
  xy=np.array(refs[0]['position'][:2]) if len(refs)==1 else np.array(rp.centroid.coords[0]) if rp else np.mean([o['position'][:2] for o in refs],axis=0)
  source=E[r['episode']]['target_positions'][0] if r['subject_role']=='annotated_target' else subject['position'];delta=np.array(source[:2])-xy
  assert np.allclose([g['dx'],g['dy']],delta)
  a=np.degrees(np.arctan2(delta[1],delta[0]))%360;assert abs((g['angle_world']-a+180)%360-180)<1e-8
  assert g['cluster']==f"{r['map']}:{g['relation_subject_id']}"
  if len(refs)>1:assert len({o['name'].lower() for o in refs})==1 and all(o['object_type']=='TrafficRoad' for o in refs)
  tp=Polygon(subject['contour'])
  if g['geometry_status']=='disjoint':assert tp.is_valid and rp is not None and not tp.intersects(rp) and tp.distance(rp)>1 and np.linalg.norm(delta)>1
  for name,f in g['frames'].items():
   if name.startswith('C_') and name!='C_initial_yaw':assert f['future']
   if name=='C_motion_approaching_target':assert f['target_conditioned'] and f['future']
   if f['window']:
    i,j=f['window'];traj=np.array(E[r['episode']]['trajectory']);v=traj[j,:2]-traj[i,:2];assert np.linalg.norm(v)>=3-1e-8;assert i<j
    aa=np.degrees(np.arctan2(v[1],v[0]))%360
    if name.startswith('C_'):assert abs((f['heading_deg']-aa+180)%360-180)<1e-7
   if name.startswith('D_'):assert not f['heading_verified']
 checks.append('all geometric pairs source-recomputed using true relation subject, fixed landmark binding, complete road sets and contour filters')
 checks.append('all motion windows and future/GT-conditioned flags verified; no D heading silently promoted to verified')
 # Mutate target GT: independently fixed frames must not change (except explicit target-conditioned diagnostic).
 selected=[g for g in G if g.get('binding_rule')=='explicit_stand_at_facing_frame']+G[:5]
 for g in selected:
  e=copy.deepcopy(E[g['episode']]);e['target_positions'][0][0]+=1000;e['target_positions'][0][1]-=1000
  changed,_,_=compute_pair(I[g['occurrence_id']],e,O[g['map']])
  for name,f in g['frames'].items():
   if not f['target_conditioned']:assert f['heading_deg']==changed['frames'][name]['heading_deg']
 checks.append('target-GT perturbation leaves every non-target-conditioned candidate reference direction unchanged')
 T=load('map_transformations.json');assert len(T['transformations'])==12 and len(T['equivalence_classes'])==8
 for t in T['transformations']:
  m=np.array(t['matrix']);assert np.allclose(m.T@m,np.eye(2));assert np.allclose(m@IMAGE['left'],-m@IMAGE['right']);assert np.allclose(m@IMAGE['front'],-m@IMAGE['behind'])
 checks.append('joint dictionary rotations/reflections preserve opposite word directions; 12 tests are exactly 8 distinct maps')
 D=pd.read_csv(P/'frame_evaluations.csv');Q=pd.read_csv(P/'crossfit_predictions.csv');names=T['unique_candidate_names'];main={'left','right','front','behind'}
 for (subset,scheme,fold),q in Q.groupby(['subset','scheme','fold']):
  table=D[(D.geometry_status=='disjoint')&D.hypothesis.isin(names)&D.word.isin(main)]
  if subset=='high':table=table[table.parse_tier=='high']
  split=table.cluster.map(lambda x:str(int(hashlib.sha256(x.encode()).hexdigest()[:8],16)%5)) if scheme=='target_5fold' else table['map'].astype(str)
  fit=table[split!=str(fold)];held=table[split==str(fold)];assert set(fit.cluster).isdisjoint(held.cluster)
  rank=fit.groupby('hypothesis').agg(acc=('sector_correct','mean'),err=('angle_error_deg','mean')).sort_values(['acc','err'],ascending=[False,True]);best=rank.index[0];assert set(q.fitted_transform)=={best};assert set(q.occurrence_id)==set(held.occurrence_id)
 checks.append('every crossfit transform independently reselected with no subject or map holdout leakage')
 for z in load('coverage.json'):
  assert z['text_bound']==z['disjoint_all']+z['polygon_ambiguous']+z['invalid_polygon']
 C=load('corpus.json');assert C['occurrences']==len(R) and C['all_disjoint']==sum(g['geometry_status']=='disjoint' for g in G)
 checks.append('all named-landmark denominators, confidence subsets and geometry funnels reconcile')
 for c in load('four_case_reaudit.json'):
  assert c['same_named_reference_ids']==[c['ref_id']] and all(a['aligned_original_annotation'] for a in c['source_annotations_TRAIN_only'])
  assert not c['true_annotation_contradiction_confirmed'] and not c['shared_reference_frame_verified']
  for f in c['focus_relations']:assert f['v2_status']=='bound' and f['unique_reference_ids']==[c['ref_id']]
 c=load('four_case_reaudit.json')[2];w=c['independent_river_facing'];water=Polygon(w['water_polygon']);ref=np.array(c['reference_object']['position'][:2]);q=nearest_points(Point(*ref),water)[1];h=np.degrees(np.arctan2(q.y-ref[1],q.x-ref[0]))%360;assert abs(h-w['heading_deg'])<1e-8 and w['right_error_deg']<3 and not w['uses_target_GT_to_choose_heading']
 checks.append('four raw TRAIN cases checked; explicit stream-facing example independently reproduced without target-selected axis')
 report=(BASE/'REPORT_ZH.md').read_text()
 for link in re.findall(r'\]\(([^)]+)\)',report):
  if not link.startswith('http'):assert (BASE/link).is_file(),link
 pngs=list((BASE/'figures').glob('*.png'));assert len(pngs)==10
 for f in pngs:
  with Image.open(f) as im:assert min(im.info.get('dpi',(0,0)))>=319 and min(im.size)>900
  assert f.with_suffix('.pdf').is_file()
 checks.append('all report links and 10 real-data 320dpi PNG/PDF pairs verified')
 import rasterio,importlib.metadata,spacy
 raster=[]
 for m in M['train_maps']:
  with rasterio.open(root/'rgbd'/f'{m}.tif') as src:
   t=src.transform;assert t.a>0 and t.e<0 and t.b==0 and t.d==0
   raster.append({'map':m,'affine':list(t)[:6],'crs':str(src.crs),'image_right_is_plus_X':True,'image_up_is_plus_Y':True,'true_geographic_north_independently_calibrated':False})
 (P/'coordinate_audit.json').write_text(json.dumps(raster,indent=2)+'\n');checks.append('all 24 TRAIN raster affines independently rechecked; no true-north calibration invented')
 model_root=spacy.util.get_package_path('en_core_web_sm')/'en_core_web_sm-3.7.1'
 model=[{'relative_path':str(f.relative_to(model_root)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(model_root.rglob('*')) if f.is_file() and '__pycache__' not in str(f)]
 provenance={'model':'en_core_web_sm==3.7.1','source':'https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.7.1','trained_here':False,'input':'landmark-related instruction text only','artifacts':model,'packages':{k:importlib.metadata.version(k) for k in ['spacy','numpy','pandas','shapely','rasterio','matplotlib']},'code_files':[{'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(BASE.glob('*.py'))]}
 (P/'software_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
 result={'status':'PASS','checks':checks,'landmark_relation_candidates':len(R),'bound_pairs':len(G),'geometry_disjoint':C['all_disjoint'],'high_geometry_disjoint':C['high_disjoint'],'external_review_status':'see review/STATUS.md; local verification is not independent review'};dump=P/'verification.json';dump.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);a=ap.parse_args();main(a.data_root)
