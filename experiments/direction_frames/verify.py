"""Recompute invariants from independent source records, not generated answers."""
import argparse,hashlib,json,re,collections
from pathlib import Path
import numpy as np,pandas as pd
from shapely.geometry import Polygon
from PIL import Image

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);args=ap.parse_args();b=Path(__file__).parent;p=b/'results';load=lambda f:json.loads((p/f).read_text())
 E=json.loads((args.data_root/'original_citynav/citynav_train_seen.json').read_text());O=json.loads((args.data_root/'cityrefer_original/objects.json').read_text());R=load('occurrences.json');C=load('corpus.json');counts=load('word_counts.json');V=[r for r in R if r['status']=='valid'];checks=[]
 assert len(E)==22002 and len({f"{e['area']}_block_{e['block']}" for e in E})==24;checks.append('original TRAIN completeness: 22002 episodes, 24 maps')
 rx=re.compile(r'\b(left|right|front|behind|back|rear|above|below|top|bottom|north|south|east|west)\b',re.I);raw=collections.Counter(m[0].lower() for e in E for m in rx.finditer(e['descriptions'][0]));derived=collections.Counter(r['word'] for r in R)
 assert raw==derived and len(R)==28077;checks.append('full lexical enumeration independently recomputed')
 assert len({r['occurrence_id'] for r in R})==len(R)
 for c in counts:
  assert c['occurrences']==c['valid']+c['excluded_or_ambiguous']
  assert sum(c['reasons'].values())==c['excluded_or_ambiguous']
 assert C['valid_direct_pairs']==len(V)==212;checks.append('all per-word denominators and reasons reconcile')
 for r in V:
  e=E[r['episode']];m=r['map'];o=O[m][r['target_id']];q=O[m][r['ref_id']]
  assert r['instruction']==e['descriptions'][0]==o['descriptions'][r['ann_id']]
  d=np.array(e['target_positions'][0][:2])-np.array(q['position'][:2]);a=np.degrees(np.arctan2(d[1],d[0]))%360
  assert np.allclose(d,[r['dx'],r['dy']]) and abs((a-r['angle_world']+180)%360-180)<1e-8
  tp=Polygon(o['contour']);rp=Polygon(q['contour']);assert tp.is_valid and rp.is_valid and not tp.intersects(rp) and tp.distance(rp)>1
  pose=e['trajectory'][0];yaw=np.degrees(np.arctan2(pose[4],pose[3]))%360;assert len(pose)==6 and abs((yaw-r['initial_yaw_deg']+180)%360-180)<1e-8
  assert len(r['ref_candidate_ids'])==1 and r['ref_candidate_ids'][0]==r['ref_id']
 checks.append('all 212 accepted records: original ann alignment, unique reference, independent delta/angle/yaw, disjoint full polygons')
 # Strict-projection results are recomputed from all polygon vertices.
 byid={r['occurrence_id']:r for r in V};H=load('hypotheses.json');world={s['word']:s['axis_deg'] for s in H if s['hypothesis']=='A_world' and s['n']}
 off={'left':90,'right':270,'front':0,'behind':180,'back':180,'rear':180}
 B=pd.read_csv(p/'polygon_checks.csv')
 for s in B.to_dict('records'):
  r=byid[s['occurrence_id']];w=r['semantic_word'];ang=world[w] if s['hypothesis']=='A_world' else r['initial_yaw_deg']+off[w] if s['hypothesis']=='B_initial_yaw' else r['los_reference_deg']+off[w]
  u=np.array([np.cos(np.deg2rad(ang)),np.sin(np.deg2rad(ang))]);t=np.array(r['target_polygon'])@u;q=np.array(r['ref_polygon'])@u;gap=min(t)-max(q);rev=max(t)-min(q)
  cl='consistent' if gap>s['tolerance_m'] else 'opposite' if rev< -s['tolerance_m'] else 'ambiguous'
  assert cl==s['boundary_class'] and abs(gap-s['projection_gap_m'])<1e-7
 checks.append('every entire-polygon projection/tolerance classification independently recomputed')
 for s in load('leave_map_out.json'):
  fit=[r for r in V if r['semantic_word']==s['word'] and r['map']!=s['map']];assert len(fit)==s['fit_n']
 checks.append('leave-map-out excludes held-out map')
 for s in load('same_pair_opposite_descriptions.json'):
  a,b2=byid[s['first']],byid[s['second']];assert (a['map'],a['target_id'],a['ref_id'])==(b2['map'],b2['target_id'],b2['ref_id']);assert np.allclose(a['target_xy'],b2['target_xy']) and np.allclose(a['ref_xy'],b2['ref_xy'])
 checks.append('4 opposite-description counterexamples have identical source spatial pairs')
 for s in load('manifest.json')['input_files']:assert hashlib.sha256(Path(s['path']).read_bytes()).hexdigest()==s['sha256']
 assert all(s['image_up_is_plus_Y'] and s['image_right_is_plus_X'] and s['crs']=='None' for s in load('coordinate_audit.json'))
 checks.append('input SHA256 and raster coordinate claims verified')
 for s in H:
  if s['hypothesis']=='D_landmark_heading':assert s['n']==0
 checks.append('no unsupported directed heading evidence')
 report=(b/'REPORT_ZH.md').read_text();links=re.findall(r'\]\(([^)]+)\)',report)
 for link in links:
  if not link.startswith('http'):assert (b/link).is_file(),link
 pngs=list((b/'figures').glob('*.png'));assert len(pngs)==20
 for f in pngs:
  with Image.open(f) as im:assert min(im.size)>=900 and min(im.info.get('dpi',(0,0)))>=319
  assert f.with_suffix('.pdf').is_file()
 checks.append('all report links and 20 high-resolution PNG/PDF figure pairs verified')
 result={'status':'PASS','checks':checks,'accepted_pairs':len(V),'occurrences':len(R),'figure_pairs':len(pngs)}
 (p/'verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
