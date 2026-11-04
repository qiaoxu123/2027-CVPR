"""Separate TRAIN post-hoc scoring; never edits predictions or direction rules."""
import argparse,json
from pathlib import Path
import numpy as np
from graphnav.checkpointing import file_sha256
from graphnav.direction_belief_data import UNIT
from scripts.score_direction_belief import score
from scripts.score_strict_direct_direction import paired
from scripts.audit_direction_frames import interval_status

METHODS=['anchor','centroid32','no_direction','original32','polygon_ray','polygon_cone']

def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);p.add_argument('--phase',choices=['development','confirmation'],required=True);a=p.parse_args();o=a.directory;seal=json.loads((o/'prediction_seal.json').read_text());assert file_sha256(o/'predictions.npz')==seal['prediction_sha256'];assert file_sha256(o/'public.json')==seal['public_sha256'];allrows=json.loads((o/'public.json').read_text());labels={r['index']:r for r in json.loads(a.labels.read_text())};indices=[j for j,r in enumerate(allrows) if r['index'] in labels];rows=[allrows[j] for j in indices];assert len(rows)==len(labels);expected={'confirm'} if a.phase=='confirmation' else {'fit','dev'};assert {r['subset'] for r in rows}==expected
 if a.phase=='confirmation':assert json.loads((o/'development_decision.json').read_text())['final_strategy_frozen'] is True;assert not (o/'confirmation_metrics.json').exists()
 pred=np.load(o/'predictions.npz');span=np.array([r['span'] for r in rows]);bounds=np.array([r['bounds'] for r in rows]);targets=np.array([labels[r['index']]['target_world'][:2] for r in rows]);ids=np.array([r['direction_id'] for r in rows]);anchor=pred['anchor'][indices];metrics={};details=[]
 for subset in sorted(expected):
  keep=np.array([r['subset']==subset for r in rows]);pack={}
  for name in METHODS:
   pr=pred[name][indices];pack[name]=dict(all=score(pr[keep],span[keep],targets[keep]-bounds[keep,:2],anchor[keep],ids[keep]),by_word={word:score(pr[keep&(ids==k)],span[keep&(ids==k)],targets[keep&(ids==k)]-bounds[keep&(ids==k),:2],anchor[keep&(ids==k)],ids[keep&(ids==k)]) for k,word in enumerate(['north','south','east','west'])})
  pack['polygon_cone_vs_original32']=paired(pack['original32']['all'],pack['polygon_cone']['all']);pack['polygon_cone_vs_centroid32']=paired(pack['centroid32']['all'],pack['polygon_cone']['all']);metrics[subset]=pack
 for j,r in enumerate(rows):
  yy,xx=np.meshgrid((np.arange(28)+.5)/28,(np.arange(28)+.5)/28,indexing='ij');grid=np.stack([xx,yy],-1)*span[j]+bounds[j,:2];distance=np.linalg.norm(grid-targets[j],axis=-1);center=(anchor[j][...,None]*grid).sum((0,1))/anchor[j].sum();record=dict(index=r['index'],candidate_id=r['candidate_id'],episode=r['episode'],subset=r['subset'],group=r['group'],word=r['word'],landmark_name=r['landmark_name'],map_name=r['map_name'],instruction=r['instruction'],target_world=targets[j].tolist(),gt_engineering_direction_margin=float((targets[j]-center)@UNIT[ids[j]]),gt_polygon_axis_status=interval_status(np.concatenate(r['public_polygons']),targets[j],UNIT[ids[j]])[0],methods={})
  for name in METHODS:
   pr=pred[name][indices[j]];peak=grid.reshape(-1,2)[pr.argmax()];ne=float(np.linalg.norm(peak-targets[j]));order=np.argsort(-pr.ravel(),kind='stable');count=np.searchsorted(np.cumsum(pr.ravel()[order]),.9)+1;credible=np.zeros(784,bool);credible[order[:count]]=True
   v=dict(NE=ne,R20=ne<=20,GT20_mass=float(pr[distance<=20].sum()),peak=peak.tolist(),credible90_cells=int(credible.sum()),credible90_R20=bool((credible.reshape(28,28)&(distance<=20)).any()))
   if name.startswith('polygon_'):
    mode=name.split('_')[1];support=pred['support_'+mode][indices[j]];v.update(legal_support_cells=int(support.sum()),legal_support_GT20=bool((support&(distance<=20)).any()),abstained=bool(not support.any()))
   record['methods'][name]=v
  details.append(record)
 diagnostic={s:dict(n=sum(r['subset']==s for r in details),groups=len({r['group'] for r in details if r['subset']==s}),GT_sign_consistent=sum(r['gt_engineering_direction_margin']>0 for r in details if r['subset']==s),GT_wholly_opposite=sum(r['gt_polygon_axis_status']=='opposite_side' for r in details if r['subset']==s),polygon_cone_GT20_support=sum(r['methods']['polygon_cone']['legal_support_GT20'] for r in details if r['subset']==s),polygon_cone_mean_support_cells=float(np.mean([r['methods']['polygon_cone']['legal_support_cells'] for r in details if r['subset']==s])),polygon_cone_abstentions=sum(r['methods']['polygon_cone']['abstained'] for r in details if r['subset']==s)) for s in expected}
 (o/f'{a.phase}_metrics.json').write_text(json.dumps(metrics,indent=2));(o/f'{a.phase}_samples.json').write_text(json.dumps(details,ensure_ascii=False,indent=2));(o/f'{a.phase}_diagnostics.json').write_text(json.dumps(diagnostic,indent=2));(o/f'{a.phase}_scoring_seal.json').write_text(json.dumps(dict(labels_sha256=file_sha256(a.labels),prediction_sha256=file_sha256(o/'predictions.npz'),labels_used_only_for_scoring=True,producer_sha256=file_sha256(__file__)),indent=2))
 if a.phase=='development':
  old=metrics['dev']['original32']['all'];new=metrics['dev']['polygon_cone']['all'];tests=dict(GT20_mass_improved=new['GT20_mass']>old['GT20_mass'],R20_not_worse=new['R20']>=old['R20'],NE_not_worse=new['NE']<=old['NE']);passed=all(tests.values());decision=dict(predeclared_gate=seal['primary_training_gate'],tests=tests,proceed_training=passed,final_strategy_frozen=not passed,new_model_status='pending fit-only training' if passed else 'not trained: predeclared baseline gate failed',confirmation_GT_scored=False,development_metrics_sha256=file_sha256(o/'development_metrics.json'),prediction_sha256=seal['prediction_sha256']);(o/'development_decision.json').write_text(json.dumps(decision,indent=2));print(json.dumps(decision,indent=2))
 print(json.dumps({s:{name:{k:v[name]['all'][k] for k in ['n','R20','NE','GT20_mass']} for name in METHODS} for s,v in metrics.items()},indent=2))
if __name__=='__main__':main()
