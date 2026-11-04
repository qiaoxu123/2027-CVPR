"""Post-hoc scoring of immutable direction predictions with TRAIN labels only."""
import argparse,json
from pathlib import Path
import numpy as np
from graphnav.checkpointing import file_sha256
from graphnav.direction_belief import WORDS
from graphnav.direction_belief_data import UNIT


def score(prob,span,target,anchor,ids,heading=None):
 if not len(prob):return dict(n=0)
 yy,xx=np.meshgrid((np.arange(28)+.5)/28,(np.arange(28)+.5)/28,indexing='ij');grid=np.stack((xx,yy),-1)[None]*span[:,None,None];flat=grid.reshape(-1,784,2);p=prob.reshape(-1,784);a=anchor.reshape(-1,784);center=(flat*a[...,None]).sum(1);mean=(flat*p[...,None]).sum(1);peak=flat[np.arange(len(p)),p.argmax(1)];error=np.linalg.norm(peak-target,axis=-1);distance=np.linalg.norm(flat-target[:,None],axis=-1);unit=UNIT[ids].copy()
 if heading is not None:
  rel=ids>=4;c=np.cos(heading[rel]);s=np.sin(heading[rel]);u=unit[rel].copy();unit[rel,0]=c*u[:,0]-s*u[:,1];unit[rel,1]=s*u[:,0]+c*u[:,1]
 delta=mean-center;gt_delta=target-center;axis_dot=(delta*unit).sum(-1);gt_dot=(gt_delta*unit).sum(-1);cosine=axis_dot/np.linalg.norm(delta,axis=-1).clip(1e-8);half=((flat-center[:,None])*unit[:,None]).sum(-1)>0
 return dict(n=len(p),correct=int((error<=20).sum()),R20=float(np.mean(error<=20)),NE=float(error.mean()),median_NE=float(np.median(error)),GT20_mass=float((p*(distance<=20)).sum(1).mean()),direction_sign_consistency=float(np.mean(axis_dot>0)),direction_sector45=float(np.mean(cosine>=2**-.5)),direction_halfplane_mass=float((p*half).sum(1).mean()),GT_direction_sign_consistency=float(np.mean(gt_dot>0)),per_sample_NE=error.tolist())


def main():
 p=argparse.ArgumentParser();p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--labels',type=Path,required=True);a=p.parse_args();e=a.evaluation;seal=json.loads((e/'seal_before_scoring.json').read_text());assert file_sha256(e/'real_probabilities.npz')==seal['prediction_sha256'];assert file_sha256(e/'synthetic_probabilities.npz')==seal['synthetic_prediction_sha256'];labels={r['index']:r for r in json.loads(a.labels.read_text())};rows=json.loads((e/'public_dev.json').read_text());assert set(labels)=={r['index'] for r in rows};span=np.array([r['span'] for r in rows]);bounds=np.array([r['bounds'] for r in rows]);target=np.array([labels[r['index']]['target_world'][:2] for r in rows])-bounds[:,:2];ids=np.array([r['direction_id'] for r in rows]);real=np.load(e/'real_probabilities.npz');synth=np.load(e/'synthetic_probabilities.npz');truth=np.load(e/'synthetic_truth.npz');report=dict(real={},synthetic={},pgvb={},paired={},real_labels_sha256=file_sha256(a.labels),real_scope='34 held-out TRAIN relation pairs; original target point, not landmark Polygon; no human pixel labels')
 for source,dest,sp,tg,an,di,he in [(real,report['real'],span,target,real['anchor'],ids,None),(synth,report['synthetic'],truth['span'],truth['target_world'],synth['anchor'],truth['direction'],truth['heading'])]:
  for name in source.files:
   if name.startswith('pgvb_'):continue
   prob=source[name];dest[name]=dict(all=score(prob,sp,tg,an,di,he),by_word={w:score(prob[di==j],sp[di==j],tg[di==j],an[di==j],di[di==j],None if he is None else he[di==j]) for j,w in enumerate(WORDS)})
 indices=real['pgvb_indices'].astype(int)
 for name in ['nodirection_real','direction_real']:
  report['pgvb']['pgvb_'+name]=score(real['pgvb_'+name],span[indices],target[indices],real['pgvb_anchor'],ids[indices]);report['pgvb']['geometry_'+name]=score(real[name][indices],span[indices],target[indices],real['anchor'][indices],ids[indices])
 for name in ['direction_synthetic','direction_real','opposite_direction']:
  baseline=np.asarray(report['real']['nodirection_real']['all']['per_sample_NE'])<=20;current=np.asarray(report['real'][name]['all']['per_sample_NE'])<=20;report['paired'][name]=dict(corrected=int((~baseline&current).sum()),damaged=int((baseline&~current).sum()),net=int(current.sum()-baseline.sum()))
 (e/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:{m:{j:v[j] for j in ['n','R20','NE','GT20_mass','direction_sign_consistency']} for m,pack in report[k].items() if (v:=pack['all']).get('n')} for k in ['real','synthetic']},indent=2))
if __name__=='__main__':main()
