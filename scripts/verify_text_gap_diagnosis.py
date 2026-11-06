"""Independently recompute final metrics and check controls after archive recovery."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--previous',required=True);p.add_argument('--run',required=True);p.add_argument('--report',required=True);a=p.parse_args();src=Path(a.previous);run=Path(a.run);report=Path(a.report);s=json.load(open(report/'summary.json'));texts=json.load(open(report/'text_variants.json'));base=np.load(src/'full_predictions.npz');grid=base['grid'];gt=base['gt'];dist=np.linalg.norm(grid-gt[:,None],axis=-1);arrays={'full':base['logits'],'mini':np.load(src/'target_type_relation_predictions.npz')['logits']}
 for f in (run/'factors').glob('*.npz'):arrays[f.stem]=np.load(f)['logits']
 for directory in ['checked_arrays','drop_corrected_arrays']:
  for f in (run/'restore_v2'/directory).glob('*.npy'):arrays[f.stem]=np.load(f)
 for arm,l in arrays.items():
  assert l.shape==(7619,784) and np.isfinite(l).all();err=np.take_along_axis(dist,np.argsort(-l,axis=-1),axis=-1)
  for k in [1,5,10,20,50]:assert abs(float((err[:,:k].min(-1)<=20).mean())-s['metrics'][arm][f'R{k}'])<1e-12
 controls={}
 for block in ['attributes','quantities','chain']:
  mask=np.array([not r['tags'][block] for r in texts]);delta=np.max(np.abs(arrays['full_drop_'+block+'@full'][mask]-arrays['full'][mask]),axis=-1);assert np.max(delta,initial=0)<1e-4
  controls[block]=dict(no_edit_queries=int(mask.sum()),max_abs_error=float(np.max(delta,initial=0)))
 old=json.load(open(src/'summary.json'));assert hashlib.sha256(Path(old['checkpoint']).read_bytes()).hexdigest()==old['checkpoint_sha']
 for v in json.load(open(report/'figures.json')).values():assert (report/v).exists()
 out=dict(status='passed',queries=7619,configurations=len(arrays),all_metrics_recomputed=True,all_logits_finite=True,no_edit_controls=controls,checkpoint_unchanged=True)
 (report/'verification.json').write_text(json.dumps(out,indent=2));print(out)
if __name__=='__main__':main()
