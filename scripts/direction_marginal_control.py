"""Fixed-weight direction-marginal control, separate prediction and scoring phases."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from graphnav.checkpointing import file_sha256
from graphnav.direction_belief import DirectionBelief
from graphnav.direction_belief_data import synthetic
from scripts.evaluate_direction_belief import predict
from scripts.score_direction_belief import score


def main():
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['predict','score']);p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--models',type=Path);p.add_argument('--labels',type=Path);a=p.parse_args();e=a.evaluation;o=e/'marginal_control'
 if a.mode=='predict':
  assert not o.exists();o.mkdir();torch.set_num_threads(4);pub=json.loads((e/'public_dev.json').read_text());real=dict(anchor=torch.tensor([r['anchor'] for r in pub],device='cuda'),span=torch.tensor([r['span'] for r in pub],device='cuda'));syn={k:v.to('cuda') for k,v in synthetic(1024,881).items()};sources=[('real',real,4,'direction_real'),('synthetic',syn,8,'direction_synthetic')];proof={}
  for label,data,nwords,name in sources:
   path=a.models/(name+'.pt');state=torch.load(path,map_location='cpu',weights_only=True);m=DirectionBelief().to('cuda').eval().requires_grad_(False);m.load_state_dict(state['model']);averaged=np.zeros((len(data['anchor']),28,28),np.float32)
   for word in range(nwords):averaged+=predict(m,data['anchor'],torch.full((len(averaged),),word,device='cuda',dtype=torch.long),data['span'],data.get('heading'))/nwords
   np.testing.assert_allclose(averaged.sum((1,2)),1,atol=1e-5);np.save(o/(label+'.npy'),averaged);proof[label]=dict(words_marginalized=nwords,model_sha256=file_sha256(path),prediction_sha256=file_sha256(o/(label+'.npy')))
  proof['real_target_labels_read']=False;proof['purpose']='Post-result capacity control; no fitting, selection or changes to primary frozen models.';(o/'seal_before_scoring.json').write_text(json.dumps(proof,indent=2))
 else:
  seal=json.loads((o/'seal_before_scoring.json').read_text());assert all(file_sha256(o/(s+'.npy'))==seal[s]['prediction_sha256'] for s in ['real','synthetic']);labels={r['index']:r for r in json.loads(a.labels.read_text())};pub=json.loads((e/'public_dev.json').read_text());span=np.array([r['span'] for r in pub]);target=np.array([labels[r['index']]['target_world'][:2] for r in pub])-np.array([r['bounds'][:2] for r in pub]);ids=np.array([r['direction_id'] for r in pub]);base=np.load(e/'real_probabilities.npz');truth=np.load(e/'synthetic_truth.npz');synthbase=np.load(e/'synthetic_probabilities.npz');real=score(np.load(o/'real.npy'),span,target,base['anchor'],ids);syn=score(np.load(o/'synthetic.npy'),truth['span'],truth['target_world'],synthbase['anchor'],truth['direction'],truth['heading']);main=json.loads((e/'metrics.json').read_text());b=np.array(real['per_sample_NE'])<=20;c=np.array(main['real']['direction_real']['all']['per_sample_NE'])<=20;out=dict(real=real,synthetic=syn,conditional_vs_marginal=dict(corrected=int((~b&c).sum()),damaged=int((b&~c).sum()),net=int(c.sum()-b.sum())));(o/'metrics.json').write_text(json.dumps(out,indent=2));print(json.dumps({k:{j:v[j] for j in ['n','correct','R20','NE','GT20_mass']} for k,v in out.items() if k in ['real','synthetic']},indent=2));print(out['conditional_vs_marginal'])
if __name__=='__main__':main()
