"""Compare loss domains only on reused TRAIN Dev; never read confirm predictions."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.geometry_guided_data import load_landmark_data,batch_landmarks
from graphnav.geometry_guided_visual import LandmarkVisualBelief,visual_world_beliefs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data','teacher','catalog','v1','v2','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    rows=[r for r in load_landmark_data(a.data,a.teacher,a.catalog) if r['metadata']['subset']=='dev']
    metrics={}
    for name,path in [('native',None),('candidate_distill_v1',a.v1),('world_distill_v2',a.v2)]:
        net=LandmarkVisualBelief().cuda().eval().requires_grad_(False)
        if path:net.load_state_dict(torch.load(path/'final.pt',weights_only=True,map_location='cpu')['model'])
        kl=[];global_kl=[]
        with torch.inference_mode():
            for start in range(0,len(rows),16):
                b=batch_landmarks(rows[start:start+16]);scores=net(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
                field=visual_world_beliefs(b['geometry_logits'],b['local_uv'],scores,b['valid'],b['projection_supported'])
                active=b['projection_supported']&field['has_view'];g=b['geometry_logits'].float().log_softmax(-1)
                if active.any():
                    mask=field['covered'][active]
                    gl=g[active].masked_fill(~mask,float('-inf')).log_softmax(-1).masked_fill(~mask,0)
                    vl=field['raw'][active].log_softmax(-1).masked_fill(~mask,0)
                    kl+=(gl.exp()*mask*(gl-vl)).sum(-1).cpu().tolist()
                global_kl+=(g.exp()*(g-field['fused'].log_softmax(-1))).sum(-1).cpu().tolist()
        metrics[name]=dict(conditional_world_KL=float(np.mean(kl)),global_preservation_KL=float(np.mean(global_kl)),samples=len(rows),supported=len(kl))
    v2=json.loads((a.v2/'evaluation.json').read_text())
    assert set(v2)=={'dev'}
    result=dict(teacher_agreement=metrics,v2_dev_metrics={k:v['metrics'] for k,v in v2['dev'].items()},
        confirmation_rescored_for_v2=False,val64_rescored_for_v2=False,
        warning='Reused TRAIN Dev regression after correcting operator/loss mismatch, not a fresh confirmation result.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
