"""Read-only optimizer/parameter coverage audit; no data labels or model inference."""
import argparse,io,hashlib,json,math
from pathlib import Path
from collections import defaultdict
import torch
from scripts.unified_pgvb_v3_pilot import _load_base
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3
from graphnav.checkpointing import atomic_json

def main():
    p=argparse.ArgumentParser()
    for key in ['source-v2','initial-v3','resume','output']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();raw=a.resume.read_bytes();state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
    initial=torch.load(a.initial_v3,map_location='cpu',weights_only=True)['model'];g,_=_load_base(a.source_v2,'cpu');model=UnifiedPGVBV3(g)
    names=[('geometry.'+n,p) for n,p in model.geometry.named_parameters()]+[(n,p) for n,p in model.named_parameters() if not n.startswith('geometry.')]
    ids=[pid for pg in state['optimizer']['param_groups'] for pid in pg['params']];assert len(names)==len(ids)
    groups=defaultdict(lambda:dict(tensors=0,with_optimizer_state=0,with_nonzero_moment=0,moment_squared_norm=0.,weight_delta_squared_norm=0.));details=[];pose=None
    for (name,param),pid in zip(names,ids):
        saved=state['optimizer']['state'].get(pid);key=name.split('.')[0];out=groups[key];out['tensors']+=1
        delta=(state['model'][name]-initial[name]).float();out['weight_delta_squared_norm']+=float(delta.square().sum())
        moment_norm=0.
        if saved:
            avg=saved['exp_avg'];assert avg.shape==param.shape and torch.isfinite(avg).all();moment_norm=float(avg.float().norm());out['with_optimizer_state']+=1;out['with_nonzero_moment']+=int(moment_norm>0);out['moment_squared_norm']+=moment_norm**2
            if name=='pose_proj.weight':pose=dict(columns=['normalized_x','normalized_y','world_z_over100','sin_yaw','cos_yaw'],first_moment_column_norms=avg.float().norm(dim=0).tolist())
        details.append(dict(name=name,has_optimizer_state=saved is not None,first_moment_norm=moment_norm))
    for v in groups.values():v['moment_l2']=math.sqrt(v.pop('moment_squared_norm'));v['weight_delta_l2']=math.sqrt(v.pop('weight_delta_squared_norm'))
    result=dict(epoch_index=state['epoch'],next_offset=state['next_offset'],resume_sha256=hashlib.sha256(raw).hexdigest(),groups=dict(groups),pose_projection=pose,parameters=details,scope='Read-only training state audit; first moments measure recent gradient history, weight delta alone may include decay; no forward or quality evaluation')
    atomic_json(result,a.output);print(json.dumps(dict(groups=result['groups'],pose_projection=pose),indent=2))
if __name__=='__main__':main()
