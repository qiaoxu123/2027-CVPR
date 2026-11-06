"""TRAIN label-side loss-gradient conflicts; no optimizer and no inference intervention."""
import argparse,json
from pathlib import Path
import torch
from torch.nn.functional import cosine_similarity
from graphnav.unified_pgvb_v3 import UnifiedPGVBV3,v3_losses
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.unified_pgvb_v3_pilot import V3CacheDataset,_collate,_load_base,fit_dev_partition

def main():
    p=argparse.ArgumentParser()
    for k in ('run','source-v2','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();audit=json.loads((a.run/'label_geometry_audit.json').read_text())
    wanted={r['episode_id']:r for r in audit['exceptions']};ds=V3CacheDataset(a.run/'cache');_,dev=fit_dev_partition(ds,42);heldout=set(dev)
    selected={}
    for i,key in enumerate(ds.keys):
        label=torch.load(ds.root/'private'/f'{key}.pt',weights_only=True);identity=label['episode_id'];physical=':'.join(identity.split(':')[:2])
        if identity in wanted and physical not in selected:selected[physical]=(i,identity)
    geom,_=_load_base(a.source_v2,'cuda');model=UnifiedPGVBV3(geom).cuda().eval();model.load_state_dict(torch.load(a.run/'train/last.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    rows=[]
    for physical,(index,identity) in selected.items():
        kw,label=_collate([ds[index]],'cuda');out=model(**kw)
        loss=v3_losses(out,label['target_xy'],label['target_polygons'],kw['world_bounds'],label['teacher_features'],model.teacher_head)
        row=dict(physical_target=physical,episode_id=identity,heldout=index in heldout,**{k:v for k,v in wanted[identity].items() if k!='episode_id'})
        for ce,poly,logits in (('geo_ce','geo_polygon',out.geometry_logits),('fused_ce','fused_polygon',out.fused_logits)):
            left=torch.autograd.grad(loss[ce],logits,retain_graph=True)[0].flatten();right=torch.autograd.grad(loss[poly],logits,retain_graph=True)[0].flatten()
            row[ce+'_vs_'+poly]=dict(ce=float(loss[ce]),polygon=float(loss[poly]),gradient_cosine=float(cosine_similarity(left[None],right[None])),ce_gradient_norm=float(left.norm()),polygon_gradient_norm=float(right.norm()))
        rows.append(row)
    atomic_json(dict(scope='5 TRAIN physical targets with point–Contour >20m; private loss inspection only, no optimization',checkpoint_sha256=file_sha256(a.run/'train/last.pt'),rows=rows),a.output);print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
