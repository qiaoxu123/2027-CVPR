"""Recover three real dense fields for one fixed-seed geometry-ablation query."""
from common import *
import sys
import numpy as np

def main():
    import torch
    torch.set_num_threads(1);sys.path.insert(0,str(ARCHIVE))
    from scripts import polygon_ablation_engine as engine
    from graphnav.geombelief_inputs import mask_geometry,SPATIAL
    plan=read(GEO/'frozen_plan.json');split='val_unseen';parent=OLD/'polygon_ablation'
    paths={
        'name_type':(GEO/'localization/map_geometry__name_type__seed0/val_unseen.localization.jsonl',GEO/'training/map_geometry__name_type__seed0/best.pt'),
        'shuffled':(GEO/'localization/map_geometry__shuffled__seed0/val_unseen.localization.jsonl',GEO/'training/map_geometry__shuffled__seed0/best.pt'),
        'correct':(parent/'localization/trained__descriptor_only__full_polygon__seed0/val_unseen.localization.jsonl',parent/'training/shared_full_polygon__full_polygon__seed0/best.pt')}
    scores={k:list(jsonl(p[0])) for k,p in paths.items()}
    records=list(jsonl(plan['paths'][split]))
    selected=next(i for i,r in enumerate(scores['correct']) if r['R@1']==1 and scores['name_type'][i]['R@1']==0 and scores['shuffled'][i]['R@1']==0 and len(records[i]['landmarks'])>0)
    donor=read(GEO/f'{split}_shuffle_manifest.json')[selected]['donor_index']
    ids=[selected,donor];payload=torch.load(parent/f'.cache/{split}_geometry.pt',map_location='cpu',weights_only=True,mmap=True)
    tensors={k:v[ids].clone() for k,v in payload['tensors'].items()}
    if 'tokens' not in tensors:
        lang=torch.load(parent/f'.cache/{split}_language.pt',map_location='cpu',weights_only=True,mmap=True)
        tensors.update({k:lang[k][ids].clone() for k in ('tokens','token_mask','node_emb')})
    sliced=dict(payload,tensors=tensors,metadata=[payload['metadata'][i] for i in ids])
    store=engine.Store(sliced,plan);fields={};sources=[]
    xy=np.asarray(records[selected]['world_bounds']);lo=xy[:2];span=xy[2:]-lo
    target=lo+np.asarray(records[selected]['target_xy'])*span
    grid=lo+store.grid.cpu().numpy()*span
    for arm,(source,cp) in paths.items():
        x,t,m,n,y,size=store.batch(torch.tensor([0]),'descriptor_only','full_polygon')
        if arm=='shuffled':
            other,*_=store.batch(torch.tensor([1]),'descriptor_only','full_polygon')
            x=dict(x,**{k:other[k] for k in SPATIAL})
        elif arm!='correct':x=mask_geometry(x,arm)
        model=engine.model_for(plan)
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True);model.eval().requires_grad_(False)
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):out=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
        logits=out.belief_logits[0].float().cpu().numpy();fields[arm]=logits
        top=int(logits.argmax());error=float(np.linalg.norm(grid[top]-target));saved=scores[arm][selected]
        assert top==saved['top16_grid_ids'][0],(arm,top,saved['top16_grid_ids'][0])
        assert abs(error-saved['NE'])<1e-3,(arm,error,saved['NE'])
        sources.append(dict(arm=arm,checkpoint=str(cp),checkpoint_sha256=sha(cp),source=str(source),source_sha256=sha(source),top1_xy=grid[top].tolist(),top1_error_m=error))
        del model
    dest=OUT/'fields/geometry_contrast.npz';np.savez_compressed(dest,world_xy=grid,target_xy=target,**fields)
    dump(OUT/'geometry_case.json',dict(query_index=selected,query_id=scores['correct'][selected]['query_id'],episode_id=records[selected]['episode_id'],split=split,dataset_version='revised',seed=0,public_map=records[selected],donor_map=records[donor],donor_index=donor,
        sources=sources,source_query_jsonl=plan['paths'][split],source_query_sha256=sha(plan['paths'][split]),arrays=str(dest.relative_to(OUT)),arrays_sha256=sha(dest),
        selection='Fixed seed 0, first source-order query with correct R@1=1, name/type R@1=0 and shuffled R@1=0 and at least one reference. Outcome-selected illustration, not population evidence.',
        scope='Separately trained matched-budget localization heads; not a frozen main-checkpoint intervention. Distances are world-coordinate measurements; no assumption that +Y equals north.'))
    print(json.dumps(dict(query_index=selected,episode_id=records[selected]['episode_id'],errors={r['arm']:r['top1_error_m'] for r in sources})))

if __name__=='__main__':main()
