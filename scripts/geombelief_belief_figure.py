"""Actual Seed0 correct/shuffled goal posterior, predeclared Unseen query index0."""
import json,time,textwrap
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.geombelief_map_experiments import OUT,OLD,load_store
from scripts.geombelief_results import map_paths
from scripts import polygon_ablation_engine as engine

@torch.no_grad()
def main():
    while not (OUT/'all_static_complete.json').exists():time.sleep(30)
    torch.set_num_threads(1);plan=json.loads((OUT/'frozen_plan.json').read_text());engine.OUT=OUT
    store=load_store(plan,'val_unseen');index=0
    records=[json.loads(l) for l in open(plan['paths']['val_unseen'])];record=records[index]
    donor=int(store.donors[index]);drecord=records[donor]
    probabilities=[];sources=[]
    for arm in ('correct','shuffled'):
        _,_,cp=map_paths(arm,0,'val_unseen','localization');model=engine.model_for(plan)
        model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['model'],strict=True);model.eval()
        ids=torch.tensor([index]);x,t,m,n,y,size=store.base.batch(ids,'descriptor_only','full_polygon') if arm=='correct' else store.batch(ids,'map_geometry',arm)
        with torch.autocast('cuda',dtype=torch.bfloat16):field=model(x,variant='geometry',tokens=t,token_mask=m,node_emb=n)
        probabilities.append(field.belief_logits.float().softmax(-1)[0].cpu().numpy().reshape(28,28));sources.append(dict(arm=arm,checkpoint=str(cp),sha256=file_sha256(cp)))
        del model
    bounds=np.asarray(record['world_bounds']);target=bounds[:2]+np.asarray(record['target_xy'])*(bounds[2:]-bounds[:2]);span=bounds[2:]-bounds[:2]
    out=OUT/'figures';out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,5),constrained_layout=True);maximum=max(p.max() for p in probabilities)
    for ax,p,r,title in zip(axes,probabilities,[record,drecord],['Correct map geometry','Shuffled spatial bundle']):
        im=ax.imshow(p,origin='lower',extent=[bounds[0],bounds[2],bounds[1],bounds[3]],cmap='magma',vmin=0,vmax=maximum,aspect='equal')
        for landmark in r['landmarks'][:16]:
            xy=bounds[:2]+np.asarray(landmark['contour'])*span;ax.add_patch(Polygon(xy,closed=True,fill=False,edgecolor='cyan',alpha=.65,lw=.6))
        ax.scatter(*target,marker='*',c='#40ff80',s=100,label='Ground-truth goal');ax.set(title=title,xlabel='World X (m)',ylabel='World Y (m)');ax.legend(fontsize=7)
    fig.colorbar(im,ax=axes,label='Predicted goal-cell probability',shrink=.7)
    fig.suptitle('\n'.join(textwrap.wrap(record['instruction'],100))+'\nPredeclared Seed 0; Val-Unseen query index 0',fontsize=9)
    for suffix in ('png','pdf'):fig.savefig(out/f'correct_vs_shuffled_belief.{suffix}',dpi=180)
    plt.close(fig)
    np.savez_compressed(out/'correct_vs_shuffled_belief.npz',correct=probabilities[0],shuffled=probabilities[1],world_bounds=bounds,target_xy=target)
    atomic_json(dict(split='val_unseen',query_id=store.metadata[0]['query_id'],episode_id=record['episode_id'],donor_query_id=store.metadata[donor]['query_id'],query_index=0,seed=0,source=sources,donor_index=donor,selection='Fixed first query, not outcome selected',interpretation='Two separately but identically budgeted trained models, each with its own correct/shuffled input scheme; not a frozen-weight intervention',map_and_language_sha256=file_sha256(plan['paths']['val_unseen'])),OUT/'belief_figure_manifest.json')
    path=OUT/'qualitative_cases.md';text=path.read_text().split('## Correct versus shuffled geometry belief')[0]
    text+='## Correct versus shuffled geometry belief\n\nActual normalized model probabilities, predeclared Seed0 and first Val-Unseen query. Cyan outlines are the geometry supplied to each model. Recipient language and landmark names are unchanged. Separately budget-matched trained models; not a same-weight feature intervention. Source IDs, donor and checkpoint SHAs: belief_figure_manifest.json.\n\n![Actual goal belief](figures/correct_vs_shuffled_belief.png)\n';path.write_text(text)

if __name__=='__main__':main()
