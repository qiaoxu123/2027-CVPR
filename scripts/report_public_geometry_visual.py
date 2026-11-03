"""Frozen training evidence, paired statistics and actual RGB failure cases."""
import argparse
import json
from pathlib import Path
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.public_geometry_visual import PublicGeometryVisualHead
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records
from scripts.train_public_geometry_visual import model_inputs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=a.root;suite=r/'public_geometry_visual_suite_v1_s0';a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    data={m:json.loads((suite/m/'dev_final.json').read_text()) for m in ('map','unconditioned','conditioned')}
    resolved={m:{c:v['metrics'] for c,v in x.items()} for m,x in data.items()};atomic_json(resolved,a.output/'resolved_metrics.json')
    rng=np.random.default_rng(0);reference=data['map']['real']['rows'];tasks=sorted({x['task_key'] for x in reference});ci={}
    for mode in ('unconditioned','conditioned'):
        for control in ('zero','shuffle'):
            real=data[mode]['real']['rows'];other=data[mode][control]['rows']
            groups={k:[int(x['success'])-int(y['success']) for x,y in zip(real,other) if x['task_key']==k] for k in tasks}
            values=[]
            for _ in range(5000):
                chosen=rng.choice(tasks,len(tasks),replace=True);flat=[v for k in chosen for v in groups[k]];values.append(np.mean(flat)*100)
            ci[mode+'-vs-'+control]=dict(delta_pp=float(np.mean([int(x['success'])-int(y['success']) for x,y in zip(real,other)]))*100,task_bootstrap95_pp=np.quantile(values,[.025,.975]).tolist())
    atomic_json(ci,a.output/'paired_uncertainty.json')
    names=['Frozen V3','+ Map','+ Visual','+ Conditioned'];metrics=[dict(r1=79/92,ne_m=17.271351215632066)]+[resolved[m]['real'] for m in ('map','unconditioned','conditioned')]
    fig,ax=plt.subplots(1,2,figsize=(10,4));ax[0].bar(names,[x['r1']*100 for x in metrics]);ax[0].set_ylabel('Goal R@1 (%)');ax[1].bar(names,[x['ne_m'] for x in metrics]);ax[1].set_ylabel('Goal NE (m)')
    for a0 in ax:a0.tick_params(axis='x',rotation=20)
    fig.tight_layout();fig.savefig(a.output/'development_comparison.png',dpi=160);plt.close(fig)
    # Select cases after final evaluation; these illustrate outcomes, not new tuning.
    cond=data['conditioned']['real']['rows'];zero=data['conditioned']['zero']['rows'];maps=data['map']['real']['rows']
    predicates=[('visual_harm',lambda c,z,m:c['base_success'] and not c['success'] and z['success']),
        ('correction',lambda c,z,m:not c['base_success'] and c['success']),
        ('map_success_visual_failure',lambda c,z,m:m['success'] and not c['success'])]
    chosen=[];used=set()
    for kind,predicate in predicates:
        for c,z,m in zip(cond,zero,maps):
            if predicate(c,z,m) and c['task_key'] not in used:
                chosen.append((kind,c['frame_key']));used.add(c['task_key']);break
    roles=torch.load(r/'reference_vectors_v2.pt',weights_only=True);geometry=torch.load(r/'public_geometry_visual_fields_v1/geometry_public.pt',weights_only=True)
    dev=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_dev.jsonl','dev',roles,'reference')
    field={};base_fields={};initial=torch.load(r/'world_v3_envelope_s0/best.pt',weights_only=True,map_location='cpu')
    for mode in ('map','conditioned'):
        base=WorldLikelihoodPositionHead();base.load_state_dict(initial['model']);model=PublicGeometryVisualHead(base,mode=mode).cuda().eval().requires_grad_(False)
        checkpoint=suite/mode/'final.pt';state=torch.load(checkpoint,weights_only=True,map_location='cpu');model.load_state_dict(state['model'])
        for start in range(0,len(dev),8):
            selected=dev[start:start+8];x,_=model_inputs(selected,geometry)
            with torch.inference_mode():out,old,_,_=model(**x,return_base=True)
            for i,row in enumerate(selected):field[(mode,row['frame_key'])]=out[i,:784].float().cpu();base_fields[row['frame_key']]=old[i,:784].float().cpu()
    frames={x['frame_key']:x for x in map(json.loads,(r/'fixed_yaw_train400_s1/public_frames.jsonl').read_text().splitlines())};bykey={x['frame_key']:x for x in dev};cases=[]
    for kind,key in chosen:
        row=bykey[key];f=frames[key];world=row['candidate_world'][:784].numpy();target=row['target_world'].numpy();footprint=np.asarray(f['footprint_world']);image=r/'fixed_yaw_train400_s1'/f['image'];assert file_sha256(image)==f['image_sha256']
        fig,axes=plt.subplots(1,4,figsize=(15,4));axes[0].imshow(cv2.cvtColor(cv2.imread(str(image)),cv2.COLOR_BGR2RGB));axes[0].axis('off');axes[0].set_title('Actual current RGB (yaw=0)')
        dx=np.unique(world[:,0])[1]-np.unique(world[:,0])[0];dy=np.unique(world[:,1])[1]-np.unique(world[:,1])[0];extent=[world[:,0].min()-dx/2,world[:,0].max()+dx/2,world[:,1].min()-dy/2,world[:,1].max()+dy/2]
        details={}
        for ax,name,logits in zip(axes[1:],['Frozen V3','Map','Conditioned'],[base_fields[key],field[('map',key)],field[('conditioned',key)]]):
            goal=world[int(logits.argmax())];ne=float(np.linalg.norm(goal-target));details[name]=dict(goal_world=goal.tolist(),ne_m=ne)
            ax.imshow(logits.softmax(-1).numpy().reshape(28,28),origin='lower',extent=extent,cmap='magma');ax.scatter(*target,c='red',marker='x',label='TRAIN target (posthoc)');ax.scatter(*goal,c='cyan',marker='+');poly=np.vstack([footprint,footprint[0]]);ax.plot(poly[:,0],poly[:,1],color='yellow',linewidth=.7);ax.set_title(f'{name}: NE {ne:.1f}m')
        fig.suptitle(key+' / '+kind);fig.tight_layout();fig.savefig(a.output/(key+'.png'),dpi=130);plt.close(fig)
        cases.append(dict(frame_key=key,kind=kind,instruction=f['instruction'],image_sha256=f['image_sha256'],predictions=details,visibility='unknown',pixel_gt=False))
    atomic_json(cases,a.output/'cases.json')
    for m in ('map','unconditioned','conditioned'):
        (a.output/m).mkdir(exist_ok=True)
        for name in ('config.json','history.json','status.json','fit_final.json','dev_final.json'):shutil.copyfile(suite/m/name,a.output/m/name)
    shutil.copyfile(suite/'freeze.json',a.output/'freeze.json')
    print(json.dumps(resolved,indent=2))

if __name__=='__main__':main()
