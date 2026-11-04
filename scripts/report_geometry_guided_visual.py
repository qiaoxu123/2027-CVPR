"""Stage1 frozen metrics, teacher agreement, uncertainty and real RGB maps."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
import shutil
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.geometry_guided_data import load_landmark_data,batch_landmarks
from graphnav.geometry_guided_visual import LandmarkVisualBelief,visual_world_beliefs
from graphnav.landmark_projection import world_to_frame


def group_interval(real,control,key,replicates=2000):
    other={r['sample_key']:r for r in control};groups=defaultdict(list)
    for r in real:
        if r['eligible'] and r['sample_key'] in other:
            groups[r['query_key']].append(int(r[key])-int(other[r['sample_key']][key]))
    if not groups:return None
    rng=random.Random(0);names=sorted(groups);values=[]
    for _ in range(replicates):
        sample=[x for n in rng.choices(names,k=len(names)) for x in groups[n]];values.append(np.mean(sample))
    return dict(named_groups=len(groups),mean_difference=float(np.mean([x for g in groups.values() for x in g])),
        name_group_bootstrap95=np.quantile(values,[.025,.975]).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data','teacher','catalog','training','regression','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    evaluation=json.loads((a.training/'evaluation.json').read_text());state=torch.load(a.training/'final.pt',map_location='cpu',weights_only=True)
    model=LandmarkVisualBelief().cuda().eval().requires_grad_(False);model.load_state_dict(state['model'])
    native=LandmarkVisualBelief().cuda().eval().requires_grad_(False)
    data=load_landmark_data(a.data,a.teacher,a.catalog);consistency={};intervals={}
    with torch.inference_mode():
        for subset in ('dev','confirm'):
            chosen=[r for r in data if r['metadata']['subset']==subset];consistency[subset]={}
            for name,network in [('native',native),('trained',model)]:
                kl=[];fused_kl=[]
                for start in range(0,len(chosen),16):
                    rows=chosen[start:start+16];b=batch_landmarks(rows)
                    score=network(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
                    fields=visual_world_beliefs(b['geometry_logits'],b['local_uv'],score,b['valid'],b['projection_supported'])
                    g=b['geometry_logits'].float().log_softmax(-1)
                    restricted=g.masked_fill(~fields['covered'],float('-inf'))
                    active=b['projection_supported']&fields['has_view']
                    if active.any():
                        gl=restricted[active].log_softmax(-1);vl=fields['raw'][active].log_softmax(-1)
                        mask=fields['covered'][active];gp=gl.exp()
                        value=torch.where(mask,gp*(gl-vl),torch.zeros_like(gp)).sum(-1)
                        kl+=value.cpu().tolist()
                    lp=fields['fused'].log_softmax(-1)
                    fused_kl+=((g.exp()*(g-lp)).sum(-1)).cpu().tolist()
                consistency[subset][name]=dict(supported_samples=len(kl),conditional_G_to_V_KL=float(np.mean(kl)),
                    global_G_to_GV_KL=float(np.mean(fused_kl)))
            real=evaluation[subset]['real']['rows'];intervals[subset]={}
            for control in ('native','shuffle_rgb','wrong_rgb','same_type_rgb'):
                intervals[subset][control]=group_interval(real,evaluation[subset][control]['rows'],'weak_top1')
    regression=json.loads((a.regression/'metrics.json').read_text())
    if not regression['complete']:raise ValueError('64 paired regressions incomplete')
    nav=[json.loads(line) for line in (a.regression/'predictions.jsonl').read_text().splitlines()]
    paired={i:{r['method']:r for r in nav if r['index']==i} for i in range(64)}
    nav_changes=dict(corrected=sum(p['GeometryVisual']['metrics']['SR'] and not p['Geometry']['metrics']['SR'] for p in paired.values()),
        harmed=sum(not p['GeometryVisual']['metrics']['SR'] and p['Geometry']['metrics']['SR'] for p in paired.values()))
    for pair in paired.values():assert pair['Geometry']['poses'][0]==pair['GeometryVisual']['poses'][0]
    gate=dict(weak_dev_beats_native=evaluation['dev']['real']['metrics']['weak_top1']>evaluation['dev']['native']['metrics']['weak_top1'],
        weak_dev_beats_same_type=evaluation['dev']['real']['metrics']['weak_top1']>evaluation['dev']['same_type_rgb']['metrics']['weak_top1'],
        dev_fusion_no_net_harm=evaluation['dev']['real']['metrics']['fused_corrected']>=evaluation['dev']['real']['metrics']['fused_harmed'],
        dev_fusion_NE_no_worse=evaluation['dev']['real']['metrics']['fused_ne']<=evaluation['dev']['real']['metrics']['geometry_ne'])
    result=dict(consistency=consistency,weak_top1_intervals=intervals,regression=regression,nav_changes=nav_changes,
        preregistered_gate=gate,stage1_gate_passed=all(gate.values()),stage2_started=False,
        note='Weak public region supervision, no human pixel GT. Existing map-group dev/confirm exposure history retained.')
    (a.output/'analysis.json').write_text(json.dumps(result,indent=2))
    for file in ('config.json','freeze.json','history.json','evaluation.json'):
        shutil.copyfile(a.training/file,a.output/file)
    for file in ('metrics.json','predictions.jsonl','run_manifest.json'):
        shutil.copyfile(a.regression/file,a.output/('regression_'+file))
    atomic=state['gradients'];(a.output/'gpu_gradients.json').write_text(json.dumps(atomic,indent=2))
    history=state['history'];fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for name in ('total','region','distill','contrast'):axes[0].plot([r['epoch'] for r in history],[r[name] for r in history],label=name)
    axes[0].legend();axes[0].set_title('Training loss (last batch per epoch)')
    for j,subset in enumerate(('dev','confirm')):
        m=evaluation[subset];axes[1].bar(np.arange(3)+j*.35,[m[c]['metrics']['weak_top1']/m[c]['metrics']['weak_eligible'] for c in ('native','real','same_type_rgb')],width=.35,label=subset)
    axes[1].set_xticks(np.arange(3)+.175,['Native','Trained','Wrong same-type RGB']);axes[1].legend();axes[1].set_title('Weak region Top1, not pixel instance accuracy')
    fig.tight_layout();fig.savefig(a.output/'training_and_controls.png',dpi=160);plt.close(fig)
    by_key={r['metadata']['sample_key']:r for r in data};real=evaluation['dev']['real']['rows'];baseline={r['sample_key']:r for r in evaluation['dev']['native']['rows']}
    picks=[]
    for predicate in [lambda r:r['eligible'] and r['raw_success'] and not baseline[r['sample_key']]['raw_success'] and r['type']=='Building',
        lambda r:r['eligible'] and r['raw_success'] and not baseline[r['sample_key']]['raw_success'] and r['type']=='Road',
        lambda r:r['geometry_success'] and not r['fused_success'],
        lambda r:not r['public_projection_supported']]:
        found=[r for r in real if predicate(r) and r['sample_key'] not in picks]
        if found:picks.append(found[0]['sample_key'])
    cases=[]
    with torch.inference_mode():
        for index,key in enumerate(picks):
            r=by_key[key];b=batch_landmarks([r]);s=model(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
            vs=visual_world_beliefs(b['geometry_logits'],b['local_uv'],s,b['valid'],b['projection_supported'])
            n=native(image_features=b['image_features'],name_text=b['name_text'],type_text=b['type_text'],valid=b['valid'])
            ns=visual_world_beliefs(b['geometry_logits'],b['local_uv'],n,b['valid'],b['projection_supported'])
            frame=r['metadata']['frame'];bgr=cv2.imread(str(Path(r['metadata']['observations'])/frame['image']))
            fig,axes=plt.subplots(1,5,figsize=(18,3.7));axes[0].imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
            for contour in r['query']['public_polygons']:
                px=world_to_frame(np.asarray(contour),frame['footprint_world'],bgr.shape);axes[0].plot(px[:,0],px[:,1],'c--',lw=.7)
            axes[0].set_xlim(0,383);axes[0].set_ylim(383,0);axes[0].axis('off');axes[0].set_title('RGB + weak public projection',fontsize=9)
            bounds=r['teacher']['metadata']['world_bounds'];extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
            fields=[('G',b['geometry_logits'][0]),('Native V | observed',ns['raw'][0]),('Trained V | observed',vs['raw'][0]),('G + trained V',vs['fused'][0])]
            for ax,(title,field) in zip(axes[1:],fields):
                values=field.softmax(-1).cpu().numpy().reshape(28,28)
                if '| observed' in title:values=np.where(vs['covered'][0].cpu().numpy().reshape(28,28),values,np.nan)
                ax.imshow(values,origin='lower',extent=extent,cmap='magma')
                for contour in r['query']['public_polygons']:
                    xy=np.array(contour);ax.plot(xy[:,0],xy[:,1],color='cyan',linewidth=.7)
                ax.set_title(title,fontsize=9);ax.set_xlim(bounds[0],bounds[2]);ax.set_ylim(bounds[1],bounds[3])
            fig.suptitle(r['query']['instruction']+' | conditional V outside view is unknown',fontsize=11)
            fig.tight_layout();file=f'case_{index}.png';fig.savefig(a.output/file,dpi=160);plt.close(fig)
            cv2.imwrite(str(a.output/f'case_{index}_rgb.png'),bgr)
            cases.append(dict(file=file,sample_key=key,name=r['query']['name']))
    (a.output/'cases.json').write_text(json.dumps(cases,indent=2))
    page='<!doctype html><meta charset="utf-8"><title>Geometry guided visual belief</title><style>body{max-width:1600px;margin:auto;font-family:sans-serif}img{width:100%}</style><h1>Geometry-Guided Visual Belief：第一阶段</h1><p>全部区域标签为公开Polygon弱监督。V为观测域条件分布，不与全局G直接混同。</p><p><a href="../REPORT_zh.md">中文报告</a> | <a href="analysis.json">统计</a></p><img src="training_and_controls.png">'
    for case in cases:page+=f'<h2>{case["name"]}</h2><a href="{case["file"]}"><img src="{case["file"]}"></a>'
    (a.output/'index.html').write_text(page)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
