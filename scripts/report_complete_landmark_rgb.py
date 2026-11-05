"""Post-seal report: public geometry/Beliefs; licensed RGB gallery stays local."""
import argparse,json,html,shutil,collections
from pathlib import Path
import cv2
from graphnav.landmark_projection import world_to_frame
import numpy as np
import torch
import shapely
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser()
    for k in ('run','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);r=a.run
    metrics=json.load(open(r/'evaluation/metrics.json'));qs=json.load(open(r/'inventory_all_public_parts/queries_public.json'));groups=json.load(open(r/'inventory_all_public_parts/groups.json'));labels=json.load(open(r/'inventory_all_public_parts/labels_private.json'));paired=json.load(open(r/'evaluation/paired.json'));train=json.load(open(r/'training/complete.json'));cache=json.load(open(r/'cache/complete.json'))
    for src,dst in [(r/'evaluation/metrics.json','metrics.json'),(r/'evaluation/paired.json','paired.json'),(r/'training/config.json','training_config.json'),(r/'training/complete.json','training_log.json'),(r/'cache/complete.json','cache_statistics.json')]:shutil.copyfile(src,a.output/dst)
    hist=train['history'];fig,ax=plt.subplots(1,2,figsize=(10,3));ax[0].plot([x['step'] for x in hist],[x['target_CE'] for x in hist],label='Target CE');ax[0].legend();ax[1].plot([x['step'] for x in hist],[x['KL'] for x in hist],label='Preservation KL');ax[1].legend();fig.tight_layout();fig.savefig(a.output/'training_curve.png',dpi=140);plt.close(fig)
    names=['geometry_relation','old_current','old_whole','old_multi','new_current','new_whole','new_multi','new_multi_zero','new_multi_shuffle'];fig,ax=plt.subplots(1,2,figsize=(13,5));ax[0].bar(range(len(names)),[100*metrics['methods'][n]['R20'] for n in names]);ax[1].bar(range(len(names)),[metrics['methods'][n]['NE_m'] for n in names]);ax[0].set_ylabel('Goal R20 (%)');ax[1].set_ylabel('Goal NE (m)')
    for x in ax:x.set_xticks(range(len(names)),names,rotation=65,ha='right')
    fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=140);plt.close(fig)
    # Aggregate validity without interpreting map projections as real pixel masks.
    current=collections.Counter();valid_current=[]
    for q in qs:
        v=torch.load(r/'cache/queries'/f'{q["index"]:05d}.pt',weights_only=True,map_location='cpu');current[v['current_reason']]+=1;valid_current.extend(s['valid_patches'] for s in v['current_stats'])
    views=[];groupstats={}
    for key in groups:
        v=torch.load(r/'cache/groups'/f'{key}.pt',weights_only=True,map_location='cpu');views.extend(v['stats']);groupstats[key]=v['stats']
    data=dict(all_queries=len(qs),current_reasons=dict(current),privileged_unique_images=len(views),privileged_valid_patches=sum(x['valid_patches'] for x in views),privileged_possible_patches=144*len(views),mean_source_valid_fraction=float(np.mean([x['source_valid_fraction'] for x in views])),fully_usable_views=sum(x['valid_patches']==144 for x in views),zero_valid_views=sum(x['valid_patches']==0 for x in views),meters_per_pixel_quantiles=np.quantile([x['meters_per_pixel'] for x in views],[0,.25,.5,.75,1]).tolist(),labels_are_pixel_GT=False)
    atomic_json(data,a.output/'coverage.json')
    # Pair-selected successes AND regressions, plus unchanged category-diverse cases.
    corrected=[x for x in paired if x['methods']['new_multi']['R20'] and not x['methods']['geometry_relation']['R20']];damaged=[x for x in paired if not x['methods']['new_multi']['R20'] and x['methods']['geometry_relation']['R20']]
    selected=corrected[:4]+damaged[:4];seen={x['index'] for x in selected}
    for x in paired:
        if len(selected)>=12:break
        if x['index'] not in seen and x['group']:selected.append(x);seen.add(x['index'])
    methods=['geometry_relation','old_current','old_multi','new_multi','new_multi_zero'];pred={n:torch.load(r/f'evaluation/{n}.pt',weights_only=True,map_location='cpu') for n in methods};lookup={idx:i for i,idx in enumerate(pred[methods[0]]['ids'])};cases=[]
    for item in selected:
        idx=item['index'];q=qs[idx];g=groups.get(q['group'])
        if not g:continue
        b=np.array(g['world_bounds']);target=np.array(labels[idx]['target_world']);pos=lookup[idx];fig,ax=plt.subplots(1,len(methods),figsize=(18,4))
        for x,n in zip(ax,methods):
            prob=pred[n]['logits'][pos].softmax(-1).reshape(28,28).numpy();x.imshow(prob,origin='lower',extent=[b[0],b[2],b[1],b[3]],vmin=0,vmax=max(float(pred[k]['logits'][pos].softmax(-1).max()) for k in methods))
            for c in g['contours']:c=np.array(c);x.plot(*np.vstack((c,c[0])).T,'c-',lw=.6)
            x.plot(*target,'r*',ms=8);t=int(pred[n]['logits'][pos].argmax());xy=b[:2]+np.array([t%28+.5,t//28+.5])/28*(b[2:]-b[:2]);x.plot(*xy,'wx',ms=6);x.set_title(f"{n}\nNE={item['methods'][n]['NE_m']:.1f}m",fontsize=9);x.set_aspect('equal')
        fig.suptitle(q['instruction'],fontsize=8,wrap=True);fig.tight_layout();fname=f'belief_{idx:05d}.png';fig.savefig(a.output/fname,dpi=130);plt.close(fig);cases.append(dict(index=idx,instruction=q['instruction'],figure=fname,methods=item['methods']))
    # Sampling geometry example uses public polygons, no raw source crop on GitHub.
    g=groups[selected[0]['group']] if selected else next(iter(groups.values()));fig,ax=plt.subplots(figsize=(7,7))
    for c in g['contours']:c=np.array(c);ax.fill(*c.T,alpha=.3,color='blue')
    for v in [g['plan']['whole']]+g['plan']['tiles']:
        c=np.asarray(v['footprint_world']);ax.plot(*np.vstack((c,c[0])).T,lw=.4,color='gray')
    ax.set_aspect('equal');ax.set_title('Public Polygon whole/context + multiscale sampling');fig.tight_layout();fig.savefig(a.output/'sampling_plan.png',dpi=140);plt.close(fig)
    atomic_json(cases,a.output/'cases.json')
    rows=''.join(f"<tr><td>{n}</td><td>{v['R20']*100:.3f}%</td><td>{v['NE_m']:.4f}</td><td>{v['corrections']}/{v['damage']}</td><td>{v['net']:+d}</td></tr>" for n,v in metrics['methods'].items())
    body='<meta charset="utf-8"><style>body{max-width:1400px;margin:auto;font:16px sans-serif}img{max-width:100%}td,th{padding:8px;border:1px solid #ddd}</style><h1>TRAIN特权完整地标RGB实验</h1><p>位置代理指标，非像素实例识别。非闭环SR。源正射RGB图册仅保存在服务器。</p><table><tr><th>方法</th><th>R20</th><th>NE(m)</th><th>纠正/破坏</th><th>净收益</th></tr>'+rows+'</table>'
    for f in ['comparison.png','training_curve.png','sampling_plan.png']:body+=f'<p><a href="{f}"><img src="{f}"></a></p>'
    for c in cases:body+=f'<h3>TRAIN index {c["index"]}</h3><p>{html.escape(c["instruction"])}</p><a href="{c["figure"]}"><img src="{c["figure"]}"></a>'
    (a.output/'index.html').write_text(body)
    # Local-only visual comparison, original images are not copied into Git.
    local=r/'LOCAL_RGB_GALLERY';local.mkdir(exist_ok=True);parts=[body]
    for picture in a.output.glob('*.png'):shutil.copyfile(picture,local/picture.name)
    for key in list(groups)[:12]:
        source=r/'cache/LOCAL_PRIVILEGED_IMAGES'/key
        if source.exists():
            parts.append(f'<h2>{html.escape(groups[key]["map_name"])} / {html.escape(str(groups[key]["names"]))}</h2>')
            for image in sorted(source.glob('*.png')):
                dest=local/f'{key}_{image.name}';rgb=cv2.imread(str(image));view_id=int(image.stem.split('_')[-1]);view=([groups[key]['plan']['whole']]+groups[key]['plan']['tiles'])[view_id]
                for contour in groups[key]['contours']:
                    pixels=world_to_frame(np.array(contour),view['footprint_world'],rgb.shape);cv2.polylines(rgb,[np.rint(pixels).astype(np.int32)],True,(0,255,255),1)
                cv2.imwrite(str(dest),rgb);parts.append(f'<a href="{dest.name}"><img width="384" src="{dest.name}"></a>')
    (local/'index.html').write_text(''.join(parts))
    # R20 evidence gate is not a formal significance claim (one seed).
    m=metrics['methods'];passed=m['new_multi']['R20']>max(m[x]['R20'] for x in ('geometry_relation','new_multi_zero','new_multi_shuffle')) and m['new_multi']['NE_m']<m['geometry_relation']['NE_m'] and m['new_multi']['net']>0
    atomic_json(dict(clear_point_estimate_RGB_gain=passed,formal_significance_not_claimed=True,allow_privileged_navigation=False,next_step='Design separate causal current/history model before any Val closed loop' if passed else 'Do not expand to Val; diagnose weak target-instance conditioning and visual evidence'),a.output/'decision.json')
    print(json.dumps(dict(coverage=data,decision=passed),indent=2))

if __name__=='__main__':main()
