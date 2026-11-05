"""Paired weak-proxy rankings and a full-swap Polygon/RGB audit gallery."""
import argparse
from collections import Counter,defaultdict
import csv
import json
from pathlib import Path
import shutil
import cv2
import numpy as np
from graphnav.landmark_projection import world_to_frame
from scripts.polygon_rgb_matching import rank_credit

METHODS=('geometry','pgvb','pgvb_zero_shape_tokens','explicit','appearance','explicit_reference')


def evaluate(rows,family,methods,fully_visible=False):
    records=[]
    for row in rows:
        c=[x for x in row['candidates'] if x['family']in ('correct',family)]
        if len(c)<2:continue
        if fully_visible and any(x['visible_fraction']<1-1e-6 for x in c):continue
        if any(x[m]is None for x in c for m in methods):continue
        if family!='real_other_instance':c=[c[0]]+[x for x in c[1:] if x['roi_mask_sha256']!=c[0]['roi_mask_sha256']]
        if len(c)<2:continue
        records.append(dict(key=row['key'],index=row['index'],kind=row['kind'],height=row['height_agl_m'],
            visible_fraction=c[0]['visible_fraction'],area_m2=c[0]['local_area_m2'],candidate_count=len(c),
            rankings={m:rank_credit([x[m] for x in c]) for m in methods}))
    metrics={m:dict(top1=float(np.mean([r['rankings'][m]['top1'] for r in records])),
                    mean_rank=float(np.mean([r['rankings'][m]['rank'] for r in records])),
                    mean_margin=float(np.mean([r['rankings'][m]['margin'] for r in records]))) for m in methods} if records else {}
    return dict(n=len(records),tasks=len({r['index'] for r in records}),mean_candidates=float(np.mean([r['candidate_count'] for r in records])) if records else None,metrics=metrics,records=records)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','nearfield','rollout','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'frames').mkdir(exist_ok=True)
    rows=list(map(json.loads,(a.run/'results.jsonl').open()))
    base_methods=METHODS[:4];families=('real_other_instance','position_shift','shape_rotation','shape_scale')
    groups={f:evaluate(rows,f,base_methods) for f in families}
    reference_groups={f:evaluate(rows,f,METHODS) for f in families}
    fully_visible=evaluate(rows,'real_other_instance',METHODS,True)
    token_changes=[c['shape_token_score_MAE'] for r in rows for c in r['candidates']]
    eligible=[r for r in rows if r['eligibility']=='eligible'];references=[r for r in eligible if r['reference'] and r['reference_cosine_correct']is not None]
    strata={}
    for key in ('Building','Road','height_0_20','height_20_40','height_40_plus','partial_visible','mostly_visible','small_ROI','medium_ROI','large_ROI'):
        selected=[]
        for r in rows:
            if not r['candidates']:continue
            px_area=r['candidates'][0]['local_area_m2']/abs(cv2.contourArea(np.array(r['footprint_world'],np.float32)))*384**2
            keep=(r['kind']==key if key in ('Building','Road') else
                  r['height_agl_m']<20 if key=='height_0_20' else 20<=r['height_agl_m']<40 if key=='height_20_40' else r['height_agl_m']>=40 if key=='height_40_plus' else
                  r['candidates'][0]['visible_fraction']<.75 if key=='partial_visible' else r['candidates'][0]['visible_fraction']>=.75 if key=='mostly_visible' else
                  px_area<1024 if key=='small_ROI' else 1024<=px_area<8192 if key=='medium_ROI' else px_area>=8192)
            if keep:selected.append(r)
        strata[key]=evaluate(selected,'real_other_instance',METHODS)
    pairwise={}
    for family in families[1:]:
        pairs=[]
        for r in eligible:
            A=r['candidates'][0]
            for c in r['candidates'][1:]:
                if c['family']!=family or c['roi_mask_sha256']==A['roi_mask_sha256'] or any(A[m]is None or c[m]is None for m in base_methods):continue
                pairs.append(dict(key=r['key'],id=c['id'],**{m:A[m]-c[m] for m in base_methods}))
        pairwise[family]=dict(pairs=len(pairs),methods={m:dict(mean_score_delta=float(np.mean([x[m] for x in pairs])),pair_win_fraction=float(np.mean([1 if x[m]>1e-8 else .5 if abs(x[m])<=1e-8 else 0 for x in pairs]))) for m in base_methods} if pairs else {})
    # Target-group bootstrap of paired differences, not four frames as independent instances.
    paired_ci={}
    for comparator in ('geometry','pgvb','explicit'):
        per_task=defaultdict(list)
        for r in reference_groups['real_other_instance']['records']:
            per_task[r['index']].append(r['rankings']['explicit_reference']['top1']-r['rankings'][comparator]['top1'])
        values=np.array([np.mean(v) for v in per_task.values()])
        rng=np.random.default_rng(42);boot=rng.choice(values,(10000,len(values)),replace=True).mean(1) if len(values) else np.array([np.nan])
        paired_ci[comparator]=dict(task_groups=len(values),task_balanced_top1_difference=float(values.mean()) if len(values) else None,CI95=np.quantile(boot,[.025,.975]).tolist())
    summary=dict(total_frames=len(rows),eligibility=dict(Counter(r['eligibility'] for r in rows)),eligible_tasks=len({r['index'] for r in eligible}),
        true_pixel_IoU=None,true_instance_accuracy=None,true_crossview_identity_accuracy=None,
        base_rankings={f:{k:v for k,v in x.items() if k!='records'} for f,x in groups.items()},
        common_reference_rankings={f:{k:v for k,v in x.items() if k!='records'} for f,x in reference_groups.items()},
        fully_visible_common_reference={k:v for k,v in fully_visible.items() if k!='records'},
        strata={f:{k:v for k,v in x.items() if k!='records'} for f,x in strata.items()},
        perturbation_pairwise=pairwise,task_group_bootstrap=paired_ci,
        shape_token_sensitivity=dict(candidate_evaluations=len(token_changes),mean_patch_score_MAE=float(np.mean(token_changes)),max_patch_score_MAE=float(max(token_changes)),
            note='zero Polygon shape tokens keeps current ROI and RGB fixed; this single-factor ablation is NOT the full-swap negative'),
        causal_references=dict(n=len(references),tasks=len({r['index'] for r in references}),mean_same_ID_cosine=float(np.mean([r['reference_cosine_correct'] for r in references])),
            median_pose_translation_m=float(np.median([np.linalg.norm(np.array(r['pose'][:2])-r['reference']['pose'][:2]) for r in references])),
            median_yaw_change_degrees=float(np.median([abs((r['pose'][3]-r['reference']['pose'][3]+np.pi)%(2*np.pi)-np.pi)*180/np.pi for r in references])),
            identical_RGB_sha_pairs=sum(r['rgb_sha256']==r['reference']['rgb_sha256'] for r in references),
            median_common_observed_fraction=float(np.median([r['reference_visible_overlap_fraction'] for r in references])),
            orthophoto_caveat='Same orthophoto textures may recur; this is not real 3D cross-view generalization'),
        excluded_perturbations=sum(len(r.get('excluded_perturbations',[])) for r in rows),
        no_observable_ROI_change_controls=sum(c['family']not in ('correct','real_other_instance') and c['roi_mask_sha256']==r['candidates'][0]['roi_mask_sha256'] for r in eligible for c in r['candidates']),
        method_policy='All comparisons are public-map weak proxies; paired methods share candidate sets and query frames; fractional Top1 under ties')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (a.output/'ranking_records.json').write_text(json.dumps(dict(base=groups,common_reference=reference_groups),indent=2)+'\n')
    shutil.copy2(a.run/'results.jsonl',a.output/'results.jsonl')
    for name in ('manifest.json','runtime.json'):shutil.copy2(a.run/name,a.output/name)
    with (a.output/'scores.csv').open('w') as f:
        fields=['key','index','kind','candidate_id','family','visible_fraction','local_area_m2',*METHODS]
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader()
        for r in rows:
            for c in r['candidates']:w.writerow(dict(key=r['key'],index=r['index'],kind=r['kind'],candidate_id=c['id'],family=c['family'],visible_fraction=c['visible_fraction'],local_area_m2=c['local_area_m2'],**{m:c[m] for m in METHODS}))
    atlas=[]
    for r in eligible:
        im=cv2.imread(str(a.nearfield/'frames'/f'{r["key"]}.png'));cv2.imwrite(str(a.output/'frames'/f'{r["key"]}.png'),im)
        edges=cv2.Canny(cv2.cvtColor(im,cv2.COLOR_BGR2GRAY),100,200);cv2.imwrite(str(a.output/'frames'/f'{r["key"]}_edges.png'),edges)
        display=[]
        for c in r['candidates']:
            px=[world_to_frame(np.array(poly),r['footprint_world'],im.shape).tolist() for poly in c['contours']]
            display.append({**{k:v for k,v in c.items() if k not in ('contours','positive_candidate_uv')},'px':px})
        ref=None
        if r['reference']:
            rr=r['reference'];root=a.rollout/'decisions'/f'{r["index"]:02d}_StableTrackedPGVB'/f'step_{rr["step"]:02d}'
            ref_key=f'ref_{r["index"]:02d}_{rr["step"]:02d}';shutil.copy2(root.with_suffix('.png'),a.output/'frames'/f'{ref_key}.png')
            fp=np.load(root.with_suffix('.npz'))['footprint_world'];ref=dict(key=ref_key,step=rr['step'],pose=rr['pose'],px=[world_to_frame(np.array(poly),fp,im.shape).tolist() for poly in r['candidates'][0]['contours']])
        atlas.append(dict(key=r['key'],index=r['index'],name=r['query_name'],kind=r['kind'],height=r['height_agl_m'],candidates=display,reference=ref))
    (a.output/'atlas_data.js').write_text('const MATCHES='+json.dumps(atlas,ensure_ascii=False,separators=(',',':'))+';\n')
    (a.output/'index.html').write_text(HTML)
    figures(eligible,reference_groups,a.nearfield,a.rollout,a.output)
    print(json.dumps(summary,indent=2))


def figures(rows,groups,nearfield,rollout,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Deterministic coverage of available categories and successes/failures, all rows remain in HTML.
    real=[r for r in rows if any(c['family']=='real_other_instance' for c in r['candidates'])]
    chosen=[]
    for kind in ('Building','Road'):
        candidates=[r for r in real if r['kind']==kind]
        if candidates:chosen.extend(candidates[:3]+candidates[-2:])
    unique={r['key']:r for r in chosen}
    for r in unique.values():
        im=plt.imread(nearfield/'frames'/f'{r["key"]}.png');fig,axes=plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
        for ax in axes:ax.set_xlim(0,383);ax.set_ylim(383,0)
        axes[0].imshow(im);axes[1].imshow(cv2.Canny(cv2.cvtColor(cv2.imread(str(nearfield/'frames'/f'{r["key"]}.png')),cv2.COLOR_BGR2GRAY),100,200),cmap='gray')
        A=r['candidates'][0];B=next(c for c in r['candidates'] if c['family']=='real_other_instance')
        for ax in axes[:2]:
            for candidate,color in ((A,'cyan'),(B,'magenta')):
                for c in candidate['contours']:
                    px=world_to_frame(np.array(c),r['footprint_world'],im.shape);ax.plot(px[:,0],px[:,1],color=color,lw=1.5)
        axes[0].set_title('Current RGB: A cyan, real B magenta');axes[1].set_title('Canny edges (visualization only)')
        if r['reference']:
            ref=r['reference'];root=rollout/'decisions'/f'{r["index"]:02d}_StableTrackedPGVB'/f'step_{ref["step"]:02d}'
            rim=plt.imread(root.with_suffix('.png'));fp=np.load(root.with_suffix('.npz'))['footprint_world'];axes[2].imshow(rim)
            for c in A['contours']:
                px=world_to_frame(np.array(c),fp,rim.shape);axes[2].plot(px[:,0],px[:,1],color='cyan')
            axes[2].set_title(f'Past reference A, step {ref["step"]}')
        fig.suptitle(f'{r["key"]} {r["query_name"]}\nA/B PGVB {A["pgvb"]:.3f}/{B["pgvb"]:.3f}; explicit {A["explicit"]}/{B["explicit"]}',fontsize=9)
        fig.savefig(out/f'case_{r["key"]}.png',dpi=120);plt.close(fig)
    group=groups['real_other_instance'];fig,ax=plt.subplots(figsize=(10,4),constrained_layout=True)
    ax.bar(list(group['metrics']),[v['top1']*100 for v in group['metrics'].values()]);ax.set_ylabel('Public-map identity Top1 proxy (%)');ax.set_ylim(0,100);ax.tick_params(axis='x',rotation=20)
    ax.set_title(f'Common causal-reference subset: {group["n"]} queries / {group["tasks"]} task groups')
    fig.savefig(out/'ranking_summary.png',dpi=130);plt.close(fig)


HTML='''<!doctype html><meta charset="utf-8"><title>Polygon与RGB完整匹配对照</title><style>body{font:15px system-ui;margin:20px}canvas{width:31%;max-width:500px;background:#222}table{border-collapse:collapse;font-size:12px}td,th{border:1px solid #aaa;padding:5px}button,select{font:inherit;margin:4px}pre{white-space:pre-wrap}</style>
<h1>Polygon—RGB完整匹配与跨帧审查</h1><p>全部为公开地图弱监督代理，非真实像素IoU/实例识别准确率。另一真实Polygon B也可能正确贴合RGB；“配准贴合”不等于“指定A身份”。<a href="REPORT_zh.md">中文报告</a> · <a href="../nearfield_scene_audit/index.html">全部141帧（含未知）</a></p>
<button onclick="show(Math.max(0,current-1))">上一条</button><select id="samples" onchange="show(this.selectedIndex)"></select><button onclick="show(Math.min(MATCHES.length-1,current+1))">下一条</button>
<select id="candidate" onchange="draw()"></select><label><input id="tokens" type="checkbox" onchange="draw()">显示形状Token采样点</label>
<h2 id="title"></h2><canvas id="raw" width="384" height="384"></canvas><canvas id="edges" width="384" height="384"></canvas><canvas id="reference" width="384" height="384"></canvas>
<p>左：当前RGB；中：同一RGB边缘（Canny100/200，仅可视化；评分用Sobel）；右：严格过去帧的A参考。青色A，品红为选中候选。每个候选同步更新Polygon Token、ROI及候选区域。</p><div id="scores"></div><pre id="details"></pre>
<script src="atlas_data.js"></script><script>
let current=0;const select=document.getElementById('samples');MATCHES.forEach(r=>{let o=document.createElement('option');o.textContent=r.key+' '+r.name;select.append(o)});
function poly(ctx,points,color,width=2){ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke()}
function show(i){current=i;select.selectedIndex=i;const r=MATCHES[i],c=document.getElementById('candidate');c.innerHTML='';r.candidates.forEach(x=>{let o=document.createElement('option');o.textContent=x.id+' / '+x.family;c.append(o)});document.getElementById('title').textContent=r.key+' '+r.name+' / '+r.kind+' / '+r.height.toFixed(1)+'m';draw()}
function draw(){let r=MATCHES[current],which=document.getElementById('candidate').selectedIndex,c=r.candidates[which],snapshot=current;
['raw','edges'].forEach((id,k)=>{let im=new Image();im.onload=()=>{if(snapshot!==current)return;let ctx=document.getElementById(id).getContext('2d');ctx.clearRect(0,0,384,384);ctx.drawImage(im,0,0);r.candidates[0].px.forEach(p=>poly(ctx,p,'cyan'));if(which)c.px.forEach(p=>poly(ctx,p,'magenta'));if(document.getElementById('tokens').checked)c.polygon_tokens.forEach(p=>{ctx.fillStyle='yellow';ctx.beginPath();ctx.arc(p[0]*383,p[1]*383,2,0,2*Math.PI);ctx.fill()})};im.src='frames/'+r.key+(k?'_edges':'')+'.png'});
let refctx=document.getElementById('reference').getContext('2d');refctx.clearRect(0,0,384,384);if(r.reference){let im=new Image();im.onload=()=>{if(snapshot!==current)return;refctx.drawImage(im,0,0);r.reference.px.forEach(p=>poly(refctx,p,'cyan'))};im.src='frames/'+r.reference.key+'.png'}
let keys=['geometry','pgvb','pgvb_zero_shape_tokens','explicit','appearance','explicit_reference'];document.getElementById('scores').innerHTML='<table><tr><th>候选</th>'+keys.map(k=>'<th>'+k+'</th>').join('')+'</tr>'+r.candidates.map((x,j)=>'<tr><td>'+j+' '+x.family+'</td>'+keys.map(k=>'<td>'+(x[k]===null?'NA':x[k].toFixed(4))+'</td>').join('')+'</tr>').join('')+'</table>';
document.getElementById('details').textContent=JSON.stringify({candidate_id:c.id,candidate_name:c.candidate_name,query_name:c.query_name,public_ids:c.public_ids,visible_fraction:c.visible_fraction,local_area_m2:c.local_area_m2,positive_patches:c.positive_patches,negative_patches:c.negative_patches,roi_mask_sha256:c.roi_mask_sha256,explicit_components:c.explicit_components,reference:r.reference},null,2)}
show(0);
</script>'''

if __name__=='__main__':main()
