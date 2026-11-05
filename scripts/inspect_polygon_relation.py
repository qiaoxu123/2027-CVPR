"""Post-fit TRAIN diagnostics; no model selection or parameter changes."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.direct_target_data import load_model
from graphnav.direct_target_language import PREDICATES
from graphnav.landmark_region_supervision import named_union
from shapely.geometry import Point

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    source=a.source/'training_aligned/best.pt';original=torch.load(source,weights_only=True,map_location='cpu');model=load_model(original).eval().requires_grad_(False)
    saved=torch.load(a.run/'last.pt',weights_only=True,map_location='cpu');model.direct_target.load_state_dict(saved['head']);cache=torch.load(a.run/'features.pt',weights_only=True,map_location='cpu')
    tasks=json.loads((a.source/'inventory_validated/public_tasks.json').read_text());labels=json.loads((a.source/'inventory_validated/private_labels.json').read_text())
    changed={k:float((v.cpu()-original['model'][k]).abs().max()) for k,v in model.state_dict().items() if not torch.equal(v.cpu(),original['model'][k])}
    assert changed and all(k.startswith('direct_target.') for k in changed)
    regions={};probs={};groups={};counter=cache['counter'];natural=cache['natural']
    with torch.no_grad():
        for x in counter:
            if x['subset']!='dev':continue
            ids=torch.tensor([[PREDICATES.index(x['word'])]],device='cuda');valid=torch.ones_like(ids,dtype=torch.bool)
            logits=x['base'].cuda()+model.direct_target.relation(x['dense'][None].cuda(),x['tokens'][None].cuda(),x['mask'][None].cuda(),x['geometry'][None].cuda(),ids,valid)[0]
            r=logits.softmax(-1).cpu().numpy();g=x['base'].softmax(-1).numpy();e=x['prior'].softmax(-1).numpy()
            probs[x['index'],x['word']]=dict(Geometry=g,Explicit=e,Relation=r,Fusion=(r+e)/2)
            regions[x['index'],x['word']]=x['region'].numpy();groups[x['index']]=x['group']
    near=[]
    for idx in groups:
        if not all((idx,w) in probs for w in ('inside','outside','near')):continue
        mask=regions[idx,'near'];row=dict(index=idx,group=groups[idx],metrics={})
        for method in ('Geometry','Explicit','Relation','Fusion'):
            # Same near region, different synchronized text + predicate.
            mass={w:float(probs[idx,w][method][mask].sum()) for w in ('inside','outside','near')}
            row['metrics'][method]=dict(mass=mass,near_minus_outside=mass['near']-mass['outside'],near_minus_inside=mass['near']-mass['inside'])
        near.append(row)
    conflict=[]
    for x in natural:
        t=tasks[x['index']];poly=named_union(t['anchors'][0]['contours']);target=Point(labels[x['index']]['target_world']);d=poly.distance(target)
        conflict.append(dict(index=x['index'],subset=x['subset'],word=x['word'],GT_polygon_distance_m=float(d),GT_inside=poly.covers(target),GT_in_near30=d<=30,region_cells=int(x['region'].sum())))
    stats={}
    for split in ('fit','dev'):
        stats[split]={}
        for w in ('inside','near'):
            rr=[x for x in conflict if x['subset']==split and x['word']==w]
            stats[split][w]=dict(n=len(rr),GT_in_relation=sum(x['GT_inside'] if w=='inside' else x['GT_in_near30'] for x in rr),empty_grid_region=sum(x['region_cells']==0 for x in rr),median_polygon_distance_m=float(np.median([x['GT_polygon_distance_m'] for x in rr])))
    images=[]
    eligible=sorted(idx for idx in groups if all((idx,w) in probs for w in ('inside','outside','near')))
    for idx in eligible[:6]:
        t=tasks[idx];extent=t['world_bounds'];poly=named_union(t['anchors'][0]['contours']);fig,axes=plt.subplots(4,3,figsize=(11,12))
        for i,m in enumerate(('Geometry','Explicit','Relation','Fusion')):
            vmax=max(probs[idx,w][m].max() for w in ('inside','outside','near'))
            for j,w in enumerate(('inside','outside','near')):
                ax=axes[i,j];q=probs[idx,w][m];ax.imshow(q.reshape(28,28),origin='lower',extent=[extent[0],extent[2],extent[1],extent[3]],vmin=0,vmax=vmax)
                for geom in getattr(poly,'geoms',[poly]):
                    if hasattr(geom,'exterior'):ax.plot(*geom.exterior.xy,'c-',lw=1)
                ax.set_title(f'{m} {w} mass={q[regions[idx,w]].sum():.3f}')
        fig.suptitle(f'Counterfactual TRAIN dev query {idx}; cyan=public Polygon, no invented target GT');fig.tight_layout();name=f'counterfactual_{idx}.png';fig.savefig(a.output/name,dpi=110);plt.close(fig);images.append(name)
    result=dict(changed_parameter_maxabs=changed,frozen_modules_unchanged=True,source_sha256=file_sha256(source),checkpoint_SHA256=file_sha256(a.run/'last.pt'),
        relation_GT_compatibility=stats,near_common_n=len(near),near_response={m:dict(rate_vs_outside=float(np.mean([x['metrics'][m]['near_minus_outside']>1e-8 for x in near])),mean_mass_delta_vs_outside=float(np.mean([x['metrics'][m]['near_minus_outside'] for x in near])),mean_mass_delta_vs_inside=float(np.mean([x['metrics'][m]['near_minus_inside'] for x in near]))) for m in ('Geometry','Explicit','Relation','Fusion')},
        caution='near overlaps inside/outside; comparison is distribution response, not antonym correctness')
    atomic_json(result,a.output/'postfit_audit.json');atomic_json(conflict,a.output/'GT_relation_compatibility.json');atomic_json(near,a.output/'near_response.json')
    with (a.output/'REPORT_zh.md').open('a') as f:
        f.write('\n## 事后冲突审查与真实参数更新\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
        f.write('\n几何区域约束不能在同一区域内区分多个目标。均匀区域CE存在抹平目标尖峰的风险，原始语言的near并无固定30m定义；这些是本轮验证的假设，不能用dev结果重新选半径。下一轮优先将区域关系作为排序约束并保留原Geometry区域内部相对概率，而不是扩大网络或把均匀区域当真实目标标签。此建议尚未验证。\n')
        f.write('\n### 地理配准外部证据\n原始PLY头只有XYZ/RGB/class，无CRS。上游公开仓库[坐标问题#63](https://github.com/QingyongHu/SensatUrban/issues/63)及[参考原点#56](https://github.com/QingyongHu/SensatUrban/issues/56)尚未给出可用变换；不能据此反推轴向。方向实验仍被独立元数据阻塞。\n')
    with (a.output/'index.html').open('a') as f:
        f.write('<h2>关系词反事实：每行共享色标</h2>'+''.join(f'<p><a href="{n}"><img width="100%" src="{n}"></a></p>' for n in images))
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
