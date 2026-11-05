"""Report measured TRAIN experiment without touching validation/test datasets."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_region_supervision import named_union

METHODS=('Geometry','Explicit','Relation','Fusion')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((a.run/'natural_predictions.json').read_text());cf=json.loads((a.run/'counterfactual_predictions.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    def summarize(rr):
        out={}
        for method in METHODS:
            correct=sum(not x['metrics']['Geometry']['R20'] and x['metrics'][method]['R20'] for x in rr)
            damage=sum(x['metrics']['Geometry']['R20'] and not x['metrics'][method]['R20'] for x in rr)
            out[method]=dict(n=len(rr),R20=float(np.mean([x['metrics'][method]['R20'] for x in rr])) if rr else None,NE=float(np.mean([x['metrics'][method]['NE'] for x in rr])) if rr else None,mass20=float(np.mean([x['metrics'][method]['mass20'] for x in rr])) if rr else None,correct=correct,damage=damage,net=correct-damage)
        return out
    summary={key:summarize(rr) for key,rr in dict(fit=[x for x in rows if x['subset']=='fit'],dev_all=[x for x in rows if x['subset']=='dev'],dev_supported=[x for x in rows if x['subset']=='dev' and x['supported']],dev_inside=[x for x in rows if x['subset']=='dev' and x['word']=='inside'],dev_near=[x for x in rows if x['subset']=='dev' and x['word']=='near']).items()}
    cache=torch.load(a.run/'features.pt',weights_only=True,map_location='cpu');groups={}
    for x in cf:groups.setdefault(x['index'],{})[x['word']]=x
    cached={(x['index'],x['word']):x for x in cache['counter']}
    pairrows=[]
    for idx,group in groups.items():
        if 'inside' not in group or 'outside' not in group:continue
        masks=cached[idx,'inside']['region'].numpy();row=dict(index=idx,group=group['inside']['group'],methods={})
        for method in METHODS:
            ma=group['inside']['metrics'][method]['mass'];mb=group['outside']['metrics'][method]['mass']
            # Regions are exact complements on same grid: mass_inside(inside)
            # minus mass_inside(outside) == ma + mb -1.
            row['methods'][method]=dict(mass_response=ma+mb-1,correct=ma+mb-1>1e-8,
                centroid_displacement_m=float(np.linalg.norm(np.array(group['inside']['metrics'][method]['centroid'])-group['outside']['metrics'][method]['centroid'])))
        pairrows.append(row)
    summary['counterfactual_IO']={m:dict(n=len(pairrows),response_rate=float(np.mean([x['methods'][m]['correct'] for x in pairrows])),mass_response=float(np.mean([x['methods'][m]['mass_response'] for x in pairrows])),centroid_displacement_m=float(np.mean([x['methods'][m]['centroid_displacement_m'] for x in pairrows]))) for m in METHODS}
    # Near and outside overlap: don't call them antonyms. Compare near mass
    # to the inside/outside mean over the SAME near mask using cached logits.
    summary['near_region_mass']={m:float(np.mean([x['metrics'][m]['mass'] for x in cf if x['word']=='near'])) for m in METHODS}
    dev=[x for x in rows if x['subset']=='dev'];unique=sorted({x['group'] for x in dev});rng=np.random.default_rng(42);interval={}
    for m in METHODS:
        samples=[]
        for _ in range(2000):
            picked=rng.choice(unique,len(unique),replace=True);rr=[x for g in picked for x in dev if x['group']==g]
            samples.append(np.mean([int(x['metrics'][m]['R20'])-int(x['metrics']['Geometry']['R20']) for x in rr]))
        interval[m]=np.percentile(samples,[2.5,97.5]).tolist()
    summary['dev_grouped_bootstrap_R20_delta_95CI']=interval
    atomic_json(summary,a.output/'summary.json');atomic_json(pairrows,a.output/'IO_pairs.json')
    for name in ('natural_predictions.json','counterfactual_predictions.json','manifest.json','history.json'):(a.output/name).write_bytes((a.run/name).read_bytes())
    beliefs=np.load(a.run/'beliefs.npz');tasks=json.loads((a.source/'inventory_validated/public_tasks.json').read_text());labels=json.loads((a.source/'inventory_validated/private_labels.json').read_text())
    # Explicitly include both gains and damage cases; deterministic index order.
    selected=[]
    for kind,condition in [('gain',lambda x:not x['metrics']['Geometry']['R20'] and x['metrics']['Fusion']['R20']),('damage',lambda x:x['metrics']['Geometry']['R20'] and not x['metrics']['Fusion']['R20']),('other',lambda x:True)]:
        for x in dev:
            if x['supported'] and condition(x) and x['index'] not in [r['index'] for r in selected]:selected.append(dict(x,case_type=kind))
            if sum(r['case_type']==kind for r in selected)>=3:break
    selected=selected[:9];images=[]
    for row in selected:
        t=tasks[row['index']];poly=named_union(t['anchors'][0]['contours']);target=labels[row['index']]['target_world'];extent=t['world_bounds'];fig,axes=plt.subplots(1,4,figsize=(16,4))
        for ax,m in zip(axes,METHODS):
            p=beliefs[f'{row["index"]}_{m}'].reshape(28,28);ax.imshow(p,origin='lower',extent=[extent[0],extent[2],extent[1],extent[3]])
            for geom in getattr(poly,'geoms',[poly]):
                if hasattr(geom,'exterior'):ax.plot(*geom.exterior.xy,'c-',lw=1)
            ax.scatter(*target,c='red',marker='x');ax.set_title(f'{m} NE={row["metrics"][m]["NE"]:.1f}m')
        fig.suptitle(f'{row["index"]} {row["word"]} {row["case_type"]}: '+row['text'][:110],fontsize=9);fig.tight_layout();name=f'belief_{row["index"]}.png';fig.savefig(a.output/name,dpi=130);plt.close(fig);images.append(name)
    hist=json.loads((a.run/'history.json').read_text());plt.figure();plt.plot([x['epoch'] for x in hist],[x['loss'] for x in hist],marker='o');plt.xlabel('Epoch');plt.ylabel('Combined TRAIN loss');plt.tight_layout();plt.savefig(a.output/'loss.png',dpi=130);plt.close()
    lines=['# Polygon显式先验与轻量关系训练（TRAIN）','', '## 结论范围','仅TRAIN、FP32、seed42；不改普通推理，不运行Val/Test或闭环。方向地理配准未完成，所有地理方向弃权，不宣称方向理解完成。','', '## 数据与训练',json.dumps(manifest,ensure_ascii=False,indent=2),'','## 目标定位（相对冻结Geometry）','|分组|方法|N|R20|NE(m)|GT20m质量|纠正|破坏|净收益|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('dev_all','dev_supported','dev_inside','dev_near'):
        for m,d in summary[group].items():
            if not d['n']:continue
            lines.append(f'|{group}|{m}|{d["n"]}|{100*d["R20"]:.2f}%|{d["NE"]:.4f}|{d["mass20"]:.5f}|{d["correct"]}|{d["damage"]}|{d["net"]:+d}|')
    lines+=['','## 反事实区域响应（独立地图组TRAIN dev）','|方法|IO共同对数|联合响应率|inside质量差|质心位移m|','|---|---:|---:|---:|---:|']
    for m,d in summary['counterfactual_IO'].items():lines.append(f'|{m}|{d["n"]}|{100*d["response_rate"]:.2f}%|{d["mass_response"]:.5f}|{d["centroid_displacement_m"]:.3f}|')
    fusion=summary['dev_all']['Fusion'];relation=summary['dev_all']['Relation']
    lines+=['','inside/outside是拓扑变换，没有单一正确位移方向；质心移动大小不作为方向正确率。near与outside有重叠，不能人为当成反义词。','', '## 决策与限制',f'Fusion相对Geometry纠正{fusion["correct"]}、破坏{fusion["damage"]}，净收益{fusion["net"]:+d}；Relation净收益{relation["net"]:+d}。', '不自动开展56条Val闭环。区域响应改善并不等于真实目标识别，需先检查目标定位净收益、NE及跨地标不确定性。', 'dev按地图与公开地标ID隔离，但此前已用于诊断，非新的未触碰确认组；Geometry历史预训练也可能用过TRAIN，不能称全模型未见泛化。', '反事实没有目标GT；只使用公开Polygon构造区域标签。inside/outside/near的区域监督是几何弱监督，不是人工像素标注。真实目标GT仅用于原始fit指令训练与dev事后评价。', '原始权重保留不变；只训练现有Relation组件，不训练RGB、PGVB、Geometry、Stop或StableGoal。', '未进行地理方向配准：24地图TIF缺少CRS；公开对象表只有本地XY，官方rasterize.py未提供局部到地理的旋转与原点。需要官方地理元数据或至少两个可靠非GT地理控制点。不得根据目标GT翻转坐标。', '', '### 地标组bootstrap R20差95%区间',json.dumps(interval,indent=2),'','## 可视化','图中青色为完整公开Polygon、红叉为目标GT，仅用于离线评分展示；各方法共享世界坐标与28×28网格。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>TRAIN Polygon Relation</title><h1>TRAIN Polygon关系Belief</h1><p>地图弱监督；方向地理配准未完成；无闭环SR。</p><a href="REPORT_zh.md">中文报告</a><img width="600" src="loss.png">'+''.join(f'<p><a href="{n}"><img width="100%" src="{n}"></a></p>' for n in images))
    print(json.dumps(summary,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
