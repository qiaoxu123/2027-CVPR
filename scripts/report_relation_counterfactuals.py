"""Paired whole-distribution metrics, anchor-cluster uncertainty and real-map heatmaps."""
import argparse
from collections import defaultdict
import html
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.relation_counterfactual_metrics import PAIRS,probabilities


def cluster_ci(rows,branch,metric):
    groups=defaultdict(list)
    for r in rows:groups[r['group']].append(float(r['metrics'][branch][metric]))
    if len(groups)<2:return None
    values=[np.array(v) for v in groups.values()];rng=np.random.default_rng(42);out=[]
    for _ in range(1000):
        take=rng.integers(0,len(values),len(values));out.append(np.concatenate([values[i] for i in take]).mean())
    return [float(x) for x in np.quantile(out,[.025,.975])]


def aggregate(rows):
    out=dict(queries=len(rows),anchor_groups=len({r['group'] for r in rows}))
    for branch in ('Geometry','Relation','Effective'):
        vals=[r['metrics'][branch] for r in rows]
        metric=dict(joint_mass_rate=float(np.mean([v['joint_mass_correct'] for v in vals])) if vals else None,
            joint_mass_CI=cluster_ci(rows,branch,'joint_mass_correct') if vals else None,
            both_top1_regions_rate=float(np.mean([v['top1_a_region'] and v['top1_b_region'] for v in vals])) if vals else None,
            mean_mass_change_a=float(np.mean([v['mass_change_a'] for v in vals])) if vals else None,
            mean_mass_change_b=float(np.mean([v['mass_change_b'] for v in vals])) if vals else None,
            mean_centroid_distance_m=float(np.mean([v['centroid_distance_m'] for v in vals])) if vals else None,
            mean_TV=float(np.mean([v['total_variation'] for v in vals])) if vals else None)
        if vals and vals[0]['centroid_direction_correct'] is not None:
            metric.update(centroid_direction_rate=float(np.mean([v['centroid_direction_correct'] for v in vals])),
                centroid_direction_CI=cluster_ci(rows,branch,'centroid_direction_correct'),
                mean_signed_shift_m=float(np.mean([v['signed_centroid_shift_m'] for v in vals])))
        else:metric.update(centroid_direction_rate=None,mean_signed_shift_m=None)
        out[branch]=metric
    return out


def main():
    p=argparse.ArgumentParser()
    for key in ('run','source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True)
    rows=json.loads((a.run/'pairs.json').read_text());manifest=json.loads((a.run/'manifest.json').read_text())
    tasks=json.loads((a.source/'inventory_validated/public_tasks.json').read_text())
    coordinate=json.loads((a.run/'coordinate_audit.json').read_text());synth=json.loads((a.run/'synthetic_control.json').read_text())
    summary=dict(primary={},native_only={},by_subset={},quality={},metric_scope='Engineering-axis counterfactual consistency, NOT geographic accuracy/Goal R20/SR')
    for pair,(wa,wb) in PAIRS.items():
        allrows=[r for r in rows if r['pair']==pair]
        valid=[r for r in allrows if r['single_ID'] and r['relation_in_token_window'] and r['token_IDs_changed'] and r['metrics']['Geometry']['valid_regions']]
        summary['primary'][pair]=aggregate(valid)
        native=[r for r in valid if tasks[r['index']]['anchors'][0]['predicate'] in (wa,wb)]
        summary['native_only'][pair]=aggregate(native)
        summary['quality'][pair]=dict(all_queries=len(allrows),multiple_public_IDs=sum(not r['single_ID'] for r in allrows),
            language_truncated=sum(not r['relation_in_token_window'] for r in allrows),
            no_region_grid_support=sum(not r['metrics']['Geometry']['valid_regions'] for r in allrows),
            missing_explicit_compiler_rule=sum(not any(word in r['compiler_relations'][word] for word in (wa,wb)) for r in allrows),
            compiler_identical_rules=sum(r['compiler_rule_difference']<1e-9 for r in allrows))
        summary['by_subset'][pair]={subset:aggregate([r for r in valid if r['subset']==subset]) for subset in ('fit','dev')}
    atomic_json(summary,o/'summary.json')
    for name in ('coordinate_audit.json','synthetic_control.json','manifest.json','pairs.json','skipped.json'):
        shutil.copy2(a.run/name,o/name)
    synthetic=np.load(a.run/'synthetic_fields.npz');fig,axes=plt.subplots(2,6,figsize=(16,5))
    for i,word in enumerate(('north','south','east','west','inside','outside')):
        for row,key in enumerate(('analytic_','compiler_')):
            ax=axes[row,i];ax.imshow(probabilities(synthetic[key+word]).reshape(28,28),origin='lower',extent=[-100,100,-100,100],cmap='magma')
            cs=synthetic['polygon'];ax.plot(*np.vstack((cs,cs[0])).T,c='cyan');ax.set_title(word+(' / explicit reference' if row==0 else ' / current rule baseline'),fontsize=8)
    fig.suptitle('Synthetic rectangle: north := +Y, east := +X by construction; not real-map georeferencing')
    fig.tight_layout();fig.savefig(o/'synthetic_control.png',dpi=140);plt.close(fig)
    logits=np.load(a.run/'all_logits.npz');selected=[]
    for pair in PAIRS:
        pool=[r for r in rows if r['pair']==pair and r['single_ID'] and r['relation_in_token_window'] and r['metrics']['Geometry']['valid_regions']]
        conditions=[lambda r: r['metrics']['Geometry']['joint_mass_correct'] and not r['metrics']['Relation']['joint_mass_correct'],
                    lambda r:r['metrics']['Relation']['joint_mass_correct'],
                    lambda r:not r['metrics']['Geometry']['joint_mass_correct'] and not r['metrics']['Relation']['joint_mass_correct']]
        used=set()
        for condition in conditions:
            candidate=next((r for r in pool if r['index'] not in used and condition(r)),None)
            if candidate:selected.append(candidate);used.add(candidate['index'])
    case_data=[]
    for row in selected:
        task=tasks[row['index']];pair=row['pair'];wa,wb=PAIRS[pair];word_list=['north','south','east','west','inside','outside']
        g=logits[f'{row["index"]:04d}_geometry'];rel=logits[f'{row["index"]:04d}_relation'];fields={'Geometry':g,'Relation':rel,'Effective':g if pair!='IO' else rel}
        bounds=task['world_bounds'];extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
        fig,axes=plt.subplots(3,3,figsize=(12,10))
        for ridx,(branch,values) in enumerate(fields.items()):
            pa,pb=[probabilities(values[word_list.index(w)]) for w in (wa,wb)]
            for col,value in enumerate((pa,pb,pb-pa)):
                ax=axes[ridx,col]
                if col<2:ax.imshow(value.reshape(28,28),origin='lower',extent=extent,cmap='magma',vmin=0,vmax=max(pa.max(),pb.max()))
                else:
                    limit=max(np.abs(value).max(),1e-10);ax.imshow(value.reshape(28,28),origin='lower',extent=extent,cmap='coolwarm',vmin=-limit,vmax=limit)
                for cs in task['anchors'][0]['contours']:ax.plot(*np.array(cs).T,c='cyan',lw=.8)
                ax.set_title(branch+' / '+([wa,wb,wb+' minus '+wa][col]))
                ax.set_xlabel('World X');ax.set_ylabel('World Y')
            m=row['metrics'][branch]
            for col,key in enumerate(('centroid_a','centroid_b')):axes[ridx,col].scatter(*m[key],c='lime',marker='x')
        fig.suptitle(f"TRAIN {row['index']}: same map/Polygon/category, {pair}\n{row['texts'][wa]}\n{row['texts'][wb]}",fontsize=8)
        fig.tight_layout();filename=f'case_{pair}_{row["index"]:04d}.png';fig.savefig(o/filename,dpi=120);plt.close(fig)
        case_data.append(dict(index=row['index'],pair=pair,image=filename,texts=row['texts'],metrics=row['metrics']))
    atomic_json(case_data,o/'cases.json')
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for i,pair in enumerate(PAIRS):
        q=summary['primary'][pair];axes[i].bar(['Geometry','Relation','Effective'],[q[m]['joint_mass_rate']*100 for m in ('Geometry','Relation','Effective')]);axes[i].set_ylim(0,100);axes[i].set_title(f'{pair}: joint region-mass response, n={q["queries"]}')
    fig.tight_layout();fig.savefig(o/'summary.png',dpi=140);plt.close(fig)
    lines=['# 几何验证与关系词反事实：FP32冻结实验','',
        '结论：显式几何计算在人工矩形场景中通过；但现有模型对关系词的分布响应不稳定，新增Relation在NS条件下明显抵消Geometry响应。不能据当前结果认定模型已真正理解几何关系。当前仅诊断，没有训练新网络、修改权重或运行导航。','',
        '## 1. 坐标和几何先验先验收','',
        f'- 检查{coordinate["TRAIN_maps"]}张TRAIN地图：当前栅格列+X、行−Y；世界网格列+X、行+Y。4种yaw的世界/图像往返误差最大{coordinate["max_world_pixel_roundtrip_error_m"]:.6f}m，未发现数值翻轴或投影错误。',
        '- 所有实际TIF无CRS；官方栅格化脚本没有提供本地World到地理坐标的旋转/原点对应。不能把图像上方、工程+Y或本地文档的north-up措辞当独立地理证明。地理方向正确率记为未知；下文N/S、E/W指标仅按现有工程轴约定衡量，不依据GT翻轴。',
        '- 一手来源：[官方栅格化代码](https://github.com/water-cookie/citynav/blob/main/rasterize.py)、[官方位姿/视域代码](https://github.com/water-cookie/citynav/blob/main/gsamllavanav/space.py)。逐地图仿射/CRS及本地源SHA见coordinate_audit.json。',
        '- 人工场景：中心矩形、784个均匀候选。显式区域参考分布的三对词均通过方向/质量检查；Polygon内点、边界点和外点距离实测0、0、10、10、14.142m，符合解析真值。该参考分布仅作诊断正控制，没有接入模型或训练。',
        '- 现有规则编译器：N/S/E/W可生成对应关系；inside/outside没有显式算子，两者规则图完全相同。这是算子缺失，不是Polygon距离算错。见synthetic_control.png。','',
        '## 2. TRAIN反事实协议','',
        f'- 使用既有573条公开输入查询，生成{manifest["paired_conditions"]}个配对条件，即NS、EW、IO每组573对、共3438个变体。不是1719个独立物理目标。',
        '- 固定完整Polygon、地图范围、目标类别/属性及其他文本；只替换已通过作用域检查且不落在地标名称里的关系片段。连续BERT Token、结构Predicate、Geometry节点/规则均重新计算；每批重置单Episode缓存。',
        '- 神经网络前向全程FP32、TF32关闭、无RGB、无训练；离线几何/指标统计用NumPy/GEOS高精度，送入模型的浮点Tensor与权重均FP32。不读取真实目标GT或私有标签文件，不访问Val/Test。原权重SHA实验前后相同。反事实条件没有新目标GT，不计算R20/SR。',
        '- 原生N/S和inside语句另列native_only；由于新Head训练中east/west/outside计数为0，来自near等原文的对应改动属于受控语言压力测试，不冒充自然E/W训练样本。其他属性/空间约束保留，不能保证每个反事实在真实场景存在可行目标。',
        '- 主表限单公开ID、关系词完整进入96-token窗口、两侧关系区域在28×28网格均有支持；多Polygon、空区域、被截断者另计。完整573查询及未知原因仍保留在pairs.json，未按模型结果或GT选样本。',
        '- 方向区域采用完整Polygon外侧半平面；inside/outside采用Polygon覆盖/补集。inside零网格点时记未知。指标包括质心有符号位移、两侧区域概率质量、双条件质量响应、Top-1区域命中和TV，不只看Top-1移动。','',
        '## 3. 主结果（工程轴条件一致性）','',
        '|词对|有效查询/Anchor组|分支|质心方向一致率|平均有符号位移m|双向区域质量响应率|A区域质量变化|B区域质量变化|平均TV|',
        '|---|---:|---|---:|---:|---:|---:|---:|---:|']
    for pair in PAIRS:
        q=summary['primary'][pair]
        for branch in ('Geometry','Relation','Effective'):
            m=q[branch];rate='不适用' if m['centroid_direction_rate'] is None else f'{m["centroid_direction_rate"]*100:.2f}%'
            shift='不适用' if m['mean_signed_shift_m'] is None else f'{m["mean_signed_shift_m"]:.3f}'
            lines.append(f'|{pair}|{q["queries"]}/{q["anchor_groups"]}|{branch}|{rate}|{shift}|{m["joint_mass_rate"]*100:.2f}%|{m["mean_mass_change_a"]:+.5f}|{m["mean_mass_change_b"]:+.5f}|{m["mean_TV"]:.5f}|')
    lines += ['', '正位移表示north比south更靠工程+Y，east比west更靠+X；负位移相反。IO没有全局统一方向轴，不能强行用质心朝向定义正确，故以区域概率响应评价。正确响应指两条指令都增加各自对应区域概率；不意味着目标实例正确或质量超过50%。面积分母、分组bootstrap 95%区间及按fit/dev分层均在summary.json。',
        'Geometry同时含既有显式方向规则，不能把其方向响应全部解释为语言网络学会了几何；本轮未进行只改文本/只改规则的不一致输入消融。','',
        '### 最高概率位置是否也落在对应几何区域','',
        '|词对|Geometry双条件Top-1命中|Relation双条件Top-1命中|Effective双条件Top-1命中|','|---|---:|---:|---:|']
    for pair in PAIRS:
        q=summary['primary'][pair];lines.append(f'|{pair}|'+ '|'.join(f'{q[b]["both_top1_regions_rate"]*100:.2f}%' for b in ('Geometry','Relation','Effective'))+'|')
    lines += ['', '这表明“质心有正确小幅位移”仍不代表最高概率位置进入了关系目标区域。这里使用完整Polygon外侧/内外区域的几何定义，不是真实目标实例GT。','',
        '### 质控与原生词对','',
        '|词对|原生条件有效查询|多公开ID总数|两侧区域缺支持总数|Token截断|显式规则缺失|规则完全相同总数|','|---|---:|---:|---:|---:|---:|---:|']
    for pair in PAIRS:
        q=summary['quality'][pair];lines.append(f'|{pair}|{summary["native_only"][pair]["queries"]}|{q["multiple_public_IDs"]}|{q["no_region_grid_support"]}|{q["language_truncated"]}|{q["missing_explicit_compiler_rule"]}|{q["compiler_identical_rules"]}|')
    lines += ['', '各排除原因可能重叠；原生EW为0，因此不宣称测得自然EW指令准确率。CI按地图+公开Anchor ID聚类重采样，六个变体不当作六个独立样本；无需从有限TRAIN干预宣称未见泛化或统计显著提升。',
        'NS/EW另有8条未生成显式规则，属于名称/语法到规则的编译缺口，不能全部归为神经Head未学会。IO全部573条没有对应规则算子。原始文本Token与结构Predicate仍同步改变。','',
        '### 原生词对条件（非全部由near等改写）','',
        '|词对|查询|分支|质心方向一致率|双向区域质量响应率|','|---|---:|---|---:|---:|']
    for pair in PAIRS:
        q=summary['native_only'][pair]
        if not q['queries']:continue
        for branch in ('Geometry','Relation','Effective'):
            m=q[branch];rate='不适用' if m['centroid_direction_rate'] is None else f'{m["centroid_direction_rate"]*100:.2f}%'
            lines.append(f'|{pair}|{q["queries"]}|{branch}|{rate}|{m["joint_mass_rate"]*100:.2f}%|')
    lines += ['',
        '## 4. 三类问题区分','',
        '1. **坐标问题**：数值世界/栅格/Footprint转换通过；地理北/东的独立对应尚缺元数据。不能把这项未知直接归因于模型，也不根据GT或指标翻转轴。',
        '2. **模型/关系编译问题**：Geometry存在方向响应但不稳定；新增Relation在NS上大多使响应反向。IO规则算子缺失，outside没有本轮新Head监督，概率响应接近不稳定的词敏感性，不是可靠几何控制。原始文本和Predicate确实同步改动，并非沿用语言缓存。',
        '3. **运行时回退**：PR16对N/S/E/W撤销新增Relation输出，最终有效Belief等于修改文本后的Geometry，所以Effective会变化，不能误称整个系统完全忽略方向；它也不是Relation学会后的成果。IO无此方向回退，Effective与Relation相同。','',
        '## 5. 第三步建议：先约束学习，不增加复杂度','',
        '- 方法A：复用本轮已验证的Polygon显式几何区域作为可解释先验，支持多个候选距离/多峰；接入真实地理方向前先解决坐标参照，不把矩形中的north定义自动推广到城市地图。',
        '- 方法B：先在合成TRAIN几何样本用成对关系一致性损失，要求正确区域质量高于错误区域；两种相反指令都约束。只用语言/公开几何可构造的关系标签，不给反事实伪造真实实例坐标。',
        '- 同时检查全分布、区域面积基线和可行区域空集，避免outside大区域自然占优或边界裁剪被误当作学习成功。之后才在可靠真实TRAIN分组验证，保持独立开发组；本轮未执行该训练、未修改普通推理。',
        f'- GPU推理与数据构建实测{manifest["seconds"]:.2f}s（不含模型加载/绘图），进程峰值{manifest["peak_GPU_GiB"]:.3f}GiB；Checkpoint `{manifest["checkpoint_sha256"]}`。全部原始logits保存在服务器 `{a.run}/all_logits.npz`，Git保留逐对JSON、图册、配置和代表热图。']
    (o/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    page='''<!doctype html><meta charset="utf-8"><title>Relation counterfactuals</title><style>body{font:16px sans-serif;max-width:1400px;margin:auto}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>几何验证→关系反事实（FP32，不训练）</h1><p><a href="REPORT_zh.md">中文报告</a> · <a href="summary.json">完整统计/聚类区间</a> · <a href="pairs.json">全部逐对文本和指标</a></p><p>真实地图地理北/东未独立标定；图中只有公开Polygon与概率质心，没有GT目标。所有图片为模型真实输出。</p><img src="synthetic_control.png"><img src="summary.png">'''
    for case in case_data:page+=f'<h2>{case["pair"]} / TRAIN {case["index"]}</h2><pre>{html.escape(json.dumps(case["texts"],ensure_ascii=False,indent=2))}</pre><a href="{case["image"]}"><img loading="lazy" src="{case["image"]}"></a>'
    (o/'index.html').write_text(page)
    print(json.dumps(summary['primary'],indent=2),flush=True)


if __name__=='__main__':main()
