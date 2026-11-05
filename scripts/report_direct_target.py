"""Chinese actual-results report, paired tables and real RGB/Belief/trajectory gallery."""
import argparse
from collections import Counter,defaultdict
import html
import json
from pathlib import Path
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import torch
from graphnav.checkpointing import file_sha256
from graphnav.landmark_projection import world_to_frame


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.root;o=a.output;o.mkdir(parents=True,exist_ok=True)
    tasks=json.loads((r/'val_seen_selection/tasks_public.json').read_text());labels=json.loads((r/'val_seen_selection/labels_private.json').read_text())
    rows=[json.loads(l) for l in (r/'closedloop/predictions.jsonl').read_text().splitlines()]
    full=json.loads((r/'closedloop/full_summary.json').read_text());pilot=json.loads((r/'closedloop/pilot_summary.json').read_text())
    dev=json.loads((r/'dev_evaluation_corrected/summary.json').read_text());pred=json.loads((r/'dev_evaluation_corrected/predictions.json').read_text())
    train=json.loads((r/'training_aligned/complete.json').read_text());history=json.loads((r/'training_aligned/history.json').read_text())
    audit=json.loads((r/'inventory_validated/audit.json').read_text());methods=['Geometry','Relation','RGB']
    by={(x['index'],x['method']):x for x in rows}
    paired=[]
    for i in range(len(tasks)):
        q=dict(index=i,episode=tasks[i]['episode'],map_name=tasks[i]['map_name'],instruction=tasks[i]['instruction'])
        q.update({m:by[i,m]['metrics'] for m in methods});paired.append(q)
    comparisons={}
    for left,right in [('Geometry','Relation'),('Relation','RGB'),('Geometry','RGB')]:
        correct=sum(bool(not q[left]['SR'] and q[right]['SR']) for q in paired)
        damage=sum(bool(q[left]['SR'] and not q[right]['SR']) for q in paired)
        comparisons[left+' -> '+right]=dict(corrected=correct,damaged=damage,net=correct-damage)
    rgb_steps=[d for row in rows if row['method']=='RGB' for d in row['diagnostics']]
    mechanism=dict(RGB_decisions=len(rgb_steps),visual_active=sum(bool(d.get('visual_active')) for d in rgb_steps),
        tracking_states=dict(Counter((d.get('tracking') or {}).get('state','none') for d in rgb_steps)),
        stable_events=dict(Counter((d.get('stable_goal') or {}).get('event','none') for d in rgb_steps)))
    (o/'navigation_mechanism.json').write_text(json.dumps(mechanism,indent=2))
    strata={}
    for field in ('category','relation'):
        strata[field]={}
        for value in sorted({p[field] for p in pred}):
            ps=[p for p in pred if p[field]==value]
            strata[field][value]=dict(states=len(ps),queries=len({p['index'] for p in ps}),
                **{m:dict(R20=float(np.mean([p[m]['R20'] for p in ps])),NE=float(np.mean([p[m]['NE'] for p in ps]))) for m in methods})
    for name,value in [('paired_results.json',paired),('paired_changes.json',comparisons),('dev_strata.json',strata),('full_summary.json',full),('pilot_summary.json',pilot),('dev_summary.json',dev)]:
        (o/name).write_text(json.dumps(value,ensure_ascii=False,indent=2))
    # Every actual trajectory retained, not only success selections.
    shutil.copy2(r/'closedloop/predictions.jsonl',o/'all_trajectories.jsonl')
    for folder in ('training_aligned','smoke'):
        dest=o/folder;dest.mkdir(exist_ok=True)
        for name in ('config.json','history.json','complete.json','gradient_audit.json','steps.jsonl','smoke.json','resume_proof.json'):
            path=r/folder/name
            if path.exists():shutil.copy2(path,dest/name)
    shutil.copy2(r/'closedloop/freeze_manifest.json',o/'freeze_manifest.json')
    for name in ('cache_validated.log','cache_aligned.log','training_aligned.log','pilot.log','full_closedloop.log'):
        if (r/name).exists():shutil.copy2(r/name,o/name)
    selection=json.loads((r/'val_seen_selection/selection_manifest.json').read_text())
    selected=sorted(range(len(tasks)),key=lambda i:(
        -(int(by[i,'Geometry']['metrics']['SR']!=by[i,'RGB']['metrics']['SR'])),i))[:8]
    cases=[]
    for i in selected:
        task=tasks[i];rr=by[i,'RGB'];cap=rr['captures'][-1] if rr['captures'] else None
        if cap is None:continue
        z=np.load(cap);rgb=cv2.cvtColor(cv2.imread(str(Path(cap).with_suffix('.png'))),cv2.COLOR_BGR2RGB)
        fig,axes=plt.subplots(1,5,figsize=(23,4));axes[0].imshow(rgb)
        for anchor in task['anchors']:
            for contour in anchor['contours']:
                px=world_to_frame(np.asarray(contour),z['footprint'],rgb.shape);axes[0].plot(*np.vstack([px,px[0]]).T,c='yellow',lw=.8)
        axes[0].set_title('Actual RGB + public anchor (not pixel GT)');axes[0].set_xlim(0,383);axes[0].set_ylim(383,0)
        bounds=task['world_bounds'];extent=[bounds[0],bounds[2],bounds[1],bounds[3]];target=labels[i]['target']
        for ax,key,title in zip(axes[1:4],('geometry','relation','full'),methods):
            log=z[key];prob=np.exp(log-log.max());prob/=prob.sum()
            ax.imshow(prob.reshape(28,28),origin='lower',extent=extent,cmap='magma')
            for anchor in task['anchors']:
                for cs in anchor['contours']:ax.plot(*np.asarray(cs).T,c='cyan',lw=.7)
            ax.scatter(*target,c='lime',marker='x',s=35);ax.set_title(title+' target Belief')
        for m in methods:
            path=np.array(by[i,m]['poses']);axes[4].plot(path[:,0],path[:,1],label=m)
        axes[4].scatter(*target,c='red',marker='x');axes[4].legend(fontsize=7);axes[4].set_title('Actual paired rollouts')
        fig.suptitle(f"{i}: {task['instruction']}",fontsize=8,wrap=True);fig.tight_layout()
        fname=f'case_{i:02d}.png';fig.savefig(o/fname,dpi=130);plt.close(fig)
        cases.append(dict(index=i,image=fname,instruction=task['instruction'],metrics={m:by[i,m]['metrics'] for m in methods}))
    fig,axes=plt.subplots(1,3,figsize=(13,3.6))
    axes[0].plot([h['epoch'] for h in history],[h['fit_loss'] for h in history],marker='o');axes[0].set_title('Actual fit loss')
    for m in methods:axes[1].plot([h['epoch'] for h in history],[h['dev'][m]['R20'] for h in history],marker='o',label=m)
    axes[1].legend(fontsize=7);axes[1].set_title('TRAIN Dev Goal R20 (states)')
    axes[2].bar(methods,[full['all'][m]['SR']*100 for m in methods]);axes[2].set_title('All eligible Val-Seen SR (%)');fig.tight_layout();fig.savefig(o/'summary.png',dpi=150);plt.close(fig)
    lines=['# UnifiedPGVB直接关系目标Belief：真实训练与闭环结果','',
      '## 范围与结论','',
      '本轮完成端到端核心网络训练与真实闭环，不修改Stable Goal、Trajectory/Stop、Horizon或成功阈值。BERT/SigLIP2为冻结缓存编码器，不能称为所有视觉骨干参数的全参数训练。所有数字为实际测量；本轮不访问Test。','',
      f'原始22,002条TRAIN文本与Revised 21,878条任务连接，严格筛选{audit["qualified_queries"]}条，拟合493、开发80，地图和公开Anchor ID隔离。有效Polygon检查排除5条。近邻关系占多数，north/south较少，east/west无保留样本，相对方向未知时拒绝。',
      '筛选不使用目标GT；保留原始完整BERT语言、目标外观与序数约束，关系规则不是全部语言模型。32条抽读仅验证粗关系作用域合理，不是独立人工关系准确率或真实像素标注。',
      'Val-Seen全2470条经冻结规则得到56条合格任务；先固定32条Pilot，随后补齐全部56条。其余任务为当前解析能力范围外，本报告不能称为完整CityNav SR。所有56条和全部三组均计入分母。',
      '三组使用同一个联合训练后的Checkpoint；Geometry是关闭新增Head的内部消融，不是历史未训练Checkpoint的闭环成绩。原Checkpoint仅另作TRAIN Dev离线定位对照。','',
      '## 模型、训练和纠错记录','',
      'Geometry、PGVB和新增关系/目标Head联合训练。Polygon相对几何与方向嵌入和完整语言语义共同产生目标初始分布；PGVB共享视觉投影/Anchor注意力辅助目标残差，仅修改当前实际支持的世界网格。没有RGB时精确回退Relation。候选ID、坐标、Context同步按新Belief重建。',
      f'GPU为RTX PRO 6000 Blackwell 96GB；seed=42，BF16。正式5轮{train["updates"]}次更新，覆盖{train["fit_queries_seen"]}/493条拟合查询；训练循环{train["seconds"]:.2f}s、峰值{train["peak_gpu_GiB"]:.3f}GiB（不含冻结编码器）。首次真实视觉缓存115.52s；Geometry类型修复后精确复用视觉张量、重建静态缓存16.73s。',
      '8步Smoke有非零核心梯度，恢复优化器后继续到12步。原PGVB/GraphNav历史Checkpoint均保留。第一轮道路类型泛化为Road造成训练/推理不一致，已弃用其结果并重训；无效轮次原日志保存在服务器training目录。正式采用training_aligned。',
      '离线评估曾跨批复用单Episode Geometry缓存，已显式reset并重算，旧结果不用于结论。训练时Geometry可训练不会缓存；闭环每Episode重置，不受该评估器问题影响。',
      '开发集只对新增拟合地图隔离；历史Geometry本身已在TRAIN预训练，不能宣称这些场景从未被任何基础Checkpoint见过。','',
      '## TRAIN Dev位置预测（80查询/240状态）','',
      '|方法|Goal R20|NE(m)|20m概率质量|','|---|---:|---:|---:|']
    for m in methods:
        q=dev['state_metrics'][m];lines.append(f'|{m}|{q["R20"]*100:.2f}%|{q["NE"]:.3f}|{q["mass20"]:.4f}|')
    lines += [f'|原Checkpoint Geometry|{dev["original_geometry"]["R20"]*100:.2f}%|{dev["original_geometry"]["NE"]:.3f}|未统计|','',
      '以上是状态指标，三个时刻不是三个独立目标。目标真实可见性未经独立人工确认，RGB支持不是可见性金标准。分类型/关系见dev_strata.json。',
      f'真实RGB R20={dev["state_metrics"]["RGB"]["R20"]*100:.2f}%，置零={dev["RGB_controls"]["zero"]["R20"]*100:.2f}%，打乱={dev["RGB_controls"]["shuffled"]["R20"]*100:.2f}%。打乱没有全面退化，不能据此宣称模型已可靠利用目标视觉身份。','',
      '## 真实闭环（原始CityNav目标，非地标代理）','',
      '|范围|方法|完成/应测|SR|OSR|SPL|NE(m)|nDTW|平均步数|推理ms|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for group in ('pilot','all'):
        for m in methods:
            q=full[group][m];lines.append(f'|{group}|{m}|{q["episodes"]}/{q["expected"]}|{q["SR"]*100:.2f}%|{q["OSR"]*100:.2f}%|{q["SPL"]:.3f}|{q["NE"]:.3f}|{q["nDTW"]:.4f}|{q["steps"]:.2f}|{q["inference_ms"]:.2f}|')
    lines += ['', '|配对变化|纠正|破坏|净收益|','|---|---:|---:|---:|']
    for name,c in comparisons.items():lines.append(f'|{name}|{c["corrected"]}|{c["damaged"]}|{c["net"]:+d}|')
    lines += ['', '所有逐任务原文、轨迹、终止原因和逐步Goal均保存于all_trajectories.jsonl；配对表见paired_results.json。失败未删除。定位R20与SR不同，不能用最后Goal正确代替实际到达。','',
      '## 当前判断与下一步','',
      '1. 语言解析足以开展这个保守子集的训练，远不足以覆盖全部CityNav；地图绑定/方向作用域仍需扩展独立审计。尤其east/west、front/behind/left/right尚未得到本轮训练验证。',
      '2. 关系Head已实际获得梯度并改变目标Belief；但开发集Relation的NE/R20未优于联合训练Geometry，说明“分布有变化”不等于正确语义偏移。不得把地理北解释未验证的问题隐藏为训练成功。',
      '3. RGB相对Relation的状态净纠正很小，打乱RGB没有稳定退化；结合完整配对闭环判断增量，不宣称真实实例识别或显著收益。当前没有支持扩大模型复杂度的证据。',
      '原A3 Geometry已接收完整BERT和关系字段，新增Relation不是“第一次加入关系信息”；其退化可能涉及重复计入已有关系、近邻不足以指定实例及监督/坐标语义不确定，当前实验无法单独分解这些因素。',
      '优先改进一项：针对Anchor—目标关系作用域及目标实例归属，建立更可靠的可观测目标监督和独立反事实；先验证结构关系/真实RGB分别带来的贡献，再扩大训练。保留当前负结果，不依据Val重新选参数或重训。','',
      f'Checkpoint：`{r}/training_aligned/best.pt`；SHA256：`{train["best_checkpoint_sha256"]}`。全部权重、缓存及逐步PNG/NPZ留服务器。Git仅存代码、日志、配对轨迹及代表图。',
      '补充：GPU资源工具误读PATH中的nvidia-smi包装器为无GPU；实际以/usr/bin/nvidia-smi及真实CUDA运行核验。报告中的显存为本进程分配峰值，不是服务器全部进程用量。']
    integrity=o/'integrity_audit.json'
    if integrity.exists():
        verified=json.loads(integrity.read_text());probe=verified['direction_probe']
        lines += ['', '## 额外输入一致性和方向控制核验','',
            f'573条公共Geometry记录与运行时打包规则逐项匹配；{verified["reused_visual_states"]}份真实视觉张量在类型修复前后逐元素相同；{verified["missing_visual_states"]}个观测无可用RGB证据，仍保留任务。训练/开发公开ID交集为0。原Stable Goal、Trajectory/Stop、A3控制器和闭环协议文件与499694b逐字节相同。',
            f'80查询结构ID探针：保持原文和Geometry不变，仅将关系ID换为north/south时，north期望Y大于south的比例{probe["north_higher_than_south_fraction"]*100:.2f}%，平均差{probe["mean_north_minus_south_Y_m"]:.3f}m。说明Head并未稳定产生原工程预期的+Y方向偏移。该探针文字未同步改变，不能当作完整语言方向准确率，也不能据GT或该结果翻转坐标。east/west无训练数据，其结构探针没有能力验证意义。',
            f'RGB共有{mechanism["RGB_decisions"]}次真实决策，{mechanism["visual_active"]}次视觉证据启用；跟踪状态为{mechanism["tracking_states"]}。Stable Goal事件为{mechanism["stable_events"]}，没有完成重规划事件。当前收益不能解释为接近目标后持续纠正导航Goal；保持规则未修改。',
            'PGVB随目标CE联合更新，但本版没有额外保留地标定位蒸馏Loss；目标残差和地标跟踪评分共享参数，存在相互干扰风险。当前实验不能分离该因素，不能据此修改已验证的Goal/Stop。']
    (o/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    content='''<!doctype html><meta charset="utf-8"><title>UnifiedPGVB Direct Target</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}img{width:100%}pre{white-space:pre-wrap}table{border-collapse:collapse}td,th{border:1px solid #ccc;padding:6px}</style><h1>真实直接关系训练及闭环诊断</h1><p>56条冻结合格任务，非全部CityNav；GT仅在图中事后评分。黄色为公开Anchor投影，不是人工像素GT。绿色叉为事后目标坐标。</p><p><a href="REPORT_zh.md">中文报告</a> · <a href="paired_results.json">全部逐任务配对</a> · <a href="all_trajectories.jsonl">全部真实轨迹</a></p><img src="summary.png">'''
    for c in cases:
        content+=f'<h2>Task {c["index"]}</h2><p>{html.escape(c["instruction"])}</p><a href="{c["image"]}"><img loading="lazy" src="{c["image"]}"></a><pre>{html.escape(json.dumps(c["metrics"],indent=2))}</pre>'
    (o/'index.html').write_text(content)
    files=[p for p in o.rglob('*') if p.is_file() and p.name!='artifacts_sha256.json']
    (o/'artifacts_sha256.json').write_text(json.dumps({str(p.relative_to(o)):file_sha256(p) for p in files},indent=2))
    print(json.dumps(dict(tasks=len(tasks),rows=len(rows),comparisons=comparisons,report=str(o/'REPORT_zh.md')),indent=2))


if __name__=='__main__':main()
