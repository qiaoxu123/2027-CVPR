"""Final TRAIN-Dev minimal-input report, paired predictions and actual heatmaps."""
import argparse,json,gzip,html,shutil,collections
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.geometry import grid_centers
from scripts.train_landmark_geometry_words import LandmarkWordsDataset,encode_model_inputs,collate
from graphnav.landmark_geometry_only_model import LandmarkGeometryWordsOnly

LABELS={'real':'Name + geometry words','name_only':'Name only','scrambled_words':'Scrambled geometry words'}

def main():
    p=argparse.ArgumentParser()
    for k in ('run','train-jsonl','public-objects','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);report=json.load(open(a.run/'report.json'));history=json.load(open(a.run/'history.json'));pred=torch.load(a.run/'dev_predictions.pt',map_location='cpu',weights_only=True);state=torch.load(a.run/'last.pt',map_location='cpu',weights_only=True)
    ds=LandmarkWordsDataset(a.train_jsonl,a.public_objects);records=[json.loads(s) for s in a.train_jsonl.read_text().splitlines()];ids=pred['ids'];assert len(ids)==7619 and len(set(ids))==7619;assert all(ds.records[i]['map_name'].startswith('birmingham_') for i in ids)
    grid=grid_centers(28).numpy();gt=np.stack([ds.records[i]['target_xy'] for i in ids]);scale=np.stack([ds.records[i]['map_sizes'] for i in ids]);metrics={};detail={};base=None
    for arm,z in pred['logits'].items():
        assert z.shape==(7619,784) and torch.isfinite(z).all();probs=z.softmax(-1).numpy();top=z.argmax(-1).numpy();ne=np.linalg.norm((grid[top]-gt)*scale,axis=-1);good=ne<=20
        if arm=='name_only':base=good
        detail[arm]=dict(ne=ne,good=good,top=top,probs=probs)
    base=detail['name_only']['good'];rows=[]
    for j,idx in enumerate(ids):
        row=dict(index=idx,map_name=ds.records[idx]['map_name'],status=ds.records[idx]['status'],methods={})
        for arm,d in detail.items():
            mass=float(d['probs'][j,np.linalg.norm((grid-gt[j])*scale[j],axis=-1)<=20].sum());row['methods'][arm]=dict(NE_m=float(d['ne'][j]),R20=bool(d['good'][j]),GT20_mass=mass,top1_cell=int(d['top'][j]))
        rows.append(row)
    for arm,d in detail.items():
        metrics[arm]=dict(N=7619,R20=float(d['good'].mean()),NE_m=float(d['ne'].mean()),corrections=int((~base&d['good']).sum()),damage=int((base&~d['good']).sum()),net=int(d['good'].sum()-base.sum()),GT20_mass=float(np.mean([r['methods'][arm]['GT20_mass'] for r in rows])),changed_goals=int((d['top']!=detail['name_only']['top']).sum()))
    # Bootstrap public physical reference groups (not repeated instructions as IID).
    keys=[(ds.records[i]['map_name'],tuple(sorted({x['public_id'] for x in ds.records[i]['anchors']}))) for i in ids];bykey=collections.defaultdict(list)
    for j,k in enumerate(keys):bykey[k].append(j)
    groups=list(bykey);rng=np.random.default_rng(42);CI={}
    for arm in ('real','scrambled_words'):
        differences=[]
        for _ in range(2000):
            draw=rng.integers(len(groups),size=len(groups));ix=np.concatenate([bykey[groups[k]] for k in draw]);differences.append(float((detail[arm]['good'][ix].astype(float)-base[ix]).mean()))
        CI[arm]=np.quantile(differences,[.025,.975]).tolist()
    differences=[]
    for _ in range(2000):
        draw=rng.integers(len(groups),size=len(groups));ix=np.concatenate([bykey[groups[k]] for k in draw]);differences.append(float((detail['real']['good'][ix].astype(float)-detail['scrambled_words']['good'][ix]).mean()))
    CI['real_vs_scrambled']=np.quantile(differences,[.025,.975]).tolist()
    atomic_json(dict(methods=metrics,reference_group_bootstrap_delta_R20_95CI=CI,groups=len(groups)),a.output/'metrics.json')
    with gzip.open(a.output/'paired.json.gz','wt') as f:json.dump(rows,f)
    for name in ('manifest.json','history.json','report.json','text_input_audit.json'):shutil.copyfile(a.run/name,a.output/name)
    coverage=collections.defaultdict(collections.Counter);examples=[];status_examples=collections.Counter()
    for i,r in enumerate(ds.records):
        split='fit' if r['map_name'].startswith('cambridge_') else 'dev';coverage[split][r['status']]+=1
        if status_examples[r['status']]<5:
            examples.append(dict(index=i,map_name=r['map_name'],status=r['status'],original=records[i]['instruction'],input_pairs=[dict(name=x['name'],geometry_words=x['geometry_words']) for x in r['anchors']]))
            status_examples[r['status']]+=1
    atomic_json(dict(all=len(ds),distribution=dict(ds.counter),split={k:dict(v) for k,v in coverage.items()},semantic_directness_verified=False),a.output/'coverage.json');atomic_json(examples,a.output/'extraction_examples.json')
    fig,ax=plt.subplots(figsize=(6,3));ax.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],'o-');ax.set_xlabel('Epoch');ax.set_ylabel('All-fit target CE');fig.tight_layout();fig.savefig(a.output/'training_curve.png',dpi=150);plt.close(fig)
    # Include improvements, regressions, unchanged paired, and missing-input fallbacks.
    fixes=[j for j in range(len(ids)) if not base[j] and detail['real']['good'][j]];damage=[j for j in range(len(ids)) if base[j] and not detail['real']['good'][j]];chosen=fixes[:4]+damage[:4]
    for status in ('paired','name_only','no_public_name'):
        candidates=[j for j,i in enumerate(ids) if ds.records[i]['status']==status and j not in chosen];chosen.extend(candidates[:2])
    cases=[]
    for j in chosen:
        idx=ids[j];record=ds.records[idx];bounds=np.array(records[idx]['world_bounds']);truth=bounds[:2]+gt[j]*(bounds[2:]-bounds[:2]);fig,axes=plt.subplots(1,3,figsize=(12,4));vmax=max(detail[arm]['probs'][j].max() for arm in LABELS)
        for ax,(arm,label) in zip(axes,LABELS.items()):
            ax.imshow(detail[arm]['probs'][j].reshape(28,28),origin='lower',extent=[bounds[0],bounds[2],bounds[1],bounds[3]],vmin=0,vmax=vmax)
            for anchor in record['anchors']:
                c=np.asarray(anchor['contour'])*(bounds[2:]-bounds[:2])+bounds[:2];ax.plot(*np.vstack([c,c[0]]).T,'c-',lw=.6)
            loc=grid[detail[arm]['top'][j]]*(bounds[2:]-bounds[:2])+bounds[:2];ax.plot(*loc,'wx');ax.plot(*truth,'r*');ax.set_aspect('equal');ax.set_title(f"{label}\nNE={detail[arm]['ne'][j]:.1f}m")
        fig.suptitle(f'TRAIN {idx}: '+records[idx]['instruction'][:140],fontsize=8);fig.tight_layout();fname=f'belief_{idx:05d}.png';fig.savefig(a.output/fname,dpi=140);plt.close(fig)
        cases.append(dict(index=idx,original=records[idx]['instruction'],pairs=[dict(name=x['name'],geometry_words=x['geometry_words']) for x in record['anchors']],figure=fname))
    atomic_json(cases,a.output/'cases.json')
    body='<meta charset="utf-8"><style>body{max-width:1300px;margin:auto;font:16px sans-serif}img{max-width:100%}pre{white-space:pre-wrap}</style><h1>PR22：仅地标名称＋几何词组</h1><p>全部Birmingham TRAIN-Dev，非Val/Test，不是导航SR。红星GT、白叉预测、青线公开Polygon。</p><pre>'+html.escape(json.dumps(metrics,indent=2))+'</pre><img src="training_curve.png">'
    for c in cases:body+=f'<h2>TRAIN {c["index"]}</h2><p>{html.escape(c["original"])}</p><pre>{html.escape(json.dumps(c["pairs"],ensure_ascii=False,indent=2))}</pre><a href="{c["figure"]}"><img src="{c["figure"]}"></a>'
    (a.output/'index.html').write_text(body)
    counts=ds.counter;n=len(ds);paired=counts['paired'];names=counts['paired']+counts['name_only'];m=metrics
    table='\n'.join(f"|{LABELS[k]}|{v['N']}|{100*v['R20']:.3f}%|{v['NE_m']:.4f}|{v['corrections']}|{v['damage']}|{v['net']:+d}|" for k,v in m.items())
    subt='\n'.join(f"|{cat}|{v['real']['N']}|{v['name_only']['R20']*100:.3f}%|{v['real']['R20']*100:.3f}%|{v['scrambled_words']['R20']*100:.3f}%|{v['real']['NE_m']:.3f}|" for cat,v in report['dev'].items() if cat!='shuffle_audit')
    texts=json.load(open(a.run/'text_input_audit.json'));lengths=list(texts['token_lengths'].values());good_delta=m['real']['R20']-m['name_only']['R20'];shuffle_delta=m['real']['R20']-m['scrambled_words']['R20'];decision='观察到正确几何词相对两种对照的点估计增益；仍需区分关系作用域和参照系，不能宣称完整语义理解或导航提升。' if good_delta>0 and shuffle_delta>0 else '正确几何词未稳定优于两种对照；当前最小网络未证实可靠的几何词定位增益，不能据此扩大闭环。'
    md=f'''# PR22 地标名称＋相邻几何词组：正式TRAIN实验

基于PR22 `73ff64ec5383c33446382c5613e168b6fe3ce5ad`；工作分支`experiment/landmark-geometry-words-only-20261105`。原模型与main未改动，没有Val/Test或闭环导航。

## 输入与解析核验

模型从零初始化。文本只来自公开CityRefer名称和紧邻的白名单几何短语，分别经过冻结BERT得到768维CLS。无完整原始指令编码、目标颜色/类别/外观、原Geometry隐层、RGB、轨迹、目标ID或目标坐标输入。名称中自带的Church/Road等是专名的一部分，不是额外目标类别。`across the road from`按文档视为完整几何构式，不单独编码中间对象。

使用FULL cityrefer_original公共目录，只按原文名称匹配，同名全部Polygon保留；不使用旧目标专属processed引用表。目标GT只在forward后计算损失和评分。解析只是词法邻接，不证明Target语义上直接连接；例如“停车场behind地标里的车”仅提供behind构式，不能据此声称车本身具有该直接关系。

全Revised TRAIN{n:,}条纳入：Cambridge14,259拟合、Birmingham7,619最终确认。名称＋短语{paired:,}条（{paired/n*100:.2f}%）；有可用名称{names:,}条（{names/n*100:.2f}%）。仅名称{counts['name_only']:,}、无公开名{counts['no_public_name']:,}、超16 Polygon容量{sum(v for k,v in counts.items() if k.startswith('rejected')):,}；后两类均匀回退，保留分母，不用GT选片段。

检查并修复自交/零面积Polygon接受问题；复合方向`north-east/north east`不再被截成`east`，直接否定`not north of`保留否定。无相对方向到世界南北的硬编码。复杂关系作用域、并列继承及参考系仍不保证准确，覆盖率不能叫解析准确率。

BERT只缓存{len(lengths)}种允许字符串，每段含CLS/SEP的长度范围{min(lengths)}–{max(lengths)}，中位{np.median(lengths):.1f}；96是上限，不是每条固定Token数。每个Anchor有名称和词组各一枚CLS摘要，最多16个Polygon槽；网络不是原完整句Token模型。

## 训练及对照

seed42、3轮、batch16；每轮完整14,259条fit，固定最后Checkpoint，无Dev选参。AdamW2e-4、裁剪1、FP32；训练参数{sum(v.numel() for v in state['model'].values()):,}。冻结BERT，原Geometry/Relation/PGVB/导航器均不加载。

训练{report['training_seconds']:.3f}s，含Dev评估总计{report['total_training_eval_seconds']:.3f}s；峰值{report['peak_GPU_GiB']:.3f}GiB。12步真实GPU Smoke非零词组梯度、有限Loss通过；全部CPU测试结果见CPU_tests.txt。恢复支持完成epoch的优化器/模型/RNG，不宣称步内恢复。

三组为同一最终模型的输入消融，**不是分别重训的三个最优模型**。原batch内roll会依赖batch大小，已改为全Dev固定置换，按查询/名称绑住同名多Polygon词组，保留原word_mask；{pred['shuffle']['units']}个短语单元中{pred['shuffle']['changed']}个实际改变，不用GT置换。未知任务不排除。

## 全部7,619条Birmingham TRAIN-Dev

|方法|N|R@20m|NE(m)|纠正vs名称|破坏vs名称|净收益|
|---|---:|---:|---:|---:|---:|---:|
{table}

|输入子组|N|名称R20|真实短语R20|打乱短语R20|真实短语NE(m)|
|---|---:|---:|---:|---:|---:|
{subt}

参考Polygon组合分组bootstrap（{len(groups)}组）真实短语对名称R20差值95%区间[{CI['real'][0]*100:.3f},{CI['real'][1]*100:.3f}]个百分点；真实对打乱为[{CI['real_vs_scrambled'][0]*100:.3f},{CI['real_vs_scrambled'][1]*100:.3f}]个百分点。按公共参考组合分组不等于严格物理目标独立，因此不作广泛显著性结论。单种子、仅TRAIN-Dev，不能推出导航SR或未知场景泛化。

{decision}

无公开名称时输出均匀分布，Top-1按固定argmax首格计分；不能把该组偶然命中称作有效定位。历史完整Geometry+Relation使用不同输入、权重及数据拟合范围，不能直接把其约70%与本实验作公平性能差值。

名称/关系相同的不同车辆等目标被刻意映射到同一模型输入，无法恢复被丢弃的实例身份；方向样本不均衡、front/behind参考系未知也会限制结果。是否响应关系词与是否定位正确是两个问题，不能仅凭热图移动判断成功。缺少人工关系真值，不能报告语言解析准确率。

## 交付与恢复

[交互图册](index.html)、[提取示例](extraction_examples.json)、[训练曲线](training_curve.png)、[逐任务指标](paired.json.gz)、[机器指标](metrics.json)。服务器`{a.run}`保存完整提取审计、BERT允许文本、Checkpoint、预测与日志。

最终Checkpoint：`{a.run/'last.pt'}`；SHA256 `{file_sha256(a.run/'last.pt')}`。未覆盖旧权重。adversarial-review调用因workspace routing discovery timed out失败，不列为独立审查通过。
'''
    (a.output/'REPORT_zh.md').write_text(md);print(json.dumps(metrics,indent=2))

if __name__=='__main__':main()
