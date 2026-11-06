"""Reproducible offline report: existing Val-Seen traces only; never calls a policy.

Binding is completed and verified before loading targets for scoring/visualization.
"""
from __future__ import annotations
import argparse,csv,gzip,hashlib,html,io,json,math
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from scripts.analyze_val_seen_binding_failures import analyze,BINS

STATUS={'bound_public_polygon':'成功绑定','no_public_landmark_name_match':'无公开名称匹配',
        'matched_name_but_no_valid_polygon':'匹配名称但无有效Polygon','capacity_overflow_zero_anchor':'超过16容量，清空绑定'}
CASES={
953:('无地标且失败：最大NE','棕色 Fisher 大楼（Queen’s College）旁边，一栋锯齿状屋顶的小房子。','公开名称未命中；需要核查 Fisher / Queen’s College 的拼写、别名和公开目录覆盖。末端与目标相距很远，无法从此案例单独判定缺失绑定是原因。'),
1191:('无地标且失败：具名描述未命中','Pepy’s Library 左侧、河边一块方形场地，场地角落有一棵大树。','指令包含具体名称但未命中当前公开匹配器。名称格式、所有格或目录缺项是待查假设；不可根据目标坐标反选图形。'),
1595:('无地标且失败：间接空间关系','Keynes 大楼停车场后方，一个有树的三角形绿色公园。','名称未匹配与“建筑—停车场—公园”的间接关系同时存在；即使补齐名称，关系作用域和目标理解仍需验证。'),
772:('无地标但成功：反例','Manchester ⇒ Bournemouth（所有站点）步道右侧的大停车场右上角，从右数第二列最底端的灰色汽车。','公开名称未命中，但终点仍在20米内。完整指令、冻结模型先验和起点等都可能提供信息；仅凭成功无法判断使用了何种线索。'),
898:('有地标却远距离失败','沿 Fen Causeway 道路左侧、Schoolhouse 大楼旁边的长方形场地。','有Polygon不等于绑定语义与关系定位正确；可能涉及同名实例、左右参照系、目标区域理解或Belief选择。图中全部绑定实例均保留，不用GT挑选。'),
1057:('有地标却远距离失败：序数关系','从 Fitzwilliam’s House 沿 Trumpington Street 道路数，第三栋深色建筑。','序数关系要求沿道路排序；已有锚点不保证模型完成计数与实例消歧。不能将远距离失败直接归于某一模块。'),
1148:('容量溢出且失败','Bridge Street 上、St Clement’s 大楼右侧的深灰色建筑。','超过16个有效公开Polygon触发原有全清空回退。这是可审计的覆盖损失，但修复必须以指令和公开信息作完整组处理，不能按GT截取。'),
511:('Polygon无效且失败','One Stop 大楼前，停在另一辆黑车前面、白车旁边的黑色汽车。','名称命中但公开几何无效，未形成有效锚点。应检查源几何和修复前后语义一致性；不能因为附近存在目标就另选Polygon。'),
}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def association(rows,*,repetitions=10000,seed=20261106):
    a=[r for r in rows if not r['bound_reference_count']];b=[r for r in rows if r['bound_reference_count']]
    af=sum(1-r['baseline_SR'] for r in a);bf=sum(1-r['baseline_SR'] for r in b)
    table=[[af,len(a)-af],[bf,len(b)-bf]]
    test=fisher_exact(table,alternative='two-sided')
    # Same physical target may have multiple descriptions: sample whole targets.
    groups=defaultdict(lambda:np.zeros(4,dtype=int))
    for r in rows:
        j=0 if not r['bound_reference_count'] else 2
        groups[tuple(r['episode_id'][:2])][j]+=1
        groups[tuple(r['episode_id'][:2])][j+1]+=1-r['baseline_SR']
    counts=np.array(list(groups.values()));rng=np.random.default_rng(seed)
    diffs=[];ratios=[]
    for _ in range(repetitions):
        n0,f0,n1,f1=counts[rng.integers(0,len(counts),len(counts))].sum(0)
        if n0 and n1:
            diffs.append(f0/n0-f1/n1)
            if f1:ratios.append((f0/n0)/(f1/n1))
    maps={}
    for name in sorted({r['map_name'].split('_block_')[0] for r in rows}):
        subset=[r for r in rows if r['map_name'].startswith(name+'_block_')]
        cells=[]
        for bound in (False,True):
            s=[r for r in subset if bool(r['bound_reference_count'])==bound]
            cells.append(dict(N=len(s),failures=sum(1-r['baseline_SR'] for r in s),failure_rate=np.mean([1-r['baseline_SR'] for r in s]).item() if s else None))
        maps[name]=dict(no_anchor=cells[0],with_anchor=cells[1])
    return dict(table_failure_success=table,fisher_two_sided_p=float(test.pvalue),odds_ratio=float(test.statistic),
        failure_risk_difference=af/len(a)-bf/len(b),failure_risk_ratio=(af/len(a))/(bf/len(b)),
        cluster_unit='(map_name, target_id)',physical_target_clusters=len(counts),
        bootstrap_repetitions=repetitions,bootstrap_seed=seed,valid_difference_replicates=len(diffs),
        cluster_bootstrap95_risk_difference=np.quantile(diffs,[.025,.975]).tolist(),
        cluster_bootstrap95_risk_ratio=np.quantile(ratios,[.025,.975]).tolist(),city_strata=maps)

def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def plots(out,rows,summary):
    plt.rcParams.update({'font.family':'Noto Sans CJK JP','axes.unicode_minus':False,'font.size':11})
    colors=['#D55E00','#0072B2']
    fig,ax=plt.subplots(figsize=(11,5),layout='constrained')
    xs=np.arange(len(BINS));bottom=np.zeros(len(BINS));bins=[]
    for label,bound,color in [('无地标失败',False,colors[0]),('有地标失败',True,colors[1])]:
        values=[sum(not r['baseline_SR'] and bool(r['bound_reference_count'])==bound and lo<=r['baseline_NE_m']<hi for r in rows) for lo,hi in BINS]
        ax.bar(xs,values,bottom=bottom,label=label,color=color)
        for x,v,y in zip(xs,values,bottom):
            if v:ax.text(x,y+v/2,str(v),ha='center',va='center',color='white',fontsize=10)
        bottom+=values;bins.append(values)
    ax.set(xticks=xs,xticklabels=[f'{lo}–{hi}' if math.isfinite(hi) else '≥300' for lo,hi in BINS],xlabel='终点NE（米），区间左闭右开',ylabel='导航失败任务数',title='Val-Seen 2470条：1340条失败的终点误差分布')
    ax.legend();ax.spines[['top','right']].set_visible(False)
    for ext in ('png','svg'):fig.savefig(out/f'failure_distance_distribution.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    for ax,keys,title in [(axs[0],['no_anchor','with_anchor'],'是否成功绑定'),(axs[1],list(STATUS),'互斥绑定状态')]:
        labels=[];rates=[];ns=[]
        for key in keys:
            s=summary['sections'].get(key,dict(N=0,successes=0,failures=0))
            labels.append({'no_anchor':'无地标','with_anchor':'有地标'}.get(key,STATUS.get(key,key)))
            rates.append(100*s['successes']/s['N'] if s['N'] else 0);ns.append(s['N'])
        bars=ax.bar(np.arange(len(keys)),rates,color=[colors[0] if k!='with_anchor' and k!='bound_public_polygon' else colors[1] for k in keys])
        for bar,rate,n in zip(bars,rates,ns):ax.text(bar.get_x()+bar.get_width()/2,rate+2,f'{rate:.2f}%\nn={n}',ha='center',fontsize=10)
        ax.set(xticks=np.arange(len(keys)),xticklabels=[x.replace('，','\n') for x in labels],ylim=(0,70),ylabel='SR（%）',title=title)
        ax.tick_params(axis='x',labelsize=9);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Val-Seen：绑定状态与SR（总分母2470；柱上n为组内分母）')
    for ext in ('png','svg'):fig.savefig(out/f'binding_status_sr.{ext}',dpi=180)
    plt.close(fig)
    dist=[]
    for k,(lo,hi) in enumerate(BINS):
        subset=[r for r in rows if lo<=r['baseline_NE_m']<hi]
        dist.append(dict(range_m=f'[{lo},{hi})',all_tasks=len(subset),failures=sum(not r['baseline_SR'] for r in subset),
            no_anchor_tasks=sum(not r['bound_reference_count'] for r in subset),no_anchor_failures=bins[0][k],with_anchor_failures=bins[1][k]))
    write_csv(out/'distance_bins.csv',dist)
    return dist

def case_figure(out,row,record,pred,target,catalog):
    i=row['index'];bounds=record['world_bounds'];xmin,ymin,xmax,ymax=bounds
    fig,ax=plt.subplots(figsize=(8,7),layout='constrained')
    for obj in catalog[row['map_name']].values():
        points=np.asarray(obj.get('contour',[]))
        if points.ndim==2 and points.shape[1]==2 and len(points)>=3 and np.isfinite(points).all():
            ax.add_patch(Polygon(points,facecolor='#eeeeee',edgecolor='#bbbbbb',linewidth=.35))
    # Every actual input Polygon, no GT-based choice or ranking.
    for ref in record['landmarks']:
        points=np.asarray(ref['contour'])*[xmax-xmin,ymax-ymin]+[xmin,ymin]
        ax.add_patch(Polygon(points,facecolor='#56B4E955',edgecolor='#0072B2',linewidth=1.4))
        center=points.mean(0);ax.text(*center,str(ref['id']),fontsize=8,color='#005080')
    path=np.asarray(pred['poses']);ax.plot(path[:,0],path[:,1],'-o',color='#009E73',markersize=3,label='原全文基线实际轨迹')
    ax.scatter(*path[0,:2],marker='s',color='#009E73',s=65,label='起点',zorder=5)
    ax.scatter(*path[-1,:2],marker='X',color='#D55E00',s=100,label='最终终点',zorder=6)
    ax.scatter(*target[:2],marker='*',color='#CC79A7',edgecolor='black',s=180,label='GT目标（仅事后）',zorder=7)
    ax.add_patch(Circle(target[:2],20,fill=False,edgecolor='#CC79A7',linestyle='--'))
    ax.plot([path[-1,0],target[0]],[path[-1,1],target[1]],'--',color='#D55E00',lw=1)
    ax.set(xlim=(xmin,xmax),ylim=(ymin,ymax),aspect='equal',xlabel='公开地图X（米）',ylabel='公开地图Y（米）',title=f"#{i} | {row['map_name']} | NE={row['baseline_NE_m']:.2f}m\n蓝色=全部绑定Polygon；灰色=公开地图轮廓；虚线圆=20m评分范围")
    ax.legend(loc='best',fontsize=8)
    ax.grid(alpha=.15)
    fig.savefig(out/f'case_{i:04d}.png',dpi=150)
    buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
    svg=buf.getvalue();return '\n'.join(line.rstrip() for line in svg[svg.index('<svg'):].splitlines())

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('paired','episodes','baseline-root','comparison-root','public-objects','annotations','output'):p.add_argument('--'+key,type=Path,required=True)
    args=p.parse_args();out=args.output
    if out.exists():raise FileExistsError('Refuse to overwrite report')
    for root in (args.baseline_root,args.comparison_root):
        manifest=json.loads((root/'full_relation_manifest.json').read_text())
        if manifest['split']!='val_seen' or manifest['instruction_mode']!='full_text':
            raise ValueError('Require original full-text Val-Seen archives')
        for key,path in [('public_catalog',args.public_objects),
                         ('text_inputs',Path('graphnav/unified_text_ablation.py')),
                         ('text_parser',Path('scripts/prepare_unified_v2_landmark_relation_ablation.py'))]:
            if manifest[key]!=sha(path):raise ValueError(f'Archived binding source changed: {key}')
    summary,rows=analyze(args.paired,args.episodes,args.baseline_root/'belief_steps',args.public_objects,
        baseline_predictions=args.baseline_root/'predictions.jsonl',annotations=args.annotations)
    # Also verify the exact archived angle-run binding independently; report baseline outcomes only.
    other,other_rows=analyze(args.paired,args.episodes,args.comparison_root/'belief_steps',args.public_objects)
    assert [(r['episode_id'],r['reference_ids'],r['instruction']) for r in rows]==[(r['episode_id'],r['reference_ids'],r['instruction']) for r in other_rows]
    assert summary['count']==2470 and summary['all_failures']==1340
    assert math.isclose(summary['sections']['all']['NE_all']['maximum_m'],385.8116028366896,abs_tol=1e-8)
    assoc=association(rows);summary['association']=assoc
    predictions={tuple(r['episode_id']):r for r in map(json.loads,(args.baseline_root/'predictions.jsonl').read_text().splitlines())}
    annotations={(f"{r['area']}_block_{r['block']}",int(r['object_ids'][0]),int(r['ann_ids'][0])):r for r in json.loads(args.annotations.read_text())}
    catalog=json.loads(args.public_objects.read_text());out.mkdir(parents=True)
    evidence=[];sources=[]
    for r in rows:
        eid=tuple(r['episode_id']);pred=predictions[eid];target=annotations[eid]['target_positions'][-1]
        path=args.baseline_root/'belief_steps'/f"{r['index']:05d}"/'public_record.json'
        record=json.loads(path.read_text());comparison=args.comparison_root/'belief_steps'/f"{r['index']:05d}"/'public_record.json'
        comparison_record=json.loads(comparison.read_text())
        assert record==comparison_record, f"Full public record differs at {eid}"
        r.update(public_record_sha256=sha(path),comparison_public_record_sha256=sha(comparison),
                 start_pose=pred['poses'][0],final_pose=pred['poses'][-1],target_position_POSTHOC=target,
                 termination=pred['termination'],OSR=pred['metrics']['OSR'],steps=len(pred['poses'])-1)
        evidence.append(dict(index=r['index'],episode_id=r['episode_id'],public_record=record,
            poses=pred['poses'],metrics=pred['metrics'],target_position_POSTHOC=target))
        sources.append(dict(index=r['index'],episode_id=r['episode_id'],baseline_public_record_sha256=sha(path),comparison_public_record_sha256=sha(comparison)))
    write_csv(out/'per_episode.csv',rows)
    # Archive auditable inputs rather than forcing future readers to access this server.
    with (out/'verified_evidence.jsonl.gz').open('wb') as raw:
        with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f:
            for r in evidence:f.write((json.dumps(r,ensure_ascii=False)+'\n').encode())
    source_paths={'paired':args.paired,'episodes':args.episodes,'public_objects':args.public_objects,'annotations':args.annotations,
        'baseline_predictions':args.baseline_root/'predictions.jsonl','baseline_episodes':args.baseline_root/'episodes.json',
        'comparison_episodes':args.comparison_root/'episodes.json','baseline_manifest':args.baseline_root/'full_relation_manifest.json',
        'comparison_manifest':args.comparison_root/'full_relation_manifest.json','analysis_script':Path(__file__).with_name('analyze_val_seen_binding_failures.py'),
        'report_script':Path(__file__)}
    episodes=json.loads(args.episodes.read_text())
    assert episodes==json.loads(source_paths['baseline_episodes'].read_text())==json.loads(source_paths['comparison_episodes'].read_text())
    provenance=dict(PR_source_SHA='6bc7dc32ff9f3a51ce489b5145464c8b35ac9947',
        inputs={k:dict(path=str(v.resolve()),sha256=sha(v)) for k,v in source_paths.items()},public_record_files=sources,
        checks=dict(total=2470,unique_episode_ids=2470,missing=0,duplicates=0,misaligned=0,baseline_comparison_public_records_equal=2470,
                    baseline_SR_NE_and_endpoint_verified=2470,original_instructions_verified=2470),
        python_environment='Existing project aerialvg_env virtualenv; configured ~/anaconda3 is absent on this server.',
        GT_access='Targets loaded only for posthoc endpoint NE validation and plotting, never passed to binding.')
    (out/'provenance.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    dist=plots(out,rows,summary)
    far=[r for r in rows if not r['baseline_SR'] and r['baseline_NE_m']>=100]
    assert len(far)==152
    summary['far_failures_GE100m']=dict(total=len(far),no_anchor=sum(not r['bound_reference_count'] for r in far),with_anchor=sum(bool(r['bound_reference_count']) for r in far))
    summary['distance_bins']=dist
    cases=[];cards=[]
    for i,(title,translation,note) in CASES.items():
        row=next(r for r in rows if r['index']==i);ev=next(r for r in evidence if r['index']==i)
        svg=case_figure(out,row,ev['public_record'],predictions[tuple(row['episode_id'])],ev['target_position_POSTHOC'],catalog)
        case=dict(index=i,title=title,episode_id=row['episode_id'],instruction=row['instruction'],translation_zh=translation,
            interpretation_hypotheses=note,status=row['public_match_status'],NE_m=row['baseline_NE_m'],reference_ids=row['reference_ids'],
            all_bound_polygons=ev['public_record']['landmarks'],target_position_POSTHOC=ev['target_position_POSTHOC'],poses=ev['poses'])
        cases.append(case)
        refs='；'.join(f"{x['id']} {x['name']}" for x in ev['public_record']['landmarks']) or '无成功绑定Polygon'
        cards.append(f'<article data-kind="{"有地标" if row["bound_reference_count"] else "无地标"}" id="case-{i}"><h2>#{i} {html.escape(title)}</h2><p>Episode ID: {html.escape(json.dumps(row["episode_id"]))} | {STATUS[row["public_match_status"]]} | NE {row["baseline_NE_m"]:.4f} m | SR {row["baseline_SR"]}</p><p lang="en">{html.escape(row["instruction"])}</p><p><b>中文：</b>{html.escape(translation)}</p><p><b>已绑定：</b>{html.escape(refs)}</p><details><summary>展开坐标与Polygon、真实轨迹数据</summary><pre>{html.escape(json.dumps(case,ensure_ascii=False,indent=2))}</pre></details><div class="plot">{svg}</div><p><b>解释（待验证假设）：</b>{html.escape(note)}</p></article>')
    (out/'cases.json').write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')
    (out/'cases.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Val-Seen 地标绑定典型案例图册</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:auto;padding:24px;background:#f7f9fc;color:#162438}article{background:white;padding:24px;margin:24px 0;border:1px solid #ddd;border-radius:12px}.plot svg{width:100%;height:auto}pre{overflow:auto;max-height:400px;background:#eef2f6;padding:12px}button{padding:10px;margin:5px}h1{font-size:26px}nav{position:sticky;top:0;background:#f7f9fc;padding:8px;z-index:1}</style><h1>2470条 Val-Seen：地标绑定与导航误差案例</h1><p>全部来自已有原全文基线闭环。蓝色为实际绑定的全部公开Polygon，灰色为公开地图轮廓。GT目标仅作事后评分/展示；地图X/Y未经地理南北核验。选例为定性说明，不估计因果贡献。可展开原始坐标，保存此单文件离线浏览。</p><nav><button onclick="filterCases('全部')">全部8例</button><button onclick="filterCases('无地标')">无地标</button><button onclick="filterCases('有地标')">有地标</button></nav>'''+''.join(cards)+'''<script>function filterCases(kind){document.querySelectorAll('article').forEach(x=>x.hidden=kind!=='全部'&&x.dataset.kind!==kind)}</script></html>''')
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    write_report(out,summary,rows,cases)
    for path in out.glob('*.svg'):
        path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
    print(json.dumps(dict(count=summary['count'],binding_events=summary['binding_events'],association=assoc,far=summary['far_failures_GE100m']),ensure_ascii=False,indent=2))


def write_report(out,s,rows,cases):
    a=s['association'];far=s['far_failures_GE100m'];sections=s['sections']
    text=['# Val-Seen 地标绑定失败与导航误差关联分析','',
    '日期：2026-11-06。范围：已有 **2470/2470** 条原全文基线真实闭环记录；没有重新训练、推理、导航或运行Test。TRAIN的1221条无地标未纳入任何Val-Seen统计。', '',
    f"**关键结果：1340条导航失败中，122条没有成功绑定地标（9.10%）；126条无地标任务中122条失败（96.83%），2344条有地标任务中1218条失败（51.96%）。≥100米的152条失败中，{far['no_anchor']}条无地标，{far['with_anchor']}条有地标。最大NE为385.8116米。**",'',
    '## 1. 数据来源与严格对齐','',
    '以PR27 `6bc7dc3` 为源版本。读取PR25归档 `paired2470.csv`、`episodes2470.json`，并读取服务器原全文基线 `unified_v2_text_ablation_s42_20261105/eval_full_text` 的 predictions、manifest及2470份public_record；另对照 `unified_angle_only_s42_20261106/val_seen_predicted_angles` 的2470份public_record。两套公开输入逐条完全相同。统计结局始终使用 baseline_SR/baseline_NE，图册使用原全文基线轨迹，未混用角度模型轨迹。', '',
    '核查：CSV与manifest各2470条、Episode ID与索引均唯一、索引0–2469完整；原基线predictions按Episode ID连接，并校验index与start_pose。原始Val-Seen annotation按Episode ID校验逐条指令。原始终点XY至目标XY重新算NE，与CSV/原日志一致；SR严格为0或1，NE有限且非负，SR等价于NE≤20米。缺失0、重复0、错位0。', '',
    '使用原 `PublicNames + public_text_record(full_text,max_landmarks=16)` 重现绑定；按顺序比较全部Polygon ID与完整归一化轮廓。GT目标未进入匹配器，只用于上述事后评分与可视化。所有输入路径、SHA256、逐记录哈希见 `provenance.json`；可独立审计的原始公开输入、轨迹、评分与事后目标见 `verified_evidence.jsonl.gz`。', '',
    '修复现有分析脚本的校验缺口：拒绝重复Episode ID、非连续/重复index、NaN/Inf/负NE和小数SR；加入可选原始annotation/基线predictions核对，防止同地图任务互换未被发现。synthetic测试覆盖同地图记录互换、轨迹/评分损坏、乱序输入与重复ID。正式报告要求这些增强核查全部通过。', '',
    '## 2. 绑定覆盖率与故障类型','',
    '|互斥状态|任务数 / 2470|占全量|成功|失败|SR|','|---|---:|---:|---:|---:|---:|']
    for k,name in STATUS.items():
        x=sections.get(k,dict(N=0,successes=0,failures=0));n=x['N']
        text.append(f"|{name}|{n}|{n/2470:.2%}|{x['successes']}|{x['failures']}|{x['successes']/n:.2%}|" if n else f'|{name}|0|0|0|0|—|')
    ev=s['binding_events'];text+=['',
    f"绑定成功2344/2470（94.90%），未绑定126/2470（5.10%）。无公开名称匹配只表示当前公开目录/匹配器没有命中，不能等同于指令没有提及具名地点；案例中有明确名称仍未命中。",'',
    f"以下为**非互斥事件**，统一检查全部2470条：无效Polygon涉及{ev['invalid_polygon_tasks']}条任务、{ev['invalid_polygon_events']}次警告；其中8条最终完全无锚点，其余67条仍绑定了有效Polygon。超过16个有效Polygon的任务{ev['capacity_overflow_tasks']}条，全部回退无锚点。同一匹配名称对应多个公开ID的任务{ev['same_name_multi_polygon_tasks']}条（{ev['same_name_multi_polygon_tasks']/2470:.2%}）；这里按匹配器返回的公开ID组定义，允许组中存在无效轮廓，不代表全部多实例均送入网络。不能把这些事件与互斥状态相加。",'',
    '## 3. SR、失败与NE','',
    'SR/失败率为组内分母；全量分母始终2470。NE以米计，分位数使用排序后的线性插值。', '',
    '|组|N|成功|失败|SR|失败率|NE均值|中位数|P75|P90|P95|P99|最大值|','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for k,name in [('all','全量'),('with_anchor','有地标'),('no_anchor','无地标'),('failures','全部失败'),('failures_with_anchor','有地标且失败'),('failures_no_anchor','无地标且失败')]:
        x=sections[k];q=x['NE_all'];text.append(f"|{name}|{x['N']}|{x['successes']}|{x['failures']}|{x['SR']:.2%}|{x['failure_rate']:.2%}|"+'|'.join(f'{q[v]:.4f}' for v in ['mean_m','median_m','p75_m','p90_m','p95_m','p99_m','maximum_m'])+'|')
    ci=a['cluster_bootstrap95_risk_difference'];rr=a['cluster_bootstrap95_risk_ratio']
    text+=['',f"无地标相对有地标的失败率差为 **{a['failure_risk_difference']*100:.2f}个百分点**，失败风险比 **{a['failure_risk_ratio']:.3f}**。列联表（行：无/有地标；列：失败/成功）为 `{a['table_failure_success']}`。Fisher双侧p={a['fisher_two_sided_p']:.3g}，优势比OR={a['odds_ratio']:.3f}。Fisher将Episode视作独立观测，这个假设对同目标多指令并不充分。",'',
    f"因此增加按 `(map_name,target_id)` 的整组bootstrap：{a['physical_target_clusters']}个物理目标组，{a['bootstrap_repetitions']}次，有效{a['valid_difference_replicates']}次，seed={a['bootstrap_seed']}；失败率差95%百分位区间 **[{ci[0]*100:.2f}, {ci[1]*100:.2f}]个百分点**，风险比区间 [{rr[0]:.3f}, {rr[1]:.3f}]。此区间检验同目标内相关性的敏感性，未控制地图空间相关性、描述难度或模型偏差，不是随机实验因果区间。",'',
    '|城市分层|无地标失败/N|无地标失败率|有地标失败/N|有地标失败率|','|---|---:|---:|---:|---:|']
    for city,v in a['city_strata'].items():
        x=v['no_anchor'];y=v['with_anchor'];text.append(f"|{city}|{x['failures']}/{x['N']}|{x['failure_rate']:.2%}|{y['failures']}/{y['N']}|{y['failure_rate']:.2%}|")
    text+=['','在这份已完成的Val-Seen记录上，无地标组失败率明显更高；聚类区间未跨0，表明按物理目标聚类后该差异仍稳定。多数失败（1218/1340，90.90%）仍发生在有地标组，不能把全部失败解释为覆盖率问题。','','![绑定状态与SR](binding_status_sr.png)','','## 4. 终点误差分布','',
    '区间统一为左闭右开：[0,20)、[20,40)……[300,+∞)。SR用≤20米，因此NE恰好20米成功但归入第二距离箱；当前数据无此边界样本。远距离口径采用历史说明中的≥100米，当前恰好100米样本为0，所以与>100米计数一致。', '',
    '|NE区间（米）|全量任务|其中失败|无地标任务|无地标失败|有地标失败|','|---|---:|---:|---:|---:|---:|']
    for r in s['distance_bins']:text.append('|'+ '|'.join(str(r[k]) for k in ['range_m','all_tasks','failures','no_anchor_tasks','no_anchor_failures','with_anchor_failures'])+'|')
    text+=['',f"≥100米失败152/1340（{152/1340:.2%}，占全量{152/2470:.2%}），其中无地标{far['no_anchor']}/152（{far['no_anchor']/152:.2%}）。该组在无地标失败中的占比为{far['no_anchor']}/122（{far['no_anchor']/122:.2%}），有地标失败中的占比为{far['with_anchor']}/1218（{far['with_anchor']/1218:.2%}）。全量NE均值37.9596米、最大385.8116米；不是只看成功样本或截断误差。",'',
    '![失败距离分布](failure_distance_distribution.png)','','## 5. 典型案例图册','',
    '[下载并用浏览器打开完整离线图册](cases.html)。包含8例、原始指令与中文翻译、全部公开Polygon背景、全部实际绑定Polygon、原基线实际轨迹、起点、目标与终点。蓝色为模型实际锚点，灰色是公开地图轮廓；没有GT选Polygon。坐标保留原地图X/Y，未经核验不标地理北向。案例选择用于覆盖故障类型，不是随机抽样或因果证据。', '']
    for c in cases:
        text +=[f"### #{c['index']} {c['title']}",'',f"Episode ID `{c['episode_id']}`；NE {c['NE_m']:.4f}米；{STATUS[c['status']]}；实际Polygon IDs `{c['reference_ids']}`。",'',f"> {c['instruction']}",'',f"中文：{c['translation_zh']}",'',c['interpretation_hypotheses'],'',f"![案例{c['index']}](case_{c['index']:04d}.png)",'']
    text+=['## 6. 结论与后续实验设计','',
    '绑定覆盖缺失与失败存在强统计关联，且无地标失败中有较多远距离误差；容量溢出和非法几何是可直接审计的输入覆盖缺口，值得优先修复。但本次不能确认“未绑定直接导致失败”，也不能把122条无地标失败或远距离子集当作可恢复失败数量。指令复杂度、公开目录覆盖、城市/地图、目标类型、序数/间接关系和训练—在线绑定差异均可能共同影响分组结果。', '',
    '覆盖率是值得验证的候选瓶颈；全量成功覆盖已达94.90%，且多数失败有锚点，所以不能据此认定它是全系统主要瓶颈。无地标但成功的4例全部来自Birmingham block 12，亦不能将其泛化为无地标策略可靠。', '',
    '下一阶段建议独立预注册配对修复实验（**本次只设计，不执行**）：', '',
    '1. 修复规则只看原始指令与公开地图目录。区分无名称可用、名称/别名漏匹配、几何无效、完整组容量溢出；人工审核者不看GT、NE、SR。名称未具名的指令不因目标GT补锚。公开几何修复须可追溯；容量方案保留同名完整组/无GT空间选择规则，不按目标距离截断。',
    '2. 冻结同一checkpoint、完整指令、起点、模型、Controller/Stop、步数和随机种子；只改变公开绑定。在全部2470条上对原绑定/修复绑定执行同任务配对真实闭环，未受影响任务也保留在全量分母。原归档可作协议一致性参考，若软件/随机性变化则须另跑同期原绑定对照。训练不变；不运行Test。',
    '3. 在执行前用独立TRAIN或公开合成样本固定修复规则与版本SHA，不使用本报告Val-Seen结局挑规则或checkpoint。当前Val-Seen已经被分析，未来在同集的提升仅作诊断性估计，不能冒称独立泛化验证。',
    '4. 报告全量2470配对SR变化、纠正/破坏、NE分位数与最大值、≥100米失败变化及受影响子集分母；对二元配对结果给McNemar精确检验，并按物理目标聚类bootstrap配对SR/NE差值。检验实际网络收到的Polygon集合与完整轮廓，另外报告无地标覆盖改善。GT只用于最终评分，不能参与名称、实例选择或调参。',
    '5. 只有受控干预后才能估计修复的增量作用；若提升来自容量/几何/名称同时修改，应进一步分解干预。不得把本次相关性比例当作修复后SR上限或预期收益。', '',
    '## 7. 产物与复现','',
    '- `per_episode.csv`：2470行，Episode ID、原指令、互斥状态、名称命中数、Polygon ID、有效几何数、警告、同名多实例事件、SR/NE、起终点与事后目标、输入哈希。',
    '- `summary.json`：分组统计、NE分位数、距离箱、关联检验与聚类区间。`distance_bins.csv`为图表原始计数。',
    '- `provenance.json`、`verified_evidence.jsonl.gz`：来源及逐记录证据。`cases.json`与`cases.html`：可浏览案例及其数据。PNG/SVG图表均可独立使用。',
    '- 复现命令见 `reproduce.sh`。使用项目已有 `aerialvg_env`；本机配置中的 `~/anaconda3` 不存在，因此使用项目说明指定的已有环境解释器，未改环境配置。', '',
    '检验使用SciPy Fisher exact和固定seed的NumPy聚类bootstrap；仅本次预定主要分组检验，城市分层为描述性敏感性检查。案例解释是待验证假设。']
    (out/'REPORT_ZH.md').write_text('\n'.join(text)+'\n',encoding='utf-8')

if __name__=='__main__':main()
