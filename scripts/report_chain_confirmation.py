"""One-shot paired TRAIN confirmation report; never chooses model parameters."""
import argparse,itertools,json,math
from pathlib import Path
from collections import Counter
import numpy as np


def intervals(delta,maps):
    rng=np.random.default_rng(20261104);delta=np.asarray(delta,float)
    target=np.mean(delta[rng.integers(0,len(delta),(10000,len(delta)))],axis=1)
    names=sorted(set(maps));cluster=[]
    for draw in itertools.product(names,repeat=len(names)):
        ids=[i for name in draw for i,m in enumerate(maps) if m==name]
        cluster.append(float(delta[ids].mean()))
    return dict(target_bootstrap95=np.quantile(target,[.025,.975]).tolist(),map_bootstrap95=np.quantile(cluster,[.025,.975]).tolist(),maps=len(names),map_warning='Only three maps; cluster interval is descriptive, not strong significance evidence')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('run','selection','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve one-shot report')
    category={x['index']:x['category'] for x in json.loads(a.selection.read_text())}
    records={};metrics={}
    for method in ('original','real','gray-rgb','rgb-shuffle'):
        directory=a.run/method;report=json.loads((directory/'metrics.json').read_text())
        rows=[json.loads(x) for x in (directory/'predictions.jsonl').read_text().splitlines()]
        if len(rows)!=64 or report['episodes']!=64 or len({x['index'] for x in rows})!=64:raise ValueError('complete64 required')
        records[method]={x['index']:x for x in rows};metrics[method]=report
    base=records['original'];paired={};table=[]
    for method,rows in records.items():
        if set(rows)!=set(base) or set(rows)!=set(category):raise ValueError('unpaired selection')
        corrections=[];harms=[];delta=[];ne_delta=[];maps=[];by_category={}
        for index,x in rows.items():
            b=base[index]
            if x['poses'][0]!=b['poses'][0] or x['step_diagnostics'][0]['observed_rgb_sha256']!=b['step_diagnostics'][0]['observed_rgb_sha256']:raise ValueError('initial observation mismatch')
            if any(abs(d['camera_pose'][3])>1e-6 for d in x['step_diagnostics']):raise ValueError('camera yaw changed')
            success=x['metrics']['SR'];original=b['metrics']['SR'];change=int(success)-int(original)
            if change>0:corrections.append(index)
            if change<0:harms.append(index)
            delta.append(change);ne_delta.append(x['metrics']['NE']-b['metrics']['NE']);maps.append(x['map_name'])
            table.append(dict(method=method,index=index,category=category[index],map_name=x['map_name'],success=success,baseline_success=original,ne_m=x['metrics']['NE'],baseline_ne_m=b['metrics']['NE'],termination=x['termination'],**x['posthoc_goal_analysis']))
        for cat in sorted(set(category.values())):
            subset=[x for i,x in rows.items() if category[i]==cat];by_category[cat]=dict(episodes=len(subset),SR=float(np.mean([x['metrics']['SR'] for x in subset])),NE=float(np.mean([x['metrics']['NE'] for x in subset])))
        n=len(corrections)+len(harms);k=min(len(corrections),len(harms));pvalue=min(1.,2*sum(math.comb(n,j) for j in range(k+1))/2**n) if n else 1.
        paired[method]=dict(report=metrics[method],corrections=corrections,harms=harms,net=len(corrections)-len(harms),delta_SR=float(np.mean(delta)),SR_intervals=intervals(delta,maps),NE_delta_m=float(np.mean(ne_delta)),NE_intervals=intervals(ne_delta,maps),mcnemar_exact_independent_targets_p=pvalue,by_category=by_category,
            never_goal_correct_failed=sum(not x['posthoc_goal_analysis']['ever_correct'] and not x['metrics']['SR'] for x in rows.values()),once_goal_correct_failed=sum(x['posthoc_goal_analysis']['ever_correct'] and not x['metrics']['SR'] for x in rows.values()))
    a.output.mkdir(parents=True);(a.output/'paired_results.json').write_text(json.dumps(paired,indent=2));(a.output/'per_episode.json').write_text(json.dumps(table,indent=2))
    lines=['# 新64个TRAIN物理目标：冻结独立确认','',
           '排除已有424个目标，新头拟合及开发地图之外的3张TRAIN地图。31Car、30Building、3Ground，无Parking。原A3仍有合法TRAIN预训练，不是正式未见评测。此后不依据该确认集调参。','',
           '|方法|SR|OSR|SPL|NE m|nDTW|纠正/破坏|','|---|---:|---:|---:|---:|---:|---:|']
    for method,d in paired.items():
        m=d['report']['metrics'];lines.append(f"|{method}|{m['SR']:.2%}|{m['OSR']:.2%}|{m['SPL']:.4f}|{m['NE']:.4f}|{m['nDTW']:.4f}|{len(d['corrections'])}/{len(d['harms'])}|")
    real=paired['real'];ci=real['SR_intervals']['target_bootstrap95'];cluster=real['SR_intervals']['map_bootstrap95']
    lines+=['',f"真实RGB相对Original净收益{real['net']}条；配对目标bootstrap SR差95%区间[{100*ci[0]:.2f}, {100*ci[1]:.2f}]个百分点；地图分组区间[{100*cluster[0]:.2f}, {100*cluster[1]:.2f}]个百分点。仅3张地图，不能夸大统计显著性。",'',
            '已核对全部方法初始位姿、原始RGB像素SHA一致，逐决策相机yaw=0。不同方法后续轨迹/观测可以因动作不同而分叉，不能要求后续RGB相同。',
            '正常模型只收到原协议公开地图、指令/公开解析短语与当前真实观测；GT只在完成rollout后评分。SR沿用最终XY≤20m，不要求学习Stop触发，需结合policy_stop/SPL解释。',
            'gray-rgb编码前恒灰、rgb-shuffle编码前打乱16px像素块；所有模型权重及正常质量/几何生成保持冻结。它们属于分布外诊断干预。',
            '完整类别指标、纠正/破坏清单、NE配对区间及Goal曾正确后丢失记录见JSON。当前不是Val128，更不是全量Val/Test成绩。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps({k:dict(SR=v['report']['metrics']['SR'],net=v['net']) for k,v in paired.items()}))


if __name__=='__main__':main()
