"""Report the exact fixed Val-Seen128 pair and enforce the predeclared full gate."""
import argparse,json
from pathlib import Path
import numpy as np


def full_evaluation_gate(original_successes,ours_successes):
    if any(type(x) is not int or x<0 or x>128 for x in (original_successes,ours_successes)):
        raise ValueError('integer success counts from exact128 pair required')
    return dict(passed=ours_successes>=84 and ours_successes>original_successes,
                original_successes=original_successes,ours_successes=ours_successes,episodes=128,
                minimum_ours_successes=84,requires_more_successes_than_original=True)


def grouped_interval(delta,groups):
    delta=np.asarray(delta,float)
    if len(delta)!=len(groups) or not len(delta):raise ValueError('paired nonempty group data required')
    names=list(dict.fromkeys(groups));sums=np.array([delta[np.array([g==name for g in groups])].sum() for name in names]);sizes=np.array([sum(g==name for g in groups) for name in names])
    counts=np.random.default_rng(0).multinomial(len(names),np.ones(len(names))/len(names),size=10000)
    values=(counts@sums)/(counts@sizes)
    return np.quantile(values,[.025,.975]).tolist()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous pilot report')
    manifest=json.loads((a.run/'run_manifest.json').read_text())
    if manifest['split']!='val_seen' or manifest['purpose']!='fixed_val_seen_128':raise ValueError('frozen Val-Seen pilot only')
    data={};reports={}
    for method in ('original','real'):
        reports[method]=json.loads((a.run/method/'metrics.json').read_text());rows=[json.loads(x) for x in (a.run/method/'predictions.jsonl').read_text().splitlines()]
        if len(rows)!=128 or len({x['index'] for x in rows})!=128 or reports[method]['episodes']!=128:raise ValueError('complete exact128 required')
        data[method]={x['index']:x for x in rows}
    if set(data['original'])!=set(data['real']):raise ValueError('episode pairing mismatch')
    paired=[];corrections=[];harms=[]
    for index,base in data['original'].items():
        ours=data['real'][index]
        if base['poses'][0]!=ours['poses'][0] or base['step_diagnostics'][0]['observed_rgb_sha256']!=ours['step_diagnostics'][0]['observed_rgb_sha256']:raise ValueError('initial pose/RGB mismatch')
        for record in (base,ours):
            if any(abs(d['camera_pose'][3])>1e-6 for d in record['step_diagnostics']):raise ValueError('camera yaw changed')
        b=base['metrics']['SR'];o=ours['metrics']['SR']
        if b not in (0.,1.) or o not in (0.,1.):raise ValueError('binary SR required')
        if o>b:corrections.append(index)
        if b>o:harms.append(index)
        paired.append(dict(index=index,physical_group=tuple(base['episode_id'][:2]),map_name=base['map_name'],original_SR=b,ours_SR=o,original_NE=base['metrics']['NE'],ours_NE=ours['metrics']['NE'],original_goal=base['posthoc_goal_analysis'],ours_goal=ours['posthoc_goal_analysis']))
    successes={m:sum(int(x['metrics']['SR']) for x in rows.values()) for m,rows in data.items()}
    gate=full_evaluation_gate(successes['original'],successes['real']);delta=[x['ours_SR']-x['original_SR'] for x in paired];groups=[x['physical_group'] for x in paired];maps=[x['map_name'] for x in paired]
    result=dict(split='val_seen',dataset_version='processed_citynav',episodes=128,full_split_performance_claim=False,gate=gate,reports=reports,corrections=corrections,harms=harms,net=len(corrections)-len(harms),
                physical_targets=len(set(groups)),maps=len(set(maps)),paired_SR_delta=float(np.mean(delta)),physical_group_bootstrap95=grouped_interval(delta,groups),map_bootstrap95=grouped_interval(delta,maps),per_episode=paired,
                warning='Fixed128 only; no tuning/checkpoint selection on Val; Test not accessed')
    a.output.mkdir(parents=True);(a.output/'results.json').write_text(json.dumps(result,indent=2));(a.output/'full_gate.json').write_text(json.dumps(gate,indent=2))
    lines=['# 首次Val-Seen128固定配对结果','',
           '源划分为原论文processed_citynav Val-Seen2470条，本次只报告预先固定的128条，不能作为全量成绩。模型/解析/控制器在运行前冻结，不用Val挑候选或调参数。','',
           '|方法|成功数|SR|OSR|SPL|NE m|nDTW|','|---|---:|---:|---:|---:|---:|---:|']
    for method,label in [('original','Original'),('real','冻结V3高度过滤')]:
        m=reports[method]['metrics'];lines.append(f"|{label}|{successes[method]}/128|{m['SR']:.2%}|{m['OSR']:.2%}|{m['SPL']:.4f}|{m['NE']:.4f}|{m['nDTW']:.4f}|")
    ci=result['physical_group_bootstrap95'];mci=result['map_bootstrap95']
    lines+=['',f"纠正{len(corrections)}条、误改{len(harms)}条，净收益{result['net']}；128个Episode含{result['physical_targets']}个物理目标、{result['maps']}张地图。配对SR差的物理目标分组bootstrap95%区间[{100*ci[0]:.2f}, {100*ci[1]:.2f}]个百分点，地图分组区间[{100*mci[0]:.2f}, {100*mci[1]:.2f}]个百分点。",'',
            '预设门槛是至少84/128（严格超过65%），且成功数优于Original。'+('门槛通过；后续只能沿用此冻结模型扩大评估，不把小样本当全量或显著性证明。' if gate['passed'] else '门槛未通过：不启动全量评估，也不根据这128条更换模型/阈值。'),'',
            '全部配对初始位姿与原始RGB SHA一致，逐步相机yaw0。后续RGB随各自真实动作变化，不是教师轨迹。GT只在rollout结束后评分；Original保留原CandidateMemory，新头当前观测方案，不将差异唯一归因于某单一模块。',
            'SR为原协议最终XY20m内，不能与Stop触发率混同。完整Goal丢失、Stop/时延/显存与逐Episode对照在results.json。没有访问Test。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n');print(json.dumps(dict(gate=gate,net=result['net'],SR={m:r['metrics']['SR'] for m,r in reports.items()})))


if __name__=='__main__':main()
