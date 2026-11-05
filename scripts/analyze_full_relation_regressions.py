"""Strictly post-rollout failure attribution; target labels never enter policy."""
import argparse,json
from pathlib import Path
import numpy as np
from graphnav.checkpointing import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    raw=json.loads((a.data_root/'processed_citynav/citynav_val_seen.json').read_text())
    results={arm:{x['index']:x for x in map(json.loads,(a.run/f'eval_{arm}/predictions.jsonl').read_text().splitlines())} for arm in ('relation','rgb')}
    items=[]
    for idx,before in results['relation'].items():
        after=results['rgb'][idx]
        if bool(before['metrics']['SR'])==bool(after['metrics']['SR']):continue
        target=np.array(raw[idx]['target_positions'][0][:2]);d=after['step_diagnostics'];events=[]
        for step,x in enumerate(d):
            if x['stable_navigation_goal']['event']!='replanned':continue
            previous=d[step-1]['selected_goal_xy'] if step else x['selected_goal_xy']
            events.append(dict(step=step,old_goal=previous,new_goal=x['selected_goal_xy'],
                old_GT_distance_m=float(np.linalg.norm(np.array(previous)-target)),new_GT_distance_m=float(np.linalg.norm(np.array(x['selected_goal_xy'])-target)),
                stop_probability=x['stop_probability'],predicted_goal_distance_m=x['predicted_goal_distance'],tracking=x.get('landmark_tracking'),
                visual_delta_max=x.get('pgvb_delta_max_abs'),stable=x['stable_navigation_goal']))
        items.append(dict(index=idx,change='correction' if after['metrics']['SR'] else 'damage',instruction=raw[idx]['descriptions'][0],map_name=after['map_name'],
            before=before['metrics'],after=after['metrics'],termination=after['termination'],events=events,
            independently_verified_target_RGB_identity=False,public_polygon_error='unknown: no independent reference-instance or pixel mask truth'))
    summary=dict(corrections=sum(x['change']=='correction' for x in items),damage=sum(x['change']=='damage' for x in items),
        damage_entered20m=sum(x['change']=='damage' and x['after']['OSR']==1 for x in items),
        damage_with_replan=sum(x['change']=='damage' and bool(x['events']) for x in items),GT_posthoc_only=True)
    atomic_json(dict(summary=summary,tasks=items),a.output/'RGB_regression_audit.json')
    with (a.output/'REPORT_zh.md').open('a') as f:
        f.write('\n## 关键归因：Relation与RGB\n\nRelation的SR纠正/破坏为0/0；2470条中只改变1条初始Top-1，几何Top-2 logit间隔中位数0.11413，而Relation最大修正中位数仅0.00423。按逐帧概率计算，平均Belief质心变化约0.00194m，不能把非零梯度/残差称为有效关系推理。\n\nRGB纠正2条、破坏8条，净-6。8条破坏全部曾进入20m且都出现视觉触发重规划（690、999、1016、1037、1210、1213、1766、2089）。并非控制器不会到达正确Goal：它们最终都在新的错误Goal处Policy Stop。没有Horizon推延或阈值修改。\n\n例如任务690：原Goal到GT约12.1m；第5步到原Goal只剩0.088m，Stop概率0.386，但仍未满足原1mm到达守卫。第6步视觉确认后重规划到GT约28m的错误Goal，最终在该点正常Stop。公开ROI的视觉AUC接近1只证明其地图区域可区分，不证明白车目标身份正确。该案例不能作为放宽Stop的依据。\n\n结构限制：现有PGVB以地标名称/类别/轮廓评分，缺少目标外观与实例条件；其地标证据直接叠加到Target分布，Relation空间变换又接近零。角色不匹配是有代码与轨迹支持的解释，但没有独立像素GT，不能把全部错误进一步判为真实实例识别错误或Polygon绑定错误。\n\n**下一步优先**：保留训练后Geometry作为本轮候选；先在TRAIN冻结Geometry，验证Relation能否通过受控关系监督形成有效空间残差，并检查PGVB的目标条件与可观测监督。不要继续加Attention层、增加Epoch或调Stop阈值；没有新增分支的独立收益前不替换main。\n\n统计范围：Geometry本轮提升+2.47pp，地图块bootstrap 95%区间约[+0.83,+4.35]pp；仅seed42，不能替代多种子或Val-Unseen泛化验证。本轮没有区分普通Geometry继续训练与联合训练对Geometry的间接作用，需要另外的TRAIN训练对照才能归因。\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
