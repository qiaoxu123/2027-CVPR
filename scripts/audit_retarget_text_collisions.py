"""Find identical local text inputs with different TRAIN target-core requirements.

GT labels are read only for posthoc diagnosis, never to form language features.
Identical local visual inputs do not imply identical global A0 priors.
"""
import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path
import torch
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--teacher',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--trained-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous audit')
    labels=torch.load(a.teacher/'loss_labels_private.pt',map_location='cpu',weights_only=True)
    queries=torch.load(a.inputs/'public_inputs.pt',map_location='cpu',weights_only=True)
    roles=json.loads((a.teacher/'roles.json').read_text())
    tasks={r['task_key']:r for r in map(json.loads,(a.teacher/'public_tasks.jsonl').read_text().splitlines())}
    config=json.loads(a.trained_config.read_text())
    excluded=set(config['dataset']['skipped_role_queries'])
    frames=defaultdict(list)
    for link in json.loads((a.teacher/'public_links.json').read_text()):
        if link['task_key'] not in excluded:frames[link['source_frame_key']].append(link)
    collisions=[];comparisons=0;eligible_comparisons=0
    for frame, links in frames.items():
        for left,right in itertools.combinations(links,2):
            if left['task_key']==right['task_key']:continue
            comparisons+=1
            qa,qb=queries[left['task_key']],queries[right['task_key']]
            la,lb=labels[left['sample_key']],labels[right['sample_key']]
            eligible=la['soft_preference']>.5 and lb['soft_preference']>.5
            eligible_comparisons+=eligible
            if not (torch.equal(qa['target_text'],qb['target_text']) and torch.equal(qa['context_text'],qb['context_text'])):continue
            pa=la['weak_instance_labels']==1;pb=lb['weak_instance_labels']==1
            disjoint=bool(pa.any() and pb.any() and not (pa&pb).any())
            entry=dict(source_frame_key=frame,task_keys=[left['task_key'],right['task_key']],
                target_text=roles[left['task_key']]['target_text'],reference_text=roles[left['task_key']]['reference_text'],
                original_instructions=[tasks[x['task_key']]['instruction'] for x in (left,right)],
                eligible_both=eligible,disjoint_positive_cores=disjoint,
                local_top1_can_satisfy_both=not disjoint,
                global_prior_equal=torch.equal(qa['prior_logits'],qb['prior_logits']),
                full_text_equal=torch.equal(qa['full_text'],qb['full_text']),
                teacher_world_separation_m=float((torch.tensor(la['target_world'])-torch.tensor(lb['target_world'])).norm()),
                positive_indices=[pa.nonzero().flatten().tolist(),pb.nonzero().flatten().tolist()],
                annotation_warning='weak map-core supervision, not verified pixel-instance truth')
            collisions.append(entry)
    result=dict(added_samples=sum(map(len,frames.values())),source_frames=len(frames),same_frame_query_pairs=comparisons,
        eligible_same_frame_query_pairs=eligible_comparisons,identical_text_pairs=len(collisions),
        distinct_required_top1_pairs=sum(x['eligible_both'] and x['disjoint_positive_cores'] for x in collisions),
        affected_sample_queries=len({(x['source_frame_key'],k) for x in collisions for k in x['task_keys']}),
        affected_physical_queries=len({k for x in collisions for k in x['task_keys']}),
        cases=collisions,inputs_sha256=file_sha256(a.inputs/'public_inputs.pt'),private_labels_sha256=file_sha256(a.teacher/'loss_labels_private.pt'),
        source_sha256=file_sha256(__file__))
    a.output.mkdir(parents=True)
    (a.output/'results.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    lines=['# 局部视觉文本输入碰撞审计','',
        f"实际新增训练样本{result['added_samples']}，涉及{result['source_frames']}张来源图；同图不同查询共{comparisons}对。完全相同目标/参考向量{len(collisions)}对，其中两边均参与软实例监督且正核心不相交{result['distinct_required_top1_pairs']}对，涉及{result['affected_physical_queries']}个查询。配对和查询不是独立导航样本。",'',
        '这是输入信息丢失的实测证据：同图、相同视觉特征和相同目标/参考向量，局部visual_score必定相同，不可能让同一个确定Top1同时落到两个不相交目标核心。并不意味着全部损失必然无解（多个高分候选可满足软平均margin），也不意味着完整Goal必然相同：A0全文和空间分支仍可能区分。核心是弱地图标签，不能冒充真实像素实例标签。','',
        '## 原始指令与实际局部文本','']
    for x in collisions:
        lines += [f"### {x['source_frame_key']} / {' vs '.join(x['task_keys'])}",'',f"- 目标输入：`{x['target_text']}`",f"- 参考输入：`{x['reference_text']}`",f"- 原文A：{x['original_instructions'][0]}",f"- 原文B：{x['original_instructions'][1]}",f"- 地图教师中心距离{x['teacher_world_separation_m']:.2f}m，正核心不相交={x['disjoint_positive_cores']}；完整文本向量相同={x['full_text_equal']}。",'']
    lines += ['## 下一步方案及边界','',
        '优先修复目标语义接口，保留颜色/类别，同时把目标自身的序数、排列和实例限定词送入局部视觉分支；参考地标/区域继续单独表示。未知参考系保留语言，不强制转为世界方向。A0保持原全文。先验证输入碰撞被消除、原目标属性不丢失，再用相同训练预算和独立TRAIN确认。','',
        '不能直接把全文塞给所有角色当成已验证方案：此前full_text消融并无稳定收益。此次只证明特定序数信息被删除，不证明修复后SigLIP2一定具有数数能力，更不能据5对解释全部导航失败。若现有特征无法表示车排/序数，需明确报告，而非继续强加不可辨别标签。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':main()
