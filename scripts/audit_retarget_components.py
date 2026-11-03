"""Diagnostic parameter interventions on fixed TRAIN Dev; never deploy hybrids."""
import argparse
import itertools
import json
from pathlib import Path
import torch
from graphnav.checkpointing import file_sha256
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records, evaluate

GROUPS = {'visual': ('image.', 'visual_score.'), 'text': ('text.',),
          'geometry': ('geometry.', 'spatial_score.')}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve diagnostic')
    torch.set_num_threads(4)
    cfg=json.loads((a.suite/'soft_instance_s0/config.json').read_text())['config']
    roles=torch.load(cfg['role_vectors'],map_location='cpu',weights_only=True)
    dev=records(Path(cfg['cache']),Path(cfg['dev_labels']),'dev',roles,'reference')
    base=torch.load(cfg['warm_start'],map_location='cpu',weights_only=True)['model']
    reference=WorldLikelihoodPositionHead().cuda().eval();reference.load_state_dict(base)
    baseline=evaluate(reference,dev,'cuda',8)
    results={};weights={}
    for variant in ('coarse','soft_instance'):
        path=a.suite/f'{variant}_s0/final.pt'
        trained=torch.load(path,map_location='cpu',weights_only=True)['model']
        assert set(base)==set(trained)
        weights[variant]={'sha256':file_sha256(path),'parameter_l2_changes':{}}
        for group,prefixes in GROUPS.items():
            weights[variant]['parameter_l2_changes'][group]=sum(float((trained[k]-base[k]).square().sum()) for k in base if k.startswith(prefixes))**.5
        combinations={}
        for flags in itertools.product((False,True),repeat=3):
            active=[g for g,f in zip(GROUPS,flags) if f]
            selected=tuple(prefix for g in active for prefix in GROUPS[g])
            mixed={k:trained[k] if k.startswith(selected) else v for k,v in base.items()}
            model=WorldLikelihoodPositionHead().cuda().eval();model.load_state_dict(mixed,strict=True)
            scored=evaluate(model,dev,'cuda',8)
            base_rows={r['frame_key']:r for r in baseline['rows']}
            scored['paired_to_frozen']={
                'corrected':[r['frame_key'] for r in scored['rows'] if r['success'] and not base_rows[r['frame_key']]['success']],
                'harmed':[r['frame_key'] for r in scored['rows'] if not r['success'] and base_rows[r['frame_key']]['success']],
                'mean_ne_delta_m':scored['metrics']['goal_ne_m']-baseline['metrics']['goal_ne_m']}
            combinations['+'.join(active) or 'frozen']=scored
        results[variant]=combinations
    a.output.mkdir(parents=True)
    (a.output/'results.json').write_text(json.dumps(dict(results=results,weights=weights,source_sha256=file_sha256(__file__),
        warning='TRAIN Dev mechanism interventions only; shared text affects both branches; not deployable model selection'),indent=2))
    lines=['# 新增训练的参数归因诊断','','固定原92条TRAIN开发观察，仅离线替换参数。所有组合均为反事实诊断，不作为新部署模型，不根据结果调权重。文本投影同时影响视觉与几何分支，干预不是完全独立的物理机制。','','|训练|替换的新参数组|Goal R1|NE(m)|纠正/破坏原冻结模型|','|---|---|---:|---:|---:|']
    for variant,combinations in results.items():
        for name,s in combinations.items():
            m=s['metrics'];pair=s['paired_to_frozen']
            lines.append(f"|{variant}|{name}|{m['goal_r1']:.2%}|{m['goal_ne_m']:.4f}|{len(pair['corrected'])}/{len(pair['harmed'])}|")
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__=='__main__':main()
