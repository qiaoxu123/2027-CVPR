"""CPU-only parameter census of the existing frozen full-model checkpoint."""
from types import SimpleNamespace
import torch
from graphnav.full_experiment import new_model
from graphnav.stage1 import GraphNavStage1
from scripts.build_paper_experiments import MAIN, save, load, sha


def run():
    path = MAIN / 'runs/graphnav_v1_stage1/best.pt'
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = GraphNavStage1(new_model(SimpleNamespace(**state['a3_args']), 'cpu'),
                          residual_bound=state['protocol']['residual_bound'])
    model.load_state_dict(state['model'], strict=True)
    metrics = load(MAIN / 'runs/graphnav_v1_stage1/full_unseen/metrics.json')
    policy = sum(p.numel() for p in model.parameters())
    bert, vision = metrics['bert_backbone_parameters'], metrics['visual_backbone_parameters']
    names = {'a3': 'polygon_language_goal_localizer', 'matcher': 'visual_appearance_matcher',
             'history': 'causal_quality_history', 'trajectory': 'candidate_trajectory_and_stop'}
    result = dict(checkpoint=str(path), checkpoint_sha256=sha(path), policy_parameters=policy,
                  policy_components={names[k]: sum(p.numel() for p in m.parameters()) for k, m in model.named_children()},
                  frozen_bert_parameters=bert, frozen_siglip_parameters=vision,
                  all_resident_parameters=policy + bert + vision,
                  command='python -m scripts.count_polybelief_parameters',
                  note='CPU strict state loading; no training or inference. Includes inactive legacy branches; all parameters frozen at inference. Backbone text and image towers counted together exactly as runtime model.parameters().')
    save('parameter_count.json', result)
    return result


if __name__ == '__main__':
    print(run())
