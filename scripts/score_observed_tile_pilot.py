"""Post-seal TRAIN position scoring. No verified pixel-instance labels exist."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('selection','forward','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous scores')
    seal=json.loads((a.forward/'prediction_seal.json').read_text())
    if not seal['complete'] or seal['GT_read'] or seal['predictions_sha256']!=file_sha256(a.forward/'predictions_public.pt'):raise ValueError('valid blind prediction seal required')
    predictions=torch.load(a.forward/'predictions_public.pt',map_location='cpu',weights_only=True)
    public=torch.load(a.selection/'public_inputs.pt',map_location='cpu',weights_only=True)
    # Private labels are opened only after verifying the prediction seal.
    private=json.loads((a.selection/'private_scoring.json').read_text())
    if set(private)!=set(predictions):raise ValueError('same complete selected cases required')
    output={};methods=list(next(iter(predictions.values())))
    for method in methods:
        rows=[]
        for key,case in predictions.items():
            pred=case[method];label=private[key];target=torch.tensor(label['target_world'][:2],dtype=torch.float32);world=pred['local_world'];valid=pred['valid']
            distance=(world-target).norm(dim=-1)
            ne=float((torch.tensor(pred['goal_world'])-target).norm());positive_world=(public[key]['global_world']-target).norm(dim=-1)<=20
            row=dict(case_key=key,category=label['category'],label_quality=label['label_quality'],human_pixel_gt=False,
                goal_ne_m=ne,goal_success=ne<=20,goal_world=pred['goal_world'],valid_candidates=pred['valid_candidates'],
                global_20m_probability_mass=float(pred['goal_logprob'].exp()[positive_world].sum()),
                closest_candidate_m=float(distance[valid].min()) if valid.any() else None,
                spatial_coverage_1m=bool(((distance<=1)&valid).any()),spatial_coverage_20m=bool(((distance<=20)&valid).any()),
                head_seconds=pred['head_seconds'])
            for score_name in ('visual_logits','native_similarity'):
                order=pred[score_name].masked_fill(~valid,float('-inf')).argsort(descending=True)[:int(valid.sum())]
                for radius in (1,20):
                    for k in (1,5,15):row[f'{score_name}_world_{radius}m_top{k}']=bool((distance[order[:k]]<=radius).any())
            rows.append(row)
        output[method]=rows
    summary={}
    for method,rows in output.items():
        summary[method]=dict(cases=len(rows),correct=sum(x['goal_success'] for x in rows),goal_r1=float(np.mean([x['goal_success'] for x in rows])),goal_ne_m=float(np.mean([x['goal_ne_m'] for x in rows])),
            probability_mass_20m=float(np.mean([x['global_20m_probability_mass'] for x in rows])),
            spatial_coverage_1m=sum(x['spatial_coverage_1m'] for x in rows),spatial_coverage_20m=sum(x['spatial_coverage_20m'] for x in rows),
            visual_world_20m_top1=sum(x['visual_logits_world_20m_top1'] for x in rows),visual_world_20m_top5=sum(x['visual_logits_world_20m_top5'] for x in rows))
    paired={}
    for before,after in [('Whole384_144','Native768_144'),('Interpolated768_144','Native768_144'),('Interpolated768_576','Native768_576'),('Native768_144','Native768_576')]:
        first={x['case_key']:x for x in output[before]};second={x['case_key']:x for x in output[after]}
        paired[before+' -> '+after]=dict(corrected=[k for k in first if second[k]['goal_success'] and not first[k]['goal_success']],
            harmed=[k for k in first if first[k]['goal_success'] and not second[k]['goal_success']],
            mean_ne_delta_m=summary[after]['goal_ne_m']-summary[before]['goal_ne_m'])
    a.output.mkdir(parents=True)
    atomic_json(dict(summary=summary,paired=paired,rows=output,source_sha256=file_sha256(__file__),prediction_seal_sha256=file_sha256(a.forward/'prediction_seal.json'),
        warning='Exploratory biased TRAIN fit24; world position metrics only, not pixel-instance matching or navigation SR'),a.output/'results.json')
    print(json.dumps(dict(summary=summary,paired=paired)))


if __name__=='__main__':main()
