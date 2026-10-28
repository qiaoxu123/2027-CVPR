"""Opened v2FIT-only diagnosis; never read VERIFY tensors or choose a gate."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.revision_fusion import bounded_belief_mix,batch_belief_mix


def main():
    torch.set_num_threads(2);root=BASE/'revision_v2';manifest=json.loads((root/'confirmation/manifest.json').read_text())
    candidate=json.loads((root/'calibration/candidate_calibration.json').read_text())
    if candidate['fit_selected_parameters'] is not None or candidate['verification'] is not None or candidate['enabled']:raise ValueError('This diagnosis is restricted to rejected v2FIT with unopened VERIFY performance')
    counts={g:dict(corrected_views=0,broken_views=0,gain_groups=set()) for g in (.025,.0625,.125,.25)}
    scores=[];classes=[];groups=[];opportunities=set();failed_near=set();single_groups=set();near_views=single_views=failed=0
    for entry in manifest['entries']:
        if entry['fold']!='fit':continue
        path=root/'confirmation'/entry['filename']
        if sha(path)!=entry['sha256']:raise ValueError('Opened FIT raw source changed')
        row=torch.load(path,map_location='cpu',weights_only=False)
        if row['fold']!='fit':raise ValueError('No VERIFY data in diagnosis')
        s=row['public']['spatial'];base=s['base_logits'].float();q=row['risk']['q'];grid=s['grid_world'].float();target=row['labels']['target_xy'].float()
        distance=(grid-target).norm(dim=-1);before=bool(distance[base.argmax()]<=20);group=tuple(row['target_instance']);failed+=int(not before)
        actual=row['risk']['fixed_action_logits'];after=bool(distance[actual.argmax()]<=20)
        outcome=int(after and not before)-int(before and not after)
        probability=row['risk']['scores'];odds=float(probability[1]/(probability[1]+probability[2]).clamp_min(1e-20))
        scores.append([odds,float(row['prediction']['benefit_logit'].sigmoid())]);classes.append(outcome);groups.append(group)
        if not row['risk']['available']:continue
        reliable=row['prediction']['reliable'].bool();support=s['candidate_support'].bool()&reliable[:,None]
        near=bool((support.any(0)&(distance<=20)).any());near_views+=int(near and not before)
        if near and not before:failed_near.add(group)
        for weight,c in counts.items():
            mixed=actual if weight==.25 else bounded_belief_mix(base,q,weight)
            correct=bool(distance[mixed.argmax()]<=20);gain=correct and not before;harm=before and not correct
            c['corrected_views']+=int(gain);c['broken_views']+=int(harm)
            if gain:c['gain_groups'].add(group);opportunities.add(group)
        if not before and near:
            active=support.any(-1);polygons=support[active];centers=s['candidate_xy'][active].double();world=grid.double()
            kernel=(-.5*((centers[:,None]-world[None]).norm(dim=-1)/4.).square()).masked_fill(~polygons,-torch.inf).softmax(-1).float()
            mixed=batch_belief_mix(base[None].expand(len(kernel),-1),kernel,torch.full((len(kernel),),.25))
            possible=bool((distance[mixed.argmax(-1)]<=20).any());single_views+=int(possible)
            if possible:single_groups.add(group)
    y=np.asarray(classes);x=np.asarray(scores);mask=y!=0
    auc=lambda j:float(roc_auc_score(y[mask]>0,x[mask,j])) if len(np.unique(y[mask]))==2 else None
    result=dict(scope='Opened v2FIT126 only, post-rejection diagnosis; no VERIFY tensors/aggregates, no new gate fitting',
        views=len(y),target_groups=len(set(groups)),classes={str(int(k)):int((y==k).sum()) for k in np.unique(y)},
        head_odds_gain_vs_harm_auc=auc(0),old_benefit_same_action_auc=auc(1),a0_failed_views=failed,
        failed_views_with_any_public_support_within20m=near_views,failed_groups_with_any_public_support_within20m=len(failed_near),
        fixed_existing_Q_weights={str(g):dict(corrected_views=c['corrected_views'],broken_views=c['broken_views'],gain_groups=len(c['gain_groups'])) for g,c in counts.items()},
        discrete_existing_Q_recoverable_groups=len(opportunities),tested_single_candidate25_recoverable_views=single_views,tested_single_candidate25_recoverable_groups=len(single_groups),
        oracle_caveat='GT picks tested candidate/weight for diagnosis only; spatialwithin20m not object identity; finite tested opportunities, not a continuous-weight capacity upper bound or deployable performance',
        source_manifest_sha256=sha(root/'confirmation/manifest.json'),failed_candidate_sha256=sha(root/'calibration/candidate_calibration.json'),analysis_code_sha256=sha(Path(__file__)))
    save(result,ROOT/'reports/a2_execution_20261027/DIRECT_RISK_V2_FIT_DIAGNOSIS.json');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
