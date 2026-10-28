"""All-row revision-label audit and fixed .25 TRAIN-only oracle diagnostic."""
from collections import Counter
import json
from pathlib import Path
import torch
from .paths import ROOT,save,sha
from .collected_snapshot import BASE,verify_cache_origin,cache_source
from .training_protocol import validate_complete_cache
from .revision_data import RevisionViews
from .revision_fusion import proposal_distribution,bounded_belief_mix,MAX_GATE
from .observed_geometry import gaussian_belief_mix


def main():
    torch.set_num_threads(2);cache=BASE/'collected_snapshot'
    verify_cache_origin(cache);validate_complete_cache(cache,cache_source(cache),hash_rows=True)
    label_code={n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in ('revision_data.py','revision_fusion.py','data_protocol.py')}
    counters={};oracle=Counter()
    for partition in ('train','dev','calibration'):
        ds=RevisionViews(cache,partition);c=Counter()
        for i in range(len(ds)):
            row=ds[i];y=row['labels'];x=row['public']['network'];s=row['public']['spatial']
            valid=x['box_valid'] & (x['pair_valid'] & x['link_valid'][None]).any(-1)
            positive=(y['ranking']>0) & valid;negative=y['rank_label_mask'] & (y['ranking']<=0) & valid
            c.update(dict(views=1,positive_views=int(positive.any()),contrast_views=int(positive.any() and negative.any()),
                          known_positive_labels=int((y['rank_label_mask'] & (y['ranking']>0) & valid).sum()),
                          known_negative_labels=int(negative.sum()),cross_category_negative_labels=int((negative & (x['box_kind']!=y['target_kind'])).sum()),
                          zero_offset_trust_positive_labels=int((y['candidate_trust'] & valid).sum())))
            assert torch.isfinite(x['a0_context']).all()
            if partition=='train':
                base=s['base_logits'];target=y['target_xy'];grid=s['grid_world']
                before=bool((grid[base.argmax()]-target).norm()<=20)
                oracle['train_views']+=1;oracle['a0_failed_views']+=int(not before)
                # Oracle uses only confirmed target boxes, never deployed inputs.
                q,available=proposal_distribution(s,torch.zeros_like(y['ranking']),torch.zeros_like(y['ranking']),positive)
                if available:
                    new=bounded_belief_mix(base,q,MAX_GATE)
                    old=gaussian_belief_mix(base,grid,s['candidate_xy'],torch.zeros_like(y['ranking']),s['candidate_support'],positive,MAX_GATE)
                    new_correct=bool((grid[new.argmax()]-target).norm()<=20)
                    old_correct=bool((grid[old.argmax()]-target).norm()<=20)
                    oracle['views_with_confirmed_target_proposal']+=1
                    oracle['failed_views_with_confirmed_proposal']+=int(not before)
                    oracle['new_oracle_corrected']+=int(new_correct and not before)
                    oracle['new_oracle_broken']+=int(not new_correct and before)
                    oracle['old_positive_support_oracle_corrected']+=int(old_correct and not before)
            if (i+1)%4000==0:print(f'Audited revision {partition} {i+1}/{len(ds)}',flush=True)
        counters[partition]=dict(c)
    if any(sha(ROOT/'experiments/a2_waldo_landmark_belief'/n)!=d for n,d in label_code.items()):raise ValueError('Label code changed during audit')
    report=dict(label_code_sha256=label_code,status='all_revision_rows_audited',cache_signature_sha256=sha(cache/'signature.json'),
                reserve_sha256=sha(BASE/'revision_confirmation_reserve.json'),counts=counters,train_oracle=dict(oracle),
                oracle_max_gate=MAX_GATE,oracle_scope='TRAIN-only confirmed target proposal; fixed .25 gate; diagnostic upper opportunity, NOT trained gain',
                unknown_identity_negatives=False,benchmark_data_used=False)
    save(report,BASE/'revision_data_audit.json');save(report,ROOT/'reports/a2_execution_20261027/REVISION_DATA_AUDIT.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
