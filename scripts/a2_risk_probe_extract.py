"""Serial GPU inference on original TRAIN/dev; no training or calibration reuse."""
import json,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE,verify_cache_origin,cache_source
from experiments.a2_waldo_landmark_belief.training_protocol import validate_complete_cache
from experiments.a2_waldo_landmark_belief.revision_confirmation import immutable_json
from experiments.a2_waldo_landmark_belief.revision_data import RevisionViews
from experiments.a2_waldo_landmark_belief.revision_model import RevisionNetwork
from experiments.a2_waldo_landmark_belief.revision_fusion import batch_proposal
from experiments.a2_waldo_landmark_belief.batching import collate_views,to_device
from experiments.a2_waldo_landmark_belief.risk_probe import public_risk_features,outcome_labels,FEATURE_NAMES,WEIGHTS


def main():
    torch.set_num_threads(2);torch.manual_seed(0)
    root=BASE/'revision_v2_probe';root.mkdir(parents=True,exist_ok=True);checkpoint=BASE/'revision_v1/training/best.pt'
    verify_cache_origin(BASE/'collected_snapshot')
    validate_complete_cache(BASE/'collected_snapshot',cache_source(BASE/'collected_snapshot'),hash_rows=True)
    names=('risk_probe.py','revision_model.py','revision_fusion.py','revision_data.py','batching.py')
    signature=dict(checkpoint_sha256=sha(checkpoint),cache_manifest_sha256=sha(BASE/'collected_snapshot/manifest.json'),
        next_confirmation_reserve_sha256=sha(BASE/'revision_v2_confirmation_reserve.json'),
        code={n:sha(ROOT/'experiments/a2_waldo_landmark_belief'/n) for n in names},extract_code_sha256=sha(Path(__file__)),
        features=list(FEATURE_NAMES),weights=list(WEIGHTS),scope='OriginalTRAIN31415/dev4618 only; frozenv1 inference, not formal training; no confirmation/benchmark observations')
    immutable_json(signature,root/'signature.json');digest=sha(root/'signature.json')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    for name,dependency_digest in state['code_sha256'].items():
        if sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)!=dependency_digest:raise ValueError('Frozen v1 inference dependency changed')
    model=RevisionNetwork(**state['model_config']).cuda();model.load_state_dict(state['model']);model.eval()
    started=time.perf_counter();reports={}
    with torch.inference_mode():
        for partition,expected in (('train',31415),('dev',4618)):
            dataset=RevisionViews(BASE/'collected_snapshot',partition)
            if len(dataset)!=expected:raise ValueError('Original target partition count changed')
            loader=DataLoader(dataset,batch_size=16,num_workers=2,shuffle=False,collate_fn=collate_views)
            features=[];outcomes=[];before=[];after=[];count=0
            for raw in loader:
                batch=to_device(raw,'cuda');output=model(**batch['network']);spatial=batch['spatial']
                q,available=batch_proposal(spatial,output['scores'],output['candidate_match_logits'],output['reliable'])
                x=public_risk_features(spatial['base_logits'],q,spatial['grid_world'],output['benefit_logit'].sigmoid(),available)
                # Target coordinates first enter the separate offline label call.
                y=outcome_labels(spatial['base_logits'],q,spatial['grid_world'],batch['labels']['target_xy'])
                features.append(x.cpu().numpy());outcomes.append(y['outcomes'].cpu().numpy());before.append(y['before_ne'].cpu().numpy());after.append(y['after_ne'].cpu().numpy());count+=len(x)
                if count%1024==0:print(json.dumps(dict(partition=partition,views=count,seconds=time.perf_counter()-started)),flush=True)
            if count!=expected:raise ValueError('Partial extraction cannot count as complete')
            identities=[dict(map_name=dataset.source[e['episode']]['map_name'],object_id=dataset.source[e['episode']]['object_id'],episode=e['episode'],step=e['step']) for e in dataset.entries]
            path=root/f'{partition}.npz';temporary=root/f'{partition}.tmp.npz'
            np.savez_compressed(temporary,features=np.concatenate(features),outcomes=np.concatenate(outcomes),before_ne=np.concatenate(before),after_ne=np.concatenate(after));temporary.replace(path)
            save(identities,root/f'{partition}_identities.json')
            reports[partition]=dict(views=count,npz_sha256=sha(path),identities_sha256=sha(root/f'{partition}_identities.json'))
    save(dict(signature_sha256=digest,partitions=reports,status='complete_original_TRAIN_dev_risk_probe_extraction',seconds=time.perf_counter()-started),root/'complete.json')
    print(json.dumps(reports),flush=True)


if __name__=='__main__':main()
