"""Verify the identifiable name-set Oracle uses unchanged cached A3 inputs."""
import gzip
import json
from pathlib import Path
import numpy as np
import torch

from graphnav.landmark_alias import build_name_index, resolve_references
from .audit import DATA, OUT, sha, save


def run():
    public=json.loads((DATA/'cityrefer/landmarks.json').read_text())
    indexes={m:build_name_index(a) for m,a in public.items()}
    result={};sets={};independence={}
    for split in ['train','dev','seen']:
        planroot=Path('runs/instance_grounding')/('plan_'+split)
        manifest=json.loads((planroot/'manifest.json').read_text())
        cache=Path(manifest['source_cache'])
        cmanifest=json.loads((cache/'manifest.json').read_text())
        qpath=cache/'queries.jsonl'
        assert sha(qpath)==cmanifest['files']['queries.jsonl']
        lookup={}
        for line in qpath.read_text().splitlines():
            q=json.loads(line);lookup[(q['map_name'],q['instruction'])]=q
        rows=json.loads(gzip.open(OUT/(split+'_family_predictions.json.gz'),'rt').read())
        checked=0
        for r in rows:
            q=lookup[(r['map_name'],r['instruction'])]
            refs=resolve_references(r['audit']['names'],public[r['map_name']],index=indexes[r['map_name']],max_landmarks=16,match_mode='fuzzy')
            assert len(refs)==len(q['landmarks'])
            bounds=np.array(q['world_bounds']);span=bounds[2:]-bounds[:2]
            for x,y in zip(refs,q['landmarks']):
                assert x['name']==y['name'] and x['object_type']==y['object_type']
                assert x['aliases']==y['aliases']
                assert np.allclose((np.asarray(x['contour'])-bounds[:2])/span,y['contour'],atol=1e-7,rtol=0)
            assert r['geometry_ranking']==r['known_name_oracle_ranking']
            assert all(r['metrics'][k]==r['known_name_oracle_metrics'][k] for k in r['metrics'])
            checked+=1
        plan=torch.load(planroot/'plan.pt',weights_only=True)
        sets[split]={'family':{r['metadata']['family_id'] for r in plan['states']},
                     'map':{r['metadata']['map_name'] for r in plan['states']},
                     'object':{(r['metadata']['map_name'],plan['labels'][r['metadata']['sample_id']]['target_object_id']) for r in plan['states']}}
        independence[split]=dict(target_objects=len(sets[split]['object']),family_instructions=len(sets[split]['family']),maps=len(sets[split]['map']))
        result[split]=dict(verified_identical_public_reference_queries=checked,query_cache_sha256=sha(qpath),
                           source_plan_sha256=sha(planroot/'plan.pt'),reference_gt_nearest_selection=False)
    for key in ['family','map','object']:
        assert not(sets['train'][key]&sets['dev'][key])
    result['train_dev_overlap']={key:0 for key in ['family','map','object']}
    result['name_set_oracle_is_identity_intervention_not_instance_oracle']=True
    # Original evaluation is a reference for exact protocol parity, not tuning.
    original=Path('runs/instance_grounding/evaluation_seen/geometry_only_predictions.jsonl')
    old={x['sample_id']:x for x in map(json.loads,original.read_text().splitlines())}
    current=json.loads(gzip.open(OUT/'seen_all_prefix_predictions.json.gz','rt').read())
    for x in current:
        for key in ['goal_r1','goal_r4','goal_r8','ne']:
            assert abs(x['metrics'][key]-old[x['sample_id']]['metrics'][key])<1e-7,(x['sample_id'],key)
        assert x['geometry_ranking'][0]==old[x['sample_id']]['static_top1']
    result['original_seen_baseline_parity']=dict(states=len(current),all_goal_metrics_and_top1_identical=True,source_sha256=sha(original))
    checkpoint=Path('/home/rental/20260922_1/Workspace/2027-CVPR/GraphNav/runs/static_final_analysis_20261010/E1_seed1/best.pt')
    digest=sha(checkpoint)
    assert digest=='9d9be861bf7e9cac986d747867990de3d409ccb9bc1545862d71a620448380d7'
    independence['geometry_checkpoint']=dict(path=str(checkpoint),sha256=digest,changed=False)
    save(OUT/'independence_and_model.json',independence)
    save(OUT/'oracle_input_equivalence.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':run()
