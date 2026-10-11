"""Read-only provenance/population/cell-collision audit after fixed inference."""
import argparse,gzip,hashlib,json,subprocess,time
from pathlib import Path
import numpy as np
import torch
from experiments.bidirectional_core.cache import DEFAULT_DATA
from experiments.instance_grounding.association import weights
from graphnav.checkpointing import file_sha256
from .run import write


def rows(p):
    with gzip.open(p,'rt') as f:return [json.loads(line) for line in f]


def run(a):
    root=Path(a.runs);start=time.time();p2=Path(a.p2);cfg=json.loads((p2/'calibration_dev/frozen_config.json').read_text());protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());lock=json.loads((root/'dev_complete/seen_lock.json').read_text());checks={}
    for name,sha in cfg['checkpoint_sha256'].items():assert file_sha256(cfg['heads'][name])==sha
    for name,sha in cfg['source_sha256'].items():
        if name in ['data.py','dataset.py','model.py','evaluate.py','train.py','relations.py','association.py']:assert file_sha256(Path('experiments/instance_grounding')/name)==sha
    assert file_sha256(Path(__file__).with_name('protocol.json'))==lock['protocol_sha256'];assert file_sha256(root/'text_cache/text_features.pt')==lock['text_sha256']
    for name,sha in lock['source_sha256'].items():assert file_sha256(Path(__file__).with_name(name))==sha
    original=subprocess.check_output(['git','show',protocol['base_commit']+':docs/assets/instance_grounding/evidence/calibration_dev/frozen_config.json']);assert original== (p2/'calibration_dev/frozen_config.json').read_bytes()
    for split in ['dev_complete','seen']:
        folder=root/split;base=rows(folder/'city_only__full__rgb__real_identity.jsonl.gz');ids={v['sample_id']:v for v in base};assert len(ids)==640
        count=0
        for p in folder.glob('*_identity.jsonl.gz'):
            rs=rows(p);assert len(rs)==640;count+=1
            for v in rs:
                b=ids[v['sample_id']]
                for k in ['family_id','map_name','cutoff_step','candidate_object_ids','candidate_source_steps']:assert v[k]==b[k]
                assert all(s<=v['cutoff_step'] for s in v['candidate_source_steps']);assert np.isfinite(v['scores']).all()
        residual_count=0
        for p in folder.glob('*_residual.jsonl.gz'):
            rs=rows(p);assert len(rs)==640;residual_count+=1
            for v in rs:assert v['offline_only'] and v['candidate_object_ids']==ids[v['sample_id']]['candidate_object_ids']
        checks[split]=dict(identity_series=count,residual_series=residual_count,states_per_series=640,causal=True,same_candidates=True,finite_scores=True)
    data=torch.load(p2/'features_seen/features.pt',weights_only=True,mmap=True)['plan'];objects=json.loads((Path(DEFAULT_DATA)/'cityrefer/objects.json').read_text());collision=[]
    for s in data['states']:
        meta=s['metadata'];regions=s['regions']
        for v in regions:v['contour']=objects[meta['map_name']][v['object_id']]['contour']
        w=weights(s,regions);target=data['labels'][meta['sample_id']]['target_object_id'];positive=np.array([v['object_id']==target for v in regions]);cells=(w>0).sum(0);support=(w[positive]>0).any(0) if positive.any() else np.zeros(784,bool)
        collision.append(dict(sample_id=meta['sample_id'],family_id=meta['family_id'],map_name=meta['map_name'],target_visible=bool(positive.any()),target_supported=bool(support.any()),target_cell_competitors=int(cells[support].max()) if support.any() else None,unsupported_observed_instances=int((~(w>0).any(1)).sum()),multiply_associated_cells=int((cells>=2).sum())))
    write(root/'cell_collisions.json',collision)
    visible=[v for v in collision if v['target_visible']];supported=[v for v in visible if v['target_supported']];colliding=[v for v in supported if v['target_cell_competitors']>=2]
    result=dict(populations=checks,checkpoints_unchanged=True,p2_configuration_bytes_unchanged=True,experimental_code_lock_verified=True,text_cache_lock_verified=True,formal_gain=.025,formal_threshold=1.01,training_steps=0,image_encodings=0,navigation_rollouts=0,unseen_runs=0,cell_collision_summary=dict(target_visible_states=len(visible),target_supported_states=len(supported),target_supported_with_competitors_states=len(colliding),independent_colliding_families=len({v['family_id'] for v in colliding}),maximum_target_cell_candidates=max(v['target_cell_competitors'] for v in supported),unsupported_observed_instance_entries=sum(v['unsupported_observed_instances'] for v in collision)),seconds=time.time()-start)
    write(root/'integrity.json',result);print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--p2',required=True);run(p.parse_args())
if __name__=='__main__':main()
