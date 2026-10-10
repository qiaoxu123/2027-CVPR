import json
import torch
import pytest
from graphnav.checkpointing import file_sha256
from graphnav.trajectory_data import FrozenTrajectoryDataset,query_is_dev
from graphnav.trajectory_policy import CandidateTrajectoryStop
from scripts.train_trajectory_stop import run_epoch
from torch.utils.data import DataLoader


def fixture_files(tmp_path):
    queries=tmp_path/'queries.jsonl'
    queries.write_text(json.dumps(dict(map_name='map',instruction='near library',target_xy=[0.,0.]))+'\n')
    samples=tmp_path/'samples.jsonl'
    row=dict(query_index=0,pose_world=[0,0,30,0],ground_level=0,
        labels=dict(target_world=[50,50,30],teacher_world=[[50,50,30]]*8,valid_teacher_z=[True]*8))
    samples.write_text(json.dumps(row)+'\n')
    cache=tmp_path/'cache';cache.mkdir()
    payload=dict(candidate_xy=torch.tensor([[[.5,.5],[.8,.8]]]),candidate_context=torch.ones(1,2,16),
        candidate_logits=torch.tensor([[1.,0.]]),belief_entropy=torch.tensor([.5]),bounds=torch.tensor([[0.,0.,100.,100.]]))
    torch.save(payload,cache/'chunk.pt')
    (cache/'manifest.json').write_text(json.dumps(dict(complete=True,protocol=dict(query_sha256=file_sha256(queries)),
        chunks=[dict(file='chunk.pt',first_query=0,queries=1,sha256=file_sha256(cache/'chunk.pt'))])))
    return samples,queries,cache,row


def test_family_split_ignores_annotation_variant_and_target():
    base=dict(map_name='map',instruction='near library')
    assert query_is_dev(dict(**base,target_xy=[1,2],landmarks=['a']))==query_is_dev(dict(**base,target_xy=[3,4],landmarks=['b']))


def test_label_changes_cannot_change_forward_inputs(tmp_path):
    samples,queries,cache,row=fixture_files(tmp_path)
    first=FrozenTrajectoryDataset(samples,queries,cache,split='all')[0]
    row['labels']['target_world']=[999,888,777];row['labels']['teacher_world']=[[999,888,777]]*8
    samples.write_text(json.dumps(row)+'\n')
    second=FrozenTrajectoryDataset(samples,queries,cache,split='all')[0]
    assert first[0].keys()==second[0].keys()
    assert all(torch.equal(first[0][key],second[0][key]) for key in first[0])
    assert 'target_world' not in first[0] and 'teacher_world' not in first[0]
    assert not torch.equal(first[1]['target_world'],second[1]['target_world'])


def test_cache_integrity_and_real_training_step(tmp_path):
    samples,queries,cache,_=fixture_files(tmp_path)
    dataset=FrozenTrajectoryDataset(samples,queries,cache,split='all')
    model=CandidateTrajectoryStop(hidden=16);optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    before=model.path.weight.detach().clone()
    result=run_epoch(model,DataLoader(dataset,batch_size=1),torch.device('cpu'),optimizer)
    assert result['samples']==1 and result['total']>0
    assert not torch.equal(before,model.path.weight)
    queries.write_text(queries.read_text()+'\n')
    with pytest.raises(ValueError,match='SHA256'):FrozenTrajectoryDataset(samples,queries,cache,split='all')
