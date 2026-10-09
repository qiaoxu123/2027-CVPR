import json
from types import SimpleNamespace
import torch
import pytest
from graphnav.candidate_geometry import FullCandidateDataset
from graphnav.checkpointing import file_sha256
from graphnav.dynamic_data import DynamicEvidenceDataset


def files(tmp_path):
    args=SimpleNamespace(grid_size=4,max_landmarks=2,max_roads=0,max_relations=2,
        prior_sigma_m=10.,rule_sigma_m=10.,text_backbone='stub',max_text_length=96)
    queries=tmp_path/'queries.jsonl';queries.write_text(json.dumps(dict(map_name='map',instruction='near hall',
        target_xy=[0.,0.],world_bounds=[0,0,400,400],physical_map_sizes=[400,400],
        landmarks=[dict(name='hall',object_type='Building',contour=[[.2,.2],[.3,.2],[.3,.3],[.2,.3]])]))+'\n')
    evidence=tmp_path/'evidence';vectors=tmp_path/'vectors';fields=tmp_path/'fields'
    for path in (evidence,vectors,fields):path.mkdir()
    torch.save(torch.ones(2,6),vectors/'vectors.pt')
    roi=dict(map_name='map',object_id='1',source_episode_index=1,visible_pixels=400)
    (vectors/'rows.jsonl').write_text(''.join(json.dumps(dict(**roi,observed_step=step))+'\n' for step in (0,8)))
    (vectors/'manifest.json').write_text(json.dumps(dict(vectors_sha256=file_sha256(vectors/'vectors.pt'))))
    torch.save(dict(geometry={'map:1':torch.ones(16,10)},confidence=torch.tensor([.8,.8])),evidence/'geometry_quality.pt')
    sample=dict(map_name='map',source_episode_index=1,query_index=0,step=0,current_roi_indices=[0],history_roi_indices=[0,1],labels=dict(target_world=[100,100,30]))
    (evidence/'samples.jsonl').write_text(json.dumps(sample)+'\n')
    manifest=dict(sha256=dict(queries=file_sha256(queries)),samples_sha256=file_sha256(evidence/'samples.jsonl'),
        vector_manifest_sha256=file_sha256(vectors/'manifest.json'),geometry_sha256=file_sha256(evidence/'geometry_quality.pt'))
    (evidence/'manifest.json').write_text(json.dumps(manifest))
    torch.save(dict(belief_logits=torch.randn(1,16),dense_context=torch.randn(1,16,16)),fields/'chunk.pt')
    (fields/'manifest.json').write_text(json.dumps(dict(complete=True,protocol=dict(query_sha256=file_sha256(queries),dense_logits_dtype='float32'),
        chunks=[dict(file='chunk.pt',first_query=0,queries=1,sha256=file_sha256(fields/'chunk.pt'))])))
    names=FullCandidateDataset(queries,grid_size=4,max_landmarks=2,max_roads=0,max_relations=2)[0]['node_names']
    torch.save(dict(name='stub',max_length=96,cache={name:(torch.ones(2,12),torch.ones(2,dtype=torch.bool)) for name in names}),fields/'bert_tokens.pt')
    return evidence,queries,fields,vectors,args


def test_dynamic_forward_inputs_ignore_goal_and_reject_future_history(tmp_path):
    paths=files(tmp_path);dataset=DynamicEvidenceDataset(*paths,max_observed=2)
    before=dataset[0]
    dataset.rows[0]['labels']['target_world']=[300,300,30];after=dataset[0]
    assert all(torch.equal(before[0][key],after[0][key]) for key in before[0])
    assert not torch.equal(before[1]['target_xy'],after[1]['target_xy'])
    dataset.history=True
    with pytest.raises(ValueError,match='future'):dataset[0]


def test_rounded_legacy_goal_cache_is_rejected(tmp_path):
    paths=files(tmp_path);manifest=paths[2]/'manifest.json';state=json.loads(manifest.read_text())
    state['protocol'].pop('dense_logits_dtype');manifest.write_text(json.dumps(state))
    with pytest.raises(ValueError,match='FP32'):DynamicEvidenceDataset(*paths)


def test_real_cached_visual_residual_optimizer_step_keeps_a3_frozen(tmp_path):
    from graphnav.full_localization import FullTokenGeometryLocalizer
    from graphnav.dynamic_belief import DynamicGoalBelief
    from scripts.train_dynamic_belief import run_epoch
    from torch.utils.data import DataLoader
    dataset=DynamicEvidenceDataset(*files(tmp_path),max_observed=2)
    a3=FullTokenGeometryLocalizer(hidden=16,text_dim=12,heads=4,max_relations=2,top_k=3,dropout=0)
    model=DynamicGoalBelief(a3,vision_dim=6)
    optimizer=torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad],lr=1e-3)
    result=run_epoch(model,DataLoader(dataset,batch_size=1),SimpleNamespace(variant='visual'),torch.device('cpu'),optimizer)
    assert result['state_queries']==1 and result['unique_public_queries']==1
    assert model.visual_gate.grad is not None and torch.isfinite(model.visual_gate.grad)
    assert all(parameter.grad is None for parameter in a3.parameters())
