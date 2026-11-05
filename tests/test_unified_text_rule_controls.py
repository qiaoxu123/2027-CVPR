import torch


def test_manual_rule_ablation_preserves_public_geometry_and_original_inputs():
    from graphnav.unified_text_ablation import manual_relation_inputs
    static=dict(rules=torch.ones(1,4),relation_fields=torch.ones(1,4,2),node_types=torch.tensor([[0,2,3]]),node_valid=torch.tensor([[True,True,True]]),entity_valid=torch.tensor([[True,False,True]]),nodes=torch.ones(1,3,10),adjacency=torch.ones(1,3,3,dtype=torch.bool),edges=torch.ones(1,3,3,19),pair_geometry=torch.randn(1,4,3,10),prior=torch.randn(1,4))
    inp=dict(static_inputs=static,tokens=torch.randn(1,3,768),token_mask=torch.ones(1,3,dtype=torch.bool),node_emb=torch.ones(1,3,768))
    x=manual_relation_inputs(inp,'manual_off');s=x['static_inputs']
    assert s['rules'].count_nonzero()==0 and s['relation_fields'].count_nonzero()==0
    assert not s['node_valid'][0,1] and s['node_valid'][0,0] and s['node_valid'][0,2]
    assert x['node_emb'][0,1].count_nonzero()==0
    torch.testing.assert_close(s['prior'],static['prior']);torch.testing.assert_close(s['pair_geometry'],static['pair_geometry'])
    assert static['rules'].sum()==4 and static['node_valid'].all() and inp['node_emb'].sum()==2304
    y=manual_relation_inputs(inp,'additive_off');assert y['static_inputs'] is static


def test_runtime_public_record_matches_prepared_input_and_rejects_target(tmp_path):
    import json,pytest
    from scripts.prepare_unified_v2_landmark_relation_ablation import prepare
    from scripts.analyze_instruction_landmark_association import PublicNames
    from graphnav.unified_text_ablation import public_text_record
    catalog={'cambridge_block_1':{'1':dict(name='The One Stop',object_type='Building',contour=[[10,10],[20,10],[20,20],[10,20]])}}
    row=dict(map_name='cambridge_block_1',instruction='The blue car north of One Stop.',landmarks=[],target_xy=[.7,.2],dataset_version='revised',match_mode='fuzzy',world_bounds=[0,0,100,100],physical_map_sizes=[100,100])
    src=tmp_path/'in.jsonl';src.write_text(json.dumps(row)+'\n');cat=tmp_path/'catalog.json';cat.write_text(json.dumps(catalog));prepare(src,cat,tmp_path/'out',expected=1)
    for mode in ('full_text','landmark_relation'):
        saved=json.loads((tmp_path/'out'/f'{mode}.jsonl').read_text());runtime,_=public_text_record(row['map_name'],row['instruction'],row['world_bounds'],catalog,PublicNames(catalog),mode=mode)
        for k in ('instruction','landmarks','world_bounds','physical_map_sizes'):assert saved[k]==runtime[k]
        assert runtime['target_xy']==[0.,0.]
    with pytest.raises(TypeError):public_text_record(row['map_name'],row['instruction'],row['world_bounds'],catalog,PublicNames(catalog),mode='full_text',target_xy=[.7,.2])
