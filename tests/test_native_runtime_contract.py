import json
from pathlib import Path
import torch
import pytest
from scripts.evaluate_native_shared_closedloop import source_provenance


def test_archived_source_provenance_does_not_invent_a_git_commit(tmp_path):
    root=tmp_path/'snapshot';(root/'graphnav').mkdir(parents=True);(root/'graphnav/example.py').write_text('x=1\n')
    commit,diff,implementation,mode=source_provenance(root)
    assert commit is None and mode=='archive-source-hashes'
    assert len(implementation['graphnav/example.py'])==64 and json.loads(diff)==implementation


def test_native_frame_selection_is_identical_inside_outer_bf16_autocast():
    if not torch.cuda.is_available():pytest.skip('CUDA integration test')
    root=Path('aerialvg_objects_train256_20261101');prepared=Path('aerialvg_native_training_train256_20261101')
    if not (root/'manifest.json').exists():pytest.skip('Actual native observation cache required')
    from graphnav.native_object_runtime import prepare_native_frame
    manifest=json.loads((root/'manifest.json').read_text());frame=manifest['frames'][0]
    d=torch.load(root/(frame['key']+'.pt'),map_location='cpu',weights_only=True)
    row=next(r for r in torch.load(prepared/'rows_0000.pt',map_location='cpu',weights_only=True) if r['episode']==d['episode_index'] and r['step']==d['step'])
    features=d['all_query_features'].float();features[d['query_ids']]=d['query_features'].float()
    encoded=dict(features=features.cuda(),text=d['encoded_text'][0].float().cuda(),text_mask=d['text_token_mask'][0].cuda(),boxes=d['all_boxes_cxcywh'].cuda(),world_centers=d['all_world_centers'].cuda(),quality=d['all_valid_pixel_fraction'].cuda())
    observed=dict(depth_valid=d['depth_valid'].numpy(),footprint_world=d['footprint_world'].numpy())
    first=prepare_native_frame(encoded,row['grid_world'].cuda(),row['base_logits'].cuda(),observed)
    with torch.autocast('cuda',dtype=torch.bfloat16):second=prepare_native_frame(encoded,row['grid_world'].cuda(),row['base_logits'].cuda(),observed)
    for name in ('features','geometry','valid','kernel','observed','proposal_ids'):
        assert torch.equal(first[name],second[name]),name
    assert first['geometry'].dtype==torch.float32
