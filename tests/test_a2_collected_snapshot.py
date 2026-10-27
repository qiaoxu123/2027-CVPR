import torch
from experiments.a2_waldo_landmark_belief.collected_snapshot import snapshot_partition, mask_snapshot_row


def test_secondary_split_is_target_stable_and_has_three_populations():
    groups=[dict(map_name='map',object_id=i) for i in range(500)]
    assignments=[snapshot_partition(r['map_name'],r['object_id']) for r in groups]
    assert set(assignments)=={'train','dev','calibration'}
    assert assignments==[snapshot_partition('map',i) for i in range(500)]


def test_cross_partition_negatives_are_unknown_not_training_labels():
    target=1;split=snapshot_partition('map',target)
    same=next(i for i in range(2,100) if snapshot_partition('map',i)==split)
    other=next(i for i in range(2,100) if snapshot_partition('map',i)!=split)
    row=dict(target_instance=['map',target],signature_sha256='raw',labels=dict(
        ranking=torch.tensor([1.,0.,0.,0.]),rank_label_mask=torch.tensor([True,True,True,False]),
        offset_valid=torch.tensor([True,False,False,False]),match_valid=torch.tensor(True),match=torch.tensor(1.)),
        private=dict(association=[dict(object_id=i) for i in (target,same,other,None)]))
    result=mask_snapshot_row(row,split,'derived')
    assert result['labels']['rank_label_mask'].tolist()==[True,True,False,False]
    assert result['signature_sha256']=='derived'
    assert result['private']['association'][2]['supervision_allowed'] is False
    assert torch.equal(result['labels']['ranking'],torch.tensor([1.,0,0,0]))


def test_snapshot_loader_checks_raw_provenance_and_masks_only_derived_labels(tmp_path):
    import json
    from experiments.a2_waldo_landmark_belief.collected_snapshot import KIND,load_cache_row
    from experiments.a2_waldo_landmark_belief.paths import sha
    target=1;split=snapshot_partition('map',target)
    other=next(i for i in range(2,100) if snapshot_partition('map',i)!=split)
    row=dict(target_instance=['map',target],signature_sha256='raw',labels=dict(
        ranking=torch.tensor([1.,0.]),rank_label_mask=torch.tensor([True,True]),
        offset_valid=torch.tensor([True,False]),match_valid=torch.tensor(True),match=torch.tensor(1.)),
        private=dict(association=[dict(object_id=i) for i in (target,other)]))
    (tmp_path/'signature.json').write_text(json.dumps(dict(kind=KIND,parent_signature_sha256='raw')))
    torch.save(row,tmp_path/'row.pt')
    result=load_cache_row(tmp_path,dict(filename='row.pt'),split)
    assert result['labels']['rank_label_mask'].tolist()==[True,False]
    assert result['signature_sha256']==sha(tmp_path/'signature.json')
    assert torch.load(tmp_path/'row.pt',weights_only=False)['labels']['rank_label_mask'].all()
    import pytest
    row['signature_sha256']='changed';torch.save(row,tmp_path/'row.pt')
    with pytest.raises(ValueError,match='signature'):load_cache_row(tmp_path,dict(filename='row.pt'),split)


def test_freeze_uses_completed_episodes_only_and_keeps_original_cache_incomplete(tmp_path,monkeypatch):
    import json,sys
    from experiments.a2_waldo_landmark_belief import collected_snapshot as snapshot
    from experiments.a2_waldo_landmark_belief.paths import sha
    base=tmp_path/'run';parent=base/'cache';parent.mkdir(parents=True)
    (parent/'signature.json').write_text('{}');signature=sha(parent/'signature.json')
    entries=[]
    for i in range(40):
        entries.append(dict(index=i,map_name='map',object_id=i,description_id=i,split='train'))
        path=parent/f'train_{i}_0.pt';path.write_bytes(b'raw-feature')
        if i==39:continue  # A fully written row without episode commit is incomplete.
        item=dict(filename=path.name,sha256=sha(path),step=0,counts=dict(views=1))
        (parent/f'train_{i}_complete.json').write_text(json.dumps(dict(signature_sha256=signature,entries=[item])))
    (base/'source_manifest.json').write_text(json.dumps(dict(entries=entries)))
    (base/'preparation_progress.json').write_text(json.dumps(dict(status='extracting')))
    monkeypatch.setattr(snapshot,'BASE',base);monkeypatch.setattr(snapshot,'ROOT',tmp_path)
    monkeypatch.setattr(snapshot,'verify_cache_origin',lambda path:None)
    output=base/'snapshot';monkeypatch.setattr(sys,'argv',['freeze','--output',str(output)])
    snapshot.main()
    manifest=json.loads((output/'manifest.json').read_text())
    assert len(manifest['entries'])==39
    assert not (output/'train_39_0.pt').exists()
    assert not (parent/'manifest.json').exists()
    assert manifest['original_full_collection_complete'] is False
    assert set(e['split'] for e in manifest['entries'])=={'train','dev','calibration'}
    groups={p:{e['episode'] for e in manifest['entries'] if e['split']==p} for p in ('train','dev','calibration')}
    assert not groups['train']&groups['dev'] and not groups['train']&groups['calibration']
    import pytest
    with pytest.raises(ValueError,match='already exists'):snapshot.main()
