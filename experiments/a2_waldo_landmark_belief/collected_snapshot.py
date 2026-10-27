"""Explicit user-authorized frozen subset; original full-cache gates stay strict.

Raw views remain byte-identical. Derived target-disjoint labels are masked on
load. No uncompleted episode or benchmark data enters this snapshot.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import torch
from .paths import ROOT,sha,save

BASE=ROOT/'runs/a2_waldo_landmark_20261027'
KIND='user_stopped_collection_target_disjoint_snapshot_v1'


def snapshot_partition(map_name,object_id):
    value=hashlib.sha256(f'a2-collected-seed0\0{map_name}\0{object_id}'.encode()).digest()
    fraction=int.from_bytes(value[:8],'big')/2**64
    return 'train' if fraction<.8 else 'dev' if fraction<.9 else 'calibration'


def mask_snapshot_row(row,partition,signature):
    map_name,oid=row['target_instance']
    if snapshot_partition(map_name,oid)!=partition:
        raise ValueError('Snapshot target partition changed')
    labels=dict(row['labels']);known=labels['rank_label_mask'].clone()
    details=[]
    for i,detail in enumerate(row['private']['association']):
        identity=detail.get('object_id')
        allowed=identity is not None and snapshot_partition(map_name,identity)==partition
        known[i]&=allowed
        details.append(dict(detail,associated_split=snapshot_partition(map_name,identity) if identity is not None else None,
                            supervision_allowed=bool(known[i])))
    if ((labels['ranking']>0) & ~known).any():
        raise ValueError('A positive was assigned outside its target partition')
    labels['rank_label_mask']=known
    labels['offset_valid']=labels['offset_valid'] & known
    labels['match_valid']=labels['offset_valid'].any() | (known.all() & ~(labels['ranking']>0).any())
    labels['match']=labels['offset_valid'].any().float()
    return dict(row,labels=labels,private=dict(row['private'],association=details),signature_sha256=signature)


def cache_source(directory):
    directory=Path(directory)
    path=directory/'snapshot_source.json'
    return json.loads((path if path.exists() else BASE/'source_manifest.json').read_text())


def verify_cache_origin(directory):
    from .prepare import cache_signature
    directory=Path(directory);saved=json.loads((directory/'signature.json').read_text())
    if saved.get('kind')!=KIND:
        if saved!=cache_signature():raise ValueError('Current extraction sources changed')
        return saved
    if saved['loader_sha256']!=sha(Path(__file__)):
        raise ValueError('Snapshot loader changed after freeze')
    if sha(directory/'snapshot_source.json')!=saved['snapshot_source_sha256']:
        raise ValueError('Snapshot partition source changed')
    if sha(BASE/'source_manifest.json')!=saved['parent_source_sha256']:
        raise ValueError('Original audited source changed')
    parent=Path(saved['parent_cache'])
    if sha(parent/'signature.json')!=saved['parent_signature_sha256'] or json.loads((parent/'signature.json').read_text())!=cache_signature():
        raise ValueError('Snapshot raw extraction provenance changed')
    original={e['index']:e for e in json.loads((BASE/'source_manifest.json').read_text())['entries']}
    source=cache_source(directory)
    indices=[e['index'] for e in source['entries']]
    if len(indices)!=len(set(indices)):raise ValueError('Duplicate snapshot source episodes')
    for entry in source['entries']:
        actual=original[entry['index']]
        if actual['split']!='train' or any(entry[k]!=actual[k] for k in ('map_name','object_id','description_id')):
            raise ValueError('Snapshot source identity changed')
        if entry['split']!=snapshot_partition(entry['map_name'],entry['object_id']):
            raise ValueError('Snapshot target split changed')
        raw=parent/f"train_{entry['index']}_complete.json"
        if sha(raw)!=entry['raw_completion_sha256']:
            raise ValueError('Raw episode completion changed')
        info=json.loads(raw.read_text())
        if info['signature_sha256']!=saved['parent_signature_sha256']:
            raise ValueError('Raw completed episode provenance mismatch')
        derived=json.loads((directory/f"{entry['split']}_{entry['index']}_complete.json").read_text())
        if derived['entries']!=info['entries']:
            raise ValueError('Snapshot rows differ from frozen completed episode')
    return saved


def load_cache_row(directory,entry,partition):
    directory=Path(directory)
    saved=json.loads((directory/'signature.json').read_text())
    row=torch.load(directory/entry['filename'],map_location='cpu',weights_only=False)
    if saved.get('kind')==KIND:
        if row['signature_sha256']!=saved['parent_signature_sha256']:
            raise ValueError('Raw snapshot row signature mismatch')
        row=mask_snapshot_row(row,partition,sha(directory/'signature.json'))
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=BASE/'collected_snapshot')
    args=parser.parse_args();parent=BASE/'cache';output=args.output
    if output.exists():raise ValueError('Snapshot already exists: never silently rewrite a frozen population')
    output.mkdir(parents=True)
    source=json.loads((BASE/'source_manifest.json').read_text())
    original={e['index']:e for e in source['entries']}
    entries=[];raw_infos={}
    parent_sha=sha(parent/'signature.json')
    for path in sorted(parent.glob('train_*_complete.json')):
        index=int(path.name.split('_')[1]);entry=original[index]
        if entry['split']!='train':raise ValueError('Unexpected source split')
        info=json.loads(path.read_text())
        if info['signature_sha256']!=parent_sha:raise ValueError('Raw completion signature changed')
        for item in info['entries']:
            if sha(parent/item['filename'])!=item['sha256']:raise ValueError('Raw cached row changed')
        entries.append(dict(entry,split=snapshot_partition(entry['map_name'],entry['object_id']),raw_completion_sha256=sha(path)))
        raw_infos[index]=info
    entries.sort(key=lambda e:e['index'])
    frozen_source=dict(kind=KIND,entries=entries,original_episode_count=len(original),seed=0)
    save(frozen_source,output/'snapshot_source.json')
    signature=dict(kind=KIND,parent_cache=str(parent.resolve()),parent_signature_sha256=parent_sha,
                   parent_source_sha256=sha(BASE/'source_manifest.json'),snapshot_source_sha256=sha(output/'snapshot_source.json'),
                   loader_sha256=sha(Path(__file__)),selection='All fully completed original TRAIN episodes at user stop; seed0 secondary target hash80/10/10',
                   calibration_observations='teacher views only; not on-policy; gate must remain off pending actual closed-loop pilot')
    save(signature,output/'signature.json');digest=sha(output/'signature.json')
    rows=[]
    for entry in entries:
        info=raw_infos[entry['index']]
        for item in info['entries']:
            os.link(parent/item['filename'],output/item['filename'])
            rows.append(dict(item,split=entry['split'],episode=entry['index']))
        save(dict(info,signature_sha256=digest),output/f"{entry['split']}_{entry['index']}_complete.json")
    counts=Counter(e['split'] for e in rows)
    if not all(counts[p]>0 for p in ('train','dev','calibration')):raise ValueError('Empty derived partition')
    manifest=dict(complete=True,scope=KIND,original_full_collection_complete=False,signature_sha256=digest,entries=rows,
                  counters={p:dict(views=counts[p]) for p in ('train','dev','calibration')})
    save(manifest,output/'manifest.json')
    verify_cache_origin(output)
    report=dict(status='frozen_collected_subset_not_full_original_collection',snapshot=str(output),
                episodes=len(entries),views=len(rows),episodes_by_partition=dict(Counter(e['split'] for e in entries)),
                views_by_partition=dict(counts),targets_by_partition={p:len({(e['map_name'],e['object_id']) for e in entries if e['split']==p}) for p in counts},
                original_progress=json.loads((BASE/'preparation_progress.json').read_text()),
                signature_sha256=digest,manifest_sha256=sha(output/'manifest.json'),
                incomplete_episode_rows_excluded=True,raw_labels_remasked_at_load=True,calibration_on_policy=False,
                collection_stopped_by_user=True,gate_enabled=False)
    save(report,BASE/'collection_stopped_snapshot.json')
    save(report,ROOT/'reports/a2_execution_20261027/COLLECTED_SNAPSHOT.json')
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
