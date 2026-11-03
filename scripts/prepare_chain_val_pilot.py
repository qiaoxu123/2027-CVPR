"""Freeze one Val-Seen128 paired diagnostic from task indices, no outcome selection."""
import argparse,hashlib,json,sys
from pathlib import Path
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','sbf-source','checkpoint','local-checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--seed',type=int,default=0);a=p.parse_args()
    if a.output.exists():raise FileExistsError('do not redraw fixed pilot')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    ds=CityNavDataset(a.data_root,'val_seen',version='processed_citynav')
    if len(ds)<128:raise ValueError('at least128 Val-Seen episodes required')
    ids=sorted(range(len(ds)),key=lambda i:hashlib.sha256(f'fixed-val-seen-128:{a.seed}:{i}'.encode()).hexdigest())[:128]
    ids.sort(key=lambda i:(str(ds._raw_records[i]['area']),str(ds._raw_records[i]['block']),i))
    public=[]
    for index in ids:
        raw=ds._raw_records[index]
        public.append(dict(task_key=f'val_seen_{index:05d}',map_name=f"{raw['area']}_block_{raw['block']}",instruction=str(raw['descriptions'][0])))
    a.output.mkdir(parents=True);query=a.output/'public_queries.jsonl';query.write_text(''.join(json.dumps(x)+'\n' for x in public))
    manifest=dict(split='val_seen',dataset_version='processed_citynav',purpose='fixed_val_seen_128',no_parameter_selection=True,indices=ids,seed=a.seed,total_split_episodes=len(ds),
        checkpoint_sha256=file_sha256(a.checkpoint),local_checkpoint_sha256=file_sha256(a.local_checkpoint),annotation_sha256=file_sha256(ds.annotation_path),public_queries_sha256=file_sha256(query),
        selection='uniform deterministic hash of index, no target coordinates/visibility/outcomes/category filtering; map-order only for execution',
        model_selection='Previously frozen TRAIN-selected height-filter model, unchanged after independent TRAIN confirmation; no other candidate searched on Val',
        full_evaluation_gate=dict(minimum_ours_successes=84,requires_more_successes_than_original=True),
        warning='128 subset diagnostic, NOT full Val-Seen performance; no Test access.')
    (a.output/'freeze_pending_roles.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(total_val_seen=len(ds),selected=128,maps=len({x['map_name'] for x in public}),local_checkpoint_sha256=manifest['local_checkpoint_sha256'])))


if __name__=='__main__':main()
