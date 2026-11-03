"""Freeze new TRAIN targets on held-out maps, using no GT position or outcomes."""
import argparse,hashlib,json,sys
from pathlib import Path
from collections import Counter
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('data-root','sbf-source','feature-manifest','local-checkpoint','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--exclude-tasks',type=Path,nargs='+',required=True)
    p.add_argument('--all-remaining',action='store_true',help='Freeze all remaining physical targets, no category quotas')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('confirmation is one-shot')
    sys.path.insert(0,str(a.sbf_source))
    from sbfnav.dataset import CityNavDataset
    ds=CityNavDataset(a.data_root,'train_seen',version='processed_citynav');objects=json.loads((a.data_root/'cityrefer/objects.json').read_text())
    maps={k for k,v in json.loads(a.feature_manifest.read_text())['map_assignment'].items() if v=='confirm'}
    excluded={tuple(t['physical_group']) for path in a.exclude_tasks for t in json.loads(path.read_text())}
    groups={}
    for i,row in enumerate(ds._raw_records):
        name=f"{row['area']}_block_{row['block']}";oid=int(row['object_ids'][0]);group=(name,oid)
        if name in maps and group not in excluded:groups.setdefault(group,(i,objects[name][str(oid)]['object_type']))
    picked=list(groups.items()) if a.all_remaining else []
    for category,count in ([] if a.all_remaining else [('Car',31),('Building',30),('Ground',3)]):
        candidates=[(g,v) for g,v in groups.items() if v[1]==category]
        candidates.sort(key=lambda x:hashlib.sha256(('fresh-confirm-20261104:'+str(x[0])).encode()).hexdigest())
        if len(candidates)<count:raise ValueError('insufficient new targets: '+category)
        picked.extend(candidates[:count])
    picked.sort(key=lambda x:(x[0][0],x[1][0]));public=[];private=[]
    for (name,oid),(index,category) in picked:
        instruction=ds[index].instruction;key=f'train_{index:05d}'
        public.append(dict(task_key=key,map_name=name,instruction=instruction))
        private.append(dict(task_key=key,index=index,physical_group=[name,oid],category=category))
    a.output.mkdir(parents=True)
    (a.output/'public_queries.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in public));(a.output/'selection_private.json').write_text(json.dumps(private,indent=2))
    freeze=dict(split='train_seen',indices=[x['index'] for x in private],local_checkpoint_sha256=file_sha256(a.local_checkpoint),
        source_feature_manifest_sha256=file_sha256(a.feature_manifest),annotation_sha256=file_sha256(ds.annotation_path),public_queries_sha256=file_sha256(a.output/'public_queries.jsonl'),
        excluded_group_count=len(excluded),categories=dict(Counter(x['category'] for x in private)),maps=sorted(maps),
        sampling=('all remaining physical targets on held-out TRAIN maps' if a.all_remaining else '31Car30Building3Ground new physical targets on prior held-out TRAIN maps; no new Parking available after exclusions')+'; no GT-coordinate/visibility/outcome filtering',
        available_group_count=len(groups),excluded_tasks_sha256={str(p):file_sha256(p) for p in a.exclude_tasks},
        independence='new head fit/selection independent; original A3 pretraining still legal TRAIN',
        no_refitting_after_confirmation=True)
    (a.output/'selection_freeze.json').write_text(json.dumps(freeze,indent=2));print(json.dumps(freeze))


if __name__=='__main__':main()
