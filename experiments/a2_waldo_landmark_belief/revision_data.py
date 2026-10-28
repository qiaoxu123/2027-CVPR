"""Revision labels from trusted raw associations; immutable cached public inputs."""
import json
from pathlib import Path
import torch
from .paths import sha
from .data_protocol import split_target_groups
from .collected_snapshot import BASE,cache_source,snapshot_partition
from .revision_fusion import a0_candidate_context


def revision_labels(row,partition,*,reserved):
    map_name,target_id=row['target_instance']
    if (map_name,target_id) in reserved:raise ValueError('Reserved confirmation target in development data')
    if snapshot_partition(map_name,target_id)!=partition:raise ValueError('Revision target partition changed')
    y=dict(row['labels']);known=torch.zeros_like(y['rank_label_mask'],dtype=torch.bool)
    truth=torch.zeros_like(y['ranking']);details=row['private']['association']
    for i,detail in enumerate(details):
        oid=detail.get('object_id')
        if oid is None or not detail['trusted'] or (map_name,oid) in reserved:continue
        original=split_target_groups([dict(map_name=map_name,object_id=oid)],seed=0)[0]
        if original=='train' and snapshot_partition(map_name,oid)==partition:
            known[i]=True;truth[i]=float(oid==target_id)
    if truth.sum()>0:truth=truth/truth.sum()
    spatial=row['public']['spatial'];target=y['target_xy'].float()
    near=(spatial['grid_world'].float()-target).norm(dim=-1)<=20.
    support=spatial['candidate_support'].bool()
    distance=torch.cdist(spatial['candidate_xy'].float(),spatial['grid_world'].float()).masked_fill(~support,torch.inf)
    peak=distance.argmin(-1)
    covered=support.any(-1) & near[peak]
    # Actual ZERO-offset local grid reachability, not old learned-offset bounds.
    trust=(truth>0) & covered & bool(row['private']['target_visible'])
    y.update(ranking=truth,rank_label_mask=known,offset_valid=trust,
             candidate_trust=trust,match=trust.any().float(),match_valid=trust.any() | known.all(),
             target_kind=torch.tensor(int(row['private']['target_kind']),dtype=torch.long))
    return y


class RevisionViews(torch.utils.data.Dataset):
    def __init__(self,directory,partition,*,reserve_file=BASE/'revision_confirmation_reserve.json'):
        self.directory=Path(directory);self.partition=partition
        self.manifest=json.loads((self.directory/'manifest.json').read_text())
        signature=json.loads((self.directory/'signature.json').read_text())
        if signature.get('kind')!='user_stopped_collection_target_disjoint_snapshot_v1':
            raise ValueError('Revision requires explicit frozen collected snapshot')
        self.raw_signature=signature['parent_signature_sha256']
        self.entries=[e for e in self.manifest['entries'] if e['split']==partition]
        self.source={e['index']:e for e in cache_source(self.directory)['entries']}
        reserve=json.loads(Path(reserve_file).read_text())
        self.reserved={(e['map_name'],e['object_id']) for e in reserve['targets']}
        if len(self.reserved)!=256:raise ValueError('Independent confirmation reserve changed')
        self.reserve_sha256=sha(reserve_file)

    def __len__(self):return len(self.entries)

    def __getitem__(self,index):
        entry=self.entries[index];expected=self.source[entry['episode']]
        row=torch.load(self.directory/entry['filename'],map_location='cpu',weights_only=False)
        if (row['signature_sha256']!=self.raw_signature or row['episode']!=entry['episode']
            or row['step']!=entry['step'] or row['target_instance']!=[expected['map_name'],expected['object_id']]
            or expected['split']!=self.partition):raise ValueError('Revision raw view identity/provenance mismatch')
        labels=revision_labels(row,self.partition,reserved=self.reserved)
        network=dict(row['public']['network'])
        network['a0_context']=a0_candidate_context(row['public']['spatial'])
        kind=torch.zeros(len(network['box_rgb']),dtype=torch.long)
        for i,candidate in enumerate(row['meta']['candidates']):kind[i]=candidate['kind']
        network['box_kind']=kind
        return dict(row,labels=labels,public=dict(row['public'],network=network))
