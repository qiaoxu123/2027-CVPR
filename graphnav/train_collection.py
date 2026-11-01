"""Strict Train grouping gate for on-policy data collection."""
import json
from pathlib import Path
import torch


def verified_train_indices(dataset,manifest_path,checkpoint,limit=None):
    manifest=json.loads(Path(manifest_path).read_text())
    if manifest.get('partition')!='train' or manifest.get('status')!='complete':
        raise ValueError('Completed explicit Train partition required')
    indices=manifest['episode_indices']
    if not indices or len(set(indices))!=len(indices):
        raise ValueError('Nonempty unique episode indices required')
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    dev_keys={(dataset[i].map_name,dataset[i].instruction) for i in state['config']['dev_episodes']}
    for index in indices:
        if not isinstance(index,int) or not 0<=index<len(dataset):raise ValueError('Invalid Train index')
        r=dataset[index]
        if (r.map_name,r.instruction) in dev_keys:raise ValueError('Train collection intersects held-out Dev group')
    if limit is not None and limit<=0:raise ValueError('Positive episode limit required')
    return indices if limit is None else indices[:limit]
