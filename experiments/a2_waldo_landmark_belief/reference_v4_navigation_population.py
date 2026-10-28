"""Public episode identities for the fixed pilot and both complete V4 splits."""
import json
from pathlib import Path

from .reference_overlay_context import BASE
from .reference_v4_navigation_protocol import prepare_pilot
from .revision_eval_protocol import require_population,FULL_COUNTS


def source_entries(dataset,scope,*,root=None):
    root=Path(root) if root is not None else BASE/'revision_v4'
    if scope=='pilot':
        prepare_pilot(root)
        entries=json.loads((root/'navigation_pilot_population.json').read_text())['entries']
        require_population(scope,entries)
        normalized=[]
        for entry in entries:
            index=entry['index']
            if not isinstance(index,int) or not 0<=index<len(dataset):
                raise ValueError('Pilot source index is outside the original TRAIN dataset')
            record=dataset[index]
            if record.index!=index or any(getattr(record,key)!=entry[key] for key in ('map_name','object_id','description_id')):
                raise ValueError('Fixed pilot query differs from original dataset record')
            episode_id=list(record.episode_id)
            if 'episode_id' in entry and list(entry['episode_id'])!=episode_id:
                raise ValueError('Pilot episode identity changed')
            normalized.append(dict(entry,episode_id=episode_id))
        entries=normalized
    elif scope in FULL_COUNTS:
        entries=[dict(index=row.index,map_name=row.map_name,object_id=row.object_id,
                      description_id=row.description_id,episode_id=list(row.episode_id)) for row in dataset]
    else:
        raise ValueError('Only fixed pilot and complete val_seen/val_unseen scopes are supported')
    require_population(scope,entries)
    return entries
