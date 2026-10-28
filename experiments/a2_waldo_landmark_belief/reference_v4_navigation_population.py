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
    elif scope in FULL_COUNTS:
        entries=[dict(index=row.index,map_name=row.map_name,object_id=row.object_id,
                      description_id=row.description_id,episode_id=list(row.episode_id)) for row in dataset]
    else:
        raise ValueError('Only fixed pilot and complete val_seen/val_unseen scopes are supported')
    require_population(scope,entries)
    return entries
