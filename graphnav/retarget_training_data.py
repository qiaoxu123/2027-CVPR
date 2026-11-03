"""TRAIN teacher bookkeeping; never select target IDs during normal inference."""
import hashlib
import torch

PUBLIC_FEATURE_FIELDS = ('prior_logits','local_uv','image_features','target_text','context_text',
    'public_geometry','quality','local_valid','candidate_world','frame_key','task_key','map_name','subset','step')


def select_training_query(indices, map_name, object_id):
    if not indices: raise ValueError('TRAIN instruction required')
    return min(set(indices), key=lambda i: hashlib.sha256(f'retarget-seed0:{map_name}:{object_id}:{i}'.encode()).hexdigest())


def weak_instance_mask(proposals, target_id, candidate_count):
    mask=torch.full((candidate_count,),-1,dtype=torch.int8)
    occupied=set()
    for oid,indices in proposals.items():
        if any(i<0 or i>=candidate_count or i in occupied for i in indices):
            raise ValueError('conflicting/out-of-range private core labels')
        mask[indices]=1 if oid==target_id else 0
        occupied.update(indices)
    return mask


def retarget_overrides(source, query, public_geometry):
    """Reuse only language-independent pixels/geometry; replace ALL text paths."""
    if source['prior_logits'].shape!=query['prior_logits'].shape:
        raise ValueError('retarget fixed global support mismatch')
    if 'map_name' in query and source.get('map_name')!=query['map_name']:
        raise ValueError('cross-map image re-pairing prohibited')
    output={key:value for key,value in source.items() if key in PUBLIC_FEATURE_FIELDS}
    for key in ('prior_logits','target_text','context_text','full_text','target_phrase'):
        output[key]=query[key]
    output['public_geometry']=public_geometry
    return output
