"""Small immutable-parent reference overlays; no image or private-label copies."""
import hashlib
import json
from pathlib import Path
from copy import deepcopy

import torch

from .paths import sha


KIND = 'a2-reference-overlay-v1'
BINDING_KEYS = ('role','source_kind','directory','filename','episode','step','map_name','object_id','sha256')
NETWORK_KEYS = {'mention_mask','landmark_name_tokens','name_valid','link_valid','pair_geometry','pair_valid'}
META_KEYS = {'references','reference_surfaces','valid_references','public_reference_records',
             'reference_rebuild_version','reference_rebuild_seconds','extraction_seconds'}


def _binding(entry):
    return {key:entry[key] for key in BINDING_KEYS}


def overlay_filename(entry):
    payload = json.dumps(_binding(entry),sort_keys=True,separators=(',',':')).encode()
    return hashlib.sha256(payload).hexdigest()+'.pt'


def make_reference_overlay(public, meta, entry, signature_sha256):
    if entry['role'] not in ('train','dev'):
        raise ValueError('Only TRAIN/DEV reference overlays are authorized')
    return dict(kind=KIND,signature_sha256=signature_sha256,parent_binding=_binding(entry),
                network={key:public['network'][key].detach().cpu().clone() for key in NETWORK_KEYS},
                meta={key:deepcopy(meta[key]) for key in META_KEYS})


def apply_reference_overlay(raw, overlay, entry, signature_sha256):
    if (set(overlay) != {'kind','signature_sha256','parent_binding','network','meta'}
        or overlay['kind'] != KIND or overlay['signature_sha256'] != signature_sha256
        or overlay['parent_binding'] != _binding(entry) or entry['role'] not in ('train','dev')):
        raise ValueError('Foreign reference overlay or parent binding')
    if (raw['episode'] != entry['episode'] or raw['step'] != entry['step']
        or raw['target_instance'] != [entry['map_name'],entry['object_id']]):
        raise ValueError('Reference overlay raw parent identity mismatch')
    network,meta = overlay['network'],overlay['meta']
    if set(network) != NETWORK_KEYS or set(meta) != META_KEYS:
        raise ValueError('Unexpected/private reference overlay fields')
    refs = meta['references']; l = max(len(refs),1)
    original = raw['public']['network']
    n,t,d = len(original['box_rgb']),len(original['instruction_tokens']),original['instruction_tokens'].shape[-1]
    if meta['valid_references'] != len(refs) or len(meta['reference_surfaces']) != len(refs):
        raise ValueError('Reference overlay metadata axes changed')
    expected = dict(mention_mask=(l,t),link_valid=(l,),pair_geometry=(n,l,14),pair_valid=(n,l))
    for key,shape in expected.items():
        if not torch.is_tensor(network[key]) or tuple(network[key].shape) != shape:
            raise ValueError('Reference overlay feature axes changed')
    names,valid = network['landmark_name_tokens'],network['name_valid']
    if (not torch.is_tensor(names) or names.ndim != 3 or names.shape[0] != l or names.shape[-1] != d
        or names.shape[1]<1 or not torch.is_tensor(valid) or valid.shape != names.shape[:2]):
        raise ValueError('Reference overlay name axes changed')
    for key in ('mention_mask','name_valid','link_valid','pair_valid'):
        if network[key].dtype != torch.bool:
            raise ValueError('Reference overlay validity dtype changed')
    for key in ('landmark_name_tokens','pair_geometry'):
        if network[key].dtype != torch.float32 or not torch.isfinite(network[key]).all():
            raise ValueError('Invalid reference overlay floating features')
    link = network['link_valid']
    if not torch.equal(link,torch.arange(l,device=link.device)<len(refs)):
        raise ValueError('Reference overlay placeholder validity changed')
    result = deepcopy(raw)
    result['public']['network'].update(deepcopy(network))
    result['meta'].update(deepcopy(meta))
    return result


def read_reference_overlay(directory, entry, *, role, allowed_targets):
    """Caller binds manifest/signature; role guard precedes torch deserialization."""
    if role not in ('train','dev') or entry['role'] != role or (entry['map_name'],entry['object_id']) not in allowed_targets:
        raise ValueError('Calibration/foreign target forbidden before overlay deserialization')
    path = Path(directory)/overlay_filename(entry)
    if sha(path) != entry['overlay_sha256']:
        raise ValueError('Reference overlay content changed')
    return torch.load(path,map_location='cpu',weights_only=True)


def verify_overlay_population(records, expected):
    actual = [overlay_filename(entry) for entry in records]
    parents = [overlay_filename(entry) for entry in expected]
    if len(actual) != len(set(actual)) or len(parents) != len(set(parents)) or set(actual) != set(parents):
        raise ValueError('Reference overlay population incomplete, duplicated or foreign')
    for entry in records:
        digest = entry.get('overlay_sha256','')
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid reference overlay content hash')
