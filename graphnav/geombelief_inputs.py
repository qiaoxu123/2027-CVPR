"""Parameter-free map-input controls; no target coordinates are read."""
import hashlib
import numpy as np
import torch

ARMS=('name_type','shuffled','distance','direction','distance_direction')
SPATIAL=('nodes','prior','prior_buildings','rules','relation_fields','pair_geometry','edges','adjacency')

def mask_geometry(x,arm):
    if arm not in ('name_type','distance','direction','distance_direction'):
        raise ValueError(arm)
    out=dict(x)
    for k in SPATIAL:out[k]=torch.zeros_like(x[k])
    channels={'name_type':(), 'distance':(2,), 'direction':(3,4), 'distance_direction':(2,3,4)}[arm]
    for c in channels:out['pair_geometry'][...,c]=x['pair_geometry'][...,c]
    # grid_xy is the shared candidate OUTPUT coordinate lattice, never a landmark
    # location. The frozen architecture has relative_position=False (asserted).
    # All relation-node geometry, not just physical nodes, is erased.
    return out


def shuffle_manifest(metadata,node_types,node_valid,entity_valid,partition,seed=20261017):
    """Permutation of complete spatial bundles within map/count/schema strata.

    Instructions, name embeddings, types and masks stay with the recipient.
    Exact marginal distribution of every spatial tensor is retained, including
    prior/rules; spatial rule fields deliberately no longer ground recipient text.
    Train-Fit and Train-Dev never donate to each other. No targets are consulted.
    """
    groups={}
    for i,m in enumerate(metadata):
        key=(partition[i],m['map_name'],tuple(node_types[i].tolist()),tuple(node_valid[i].tolist()),tuple(entity_valid[i].tolist()))
        groups.setdefault(key,[]).append(i)
    donors=list(range(len(metadata)));rng=np.random.default_rng(seed)
    for key,indices in sorted(groups.items(),key=lambda p:repr(p[0])):
        if len(indices)>1:
            order=rng.permutation(indices)
            for a,b in zip(order,np.roll(order,1)):donors[int(a)]=int(b)
    assert sorted(donors)==list(range(len(metadata)))
    return [dict(index=i,query_id=m['query_id'],donor_index=donors[i],donor_query_id=metadata[donors[i]]['query_id'],map_name=m['map_name'],partition=partition[i],unchanged=donors[i]==i,seed=seed) for i,m in enumerate(metadata)]
