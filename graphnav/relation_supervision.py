"""Conservative real-query geometry labels, used exclusively by auxiliary loss."""
import re
import numpy as np
import torch
from .semantic_geometry import DIRECTIONS, geometry_distance, polygon_center
from .candidate_geometry import FullCandidateDataset,collate_full
from .semantic_graph import prepare_semantic_graph
from .semantic_geometry import prior_and_rule_fields


def legal_region(points, entities, relation):
    """Multiple legal locations remain positives. Ambiguous name bindings skipped."""
    p=np.asarray(points,np.float32)
    if any(len(group)!=1 for group in relation.anchors):return None
    anchors=[entities[group[0]] for group in relation.anchors]
    if relation.kind in DIRECTIONS:
        axis=np.asarray(DIRECTIONS[relation.kind],np.float32)
        axis/=np.linalg.norm(axis)
        # 5m boundary margin excludes uncertain directional boundary cases.
        return ((p-polygon_center(anchors[0]['geometry_m']))@axis)>5
    if relation.kind=='near':
        return geometry_distance(p,anchors[0]['geometry_m'],anchors[0]['kind']=='road')<=30
    if relation.kind=='along':
        # Surface polygons are not verified road centerlines.
        if anchors[0]['kind']!='road':return None
        return geometry_distance(p,anchors[0]['geometry_m'],True)<=20
    if relation.kind=='between' and len(anchors)==2:
        a,b=[polygon_center(x['geometry_m']) for x in anchors]
        delta=b-a;length2=float(delta@delta)
        if length2<100:return None
        t=(p-a)@delta/length2
        distance=np.linalg.norm(p-(a+t[...,None]*delta),axis=-1)
        return (t>.05)&(t<.95)&(distance<=20)
    return None


class RelationDataset(FullCandidateDataset):
    """Only TRAIN labels may filter contradictory/uncertain auxiliary constraints."""
    def __getitem__(self,idx):
        sample=super().__getitem__(idx)
        graph=self._prepared(idx)[0];record=self.records[idx]
        low=np.asarray(record['world_bounds'][:2],np.float32)
        span=np.asarray(record['world_bounds'][2:],np.float32)-low
        grid=low+self.grid_xy*span
        # Target is a private loss label, never part of the forward whitelist.
        target=low+np.asarray(record['target_xy'],np.float32)*span
        allowed=np.ones(len(grid),bool);used=0;cardinal=[]
        for relation in graph['relations']:
            mask=legal_region(np.concatenate([grid,target[None]]),graph['entities'],relation)
            if mask is None or not mask[-1] or not mask[:-1].any():continue
            proposal=allowed & mask[:-1]
            if not proposal.any():continue
            allowed=proposal;used+=1
            if relation.kind in ('east','west','north','south'):cardinal.append(relation)
        sample['relation_allowed']=torch.from_numpy(allowed)
        sample['relation_eligible']=torch.tensor(used>0)
        sample['relation_used']=torch.tensor(used)
        # A controlled inverse phrase never rewrites cardinal words in names.
        inverse={'east':'west','west':'east','north':'south','south':'north'}
        original=record['instruction']
        matches=list(re.finditer(r'\bto the (east|west|north|south) of\b',original,flags=re.I))
        swapped=original
        counter_valid=False;counter_allowed=np.ones(len(grid),bool)
        if len(matches)==1 and len(cardinal)==1 and cardinal[0].kind==matches[0][1].lower():
            m=matches[0];kind=inverse[m[1].lower()]
            swapped=original[:m.start()]+f'to the {kind} of'+original[m.end():]
            from .semantic_geometry import SpatialRelation
            counter_allowed=legal_region(grid,graph['entities'],SpatialRelation(kind,cardinal[0].anchors))
            counter_valid=counter_allowed is not None and counter_allowed.any()
        cg=prepare_semantic_graph(record.get('landmarks',[])[:self.max_landmarks],[],swapped,record['world_bounds'],max_landmarks=self.max_landmarks,max_roads=self.max_roads,max_relations=self.max_relations)
        _,rules,fields=prior_and_rule_fields(grid,cg['entities'],cg['relations'],sigma_m=self.rule_sigma_m)
        padded=np.zeros((len(grid),self.max_relations),np.float32)
        if len(cg['relations']):padded[:,:fields.shape[-1]]=fields
        sample['counter_text']=swapped
        sample['counter_node_names']=cg['node_names']
        for key,value in [('nodes',cg['nodes']),('node_types',cg['node_types']),('node_valid',cg['node_mask']),('adjacency',cg['adjacency']),('edges',cg['edges']),('rules',rules),('relation_fields',padded)]:sample['counter_'+key]=torch.from_numpy(value)
        sample['counter_allowed']=torch.from_numpy(counter_allowed if counter_allowed is not None else np.ones(len(grid),bool))
        sample['counter_eligible']=torch.tensor(counter_valid)
        return sample


def collate_relation(items):
    extra=('counter_text','counter_node_names')
    result=collate_full([{k:v for k,v in item.items() if k not in extra} for item in items])
    result.update({k:[item[k] for item in items] for k in extra})
    return result
