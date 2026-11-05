"""Experimental input interventions; original Unified V2/controller stay unchanged."""
import torch
from .semantic_geometry import NODE_TYPES


def manual_relation_inputs(inputs,mode):
    if mode not in ('full','additive_off','manual_off'):raise ValueError('Unknown rule intervention')
    if mode!='manual_off':return inputs
    original=inputs['static_inputs'];s=dict(original)
    s['rules']=torch.zeros_like(original['rules']);s['relation_fields']=torch.zeros_like(original['relation_fields'])
    relation=original['node_types']==NODE_TYPES['relation'];keep=~relation
    s['node_valid']=original['node_valid']&keep
    s['nodes']=original['nodes'].masked_fill(relation[...,None],0)
    s['adjacency']=original['adjacency']&keep[:,:,None]&keep[:,None,:]
    s['edges']=original['edges'].masked_fill((relation[:,:,None]|relation[:,None,:])[...,None],0)
    # Public Polygon features, prior, entity mask and entity-name embeddings survive.
    return {**inputs,'static_inputs':s,'node_emb':inputs['node_emb'].masked_fill(relation[...,None],0)}


def forward_rule_control(model,inputs,mode='full'):
    previous=model.geometry.rule_weight
    try:
        model.geometry.rule_weight=previous if mode=='full' else 0.
        model.reset_episode()
        return model.forward_full_relation(**manual_relation_inputs(inputs,mode),observed=None)
    finally:
        model.geometry.rule_weight=previous
        model.reset_episode()


def public_text_record(map_name,instruction,bounds,catalog,matcher,*,mode,max_landmarks=16):
    """Same blind binding as TRAIN preparation, no IDs/target/trajectory arguments."""
    from scripts.prepare_unified_v2_landmark_relation_ablation import extract_pairs,public_references,normalized_instruction
    if mode not in ('full_text','landmark_relation'):raise ValueError('Only the two requested methods may navigate')
    mentions=matcher.match(map_name,instruction);pairs=extract_pairs(instruction,mentions)
    refs,warnings=public_references(map_name,bounds,mentions,catalog,max_landmarks)
    legal={r['id'] for r in refs};pairs=[p for p in pairs if any(str(i) in legal for i in p['public_ids'])]
    text=instruction if mode=='full_text' else normalized_instruction(pairs,mode)
    record=dict(map_name=map_name,instruction=text,world_bounds=list(bounds),physical_map_sizes=[bounds[2]-bounds[0],bounds[3]-bounds[1]],landmarks=refs,dataset_version='revised',match_mode='fuzzy',target_xy=[0.,0.])
    return record,dict(original_instruction=instruction,model_instruction=text,mode=mode,public_polygon_count=len(refs),pairs=[dict(name=p['name'],phrase=p['phrase']) for p in pairs],warnings=warnings,GT_input=False)
