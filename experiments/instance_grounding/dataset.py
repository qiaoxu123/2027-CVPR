"""Bound feature loading and explicit weak identity vs attribute supervision."""
import json,re
from pathlib import Path
import torch
from experiments.rgb_grounding.evaluate_features import load
from experiments.bidirectional_core.cache import DEFAULT_DATA,conservative_object_attributes,conservative_attributes
from graphnav.checkpointing import file_sha256
from .data import write
from .relations import relation_mismatches

RELATION=re.compile(r'\b(near|beside|behind|next to|between|left|right|north|south|east|west|front|across|along|opposite|adjacent|closest|farthest)\b',re.I)

def city_rows(root):
    root=Path(root);m=json.loads((root/'manifest.json').read_text());assert file_sha256(root/'features.pt')==m['feature_sha256'];data=torch.load(root/'features.pt',weights_only=True)
    descriptions=json.loads((Path(DEFAULT_DATA)/'cityrefer/processed_descriptions.json').read_text())
    attributes={};rows=[]
    objects=json.loads((Path(DEFAULT_DATA)/'cityrefer/objects.json').read_text())
    for state in data['plan']['states']:
        meta=state['metadata'];regions=state['regions'];n=len(regions);lab=data['plan']['labels'][meta['sample_id']];target=lab['target_object_id'];ins=meta['instruction'];query=data['queries'][ins]
        for r in regions:r['contour']=objects[meta['map_name']][r['object_id']]['contour']
        relation_neg=relation_mismatches(ins,regions,target)
        positive=torch.tensor([r['object_id']==target for r in regions],dtype=torch.bool);seen=positive.any();judged=positive.clone();weak_attribute_positive=torch.zeros(n,dtype=torch.bool);attribute_judged=weak_attribute_positive.clone()
        typ=next((r['object_type'] for r in regions if r['object_id']==target),None);clause=re.split(r'\b(?:beside|behind|next to|in front of|with|near)\b',ins,flags=re.I)[0]
        attrs=conservative_attributes(clause,typ) if typ else {};same_color=torch.zeros(n,dtype=torch.bool);same_class=torch.zeros(n,dtype=torch.bool)
        for j,r in enumerate(regions):
            key=(meta['map_name'],r['object_id']);r['attrs_status']='language-derived weak, not pixel-verified'
            if key not in attributes:attributes[key]=conservative_object_attributes(descriptions.get(meta['map_name'],{}).get(r['object_id'],[]),r['object_type'])
            ca={k:v['label'] for k,v in attributes[key].items()};r['weak_attrs']=ca
            same_class[j]=typ is not None and r['object_type']==typ
            colors=[k for k in attrs if 'color' in k];same_color[j]=bool(colors) and same_class[j] and all(ca.get(k)==attrs[k] for k in colors)
            if seen and not positive[j]:
                contrary=bool(attrs) and same_class[j] and any(k in ca and ca[k]!=v for k,v in attrs.items())
                # Same-appearance identity negatives require a relation in the
                # supplied referring expression; still weak, never human truth.
                relational=same_class[j] and r['object_id'] in relation_neg
                judged[j]=bool(contrary or relational)
            if attrs and same_class[j] and all(k in ca for k in attrs):
                attribute_judged[j]=True;weak_attribute_positive[j]=all(ca[k]==v for k,v in attrs.items())
        if not seen:judged.zero_();positive.zero_()  # absent target NEVER mismatch
        credible=torch.tensor([r['depth_support']>=.9 for r in regions],dtype=torch.bool);judged &= credible;positive &= credible
        features={name:torch.stack([data['bank'][r['view_key']][name] for r in regions]) if n else torch.zeros(0,1024) for name in ['tight','context']}
        rows.append(dict(pixel_zero=data['pixel_zero'],metadata={**meta,'roi_objects':regions},public=dict(roi_valid=torch.ones(n,dtype=torch.bool),roi_quality=torch.tensor([r['quality'] for r in regions]).reshape(n,3)),labels=dict(positive_roi_mask=positive,semantic_valid_mask=judged,identity_positive=torch.tensor([r['object_id']==target for r in regions],dtype=torch.bool),attribute_positive=weak_attribute_positive,attribute_judged=attribute_judged,same_class=same_class,same_color=same_color,target_credible=bool(seen and credible[torch.tensor([r['object_id']==target for r in regions],dtype=torch.bool)].any())),features=features,patch_tokens=torch.stack([data['bank'][r['view_key']]['patch_tokens'] for r in regions]) if n else torch.zeros(0,16,1024),texts=query['texts'],language_tokens=query['language_tokens'],language_valid=query['language_valid'],state=state,sealed=lab))
    return rows

def external_rows(root):
    rows=load(root)
    for r in rows:
        if 'context' not in r['features']:r['features']['context']=r['features']['tight']
    return rows

def collate(rows,control='real'):
    b=len(rows);n=max(1,max(len(r['public']['roi_valid']) for r in rows));d=1024;l=max(len(r['language_tokens']) for r in rows)
    p=dict(full=torch.stack([r['texts']['full'] for r in rows]),phrase=torch.stack([r['texts']['phrase'] for r in rows]),images=torch.zeros(b,n,d),context=torch.zeros(b,n,d),patch_tokens=torch.zeros(b,n,16,d),language_tokens=torch.zeros(b,l,d),language_valid=torch.zeros(b,l,dtype=torch.bool),quality=torch.zeros(b,n,3),valid=torch.zeros(b,n,dtype=torch.bool),relative=torch.zeros(b,n,n+1,3));pos=torch.zeros(b,n,dtype=torch.bool);judged=pos.clone()
    for i,r in enumerate(rows):
        c=len(r['public']['roi_valid']);w=len(r['language_tokens']);p['images'][i,:c]=r['features']['tight'];p['context'][i,:c]=r['features']['context'];p['patch_tokens'][i,:c]=r['patch_tokens'];p['language_tokens'][i,:w]=r['language_tokens'];p['language_valid'][i,:w]=r['language_valid'];p['quality'][i,:c]=r['public']['roi_quality'];p['valid'][i,:c]=r['public']['roi_valid'];pos[i,:c]=r['labels']['positive_roi_mask'];judged[i,:c]=r['labels']['semantic_valid_mask']
        if c:
            if 'state' in r:coordinates=torch.tensor([v['center_world'] for v in r['metadata']['roi_objects']],dtype=torch.float32)
            else:coordinates=torch.tensor([[(v['bbox'][0]+v['bbox'][2])/2,-(v['bbox'][1]+v['bbox'][3])/2] for v in r['metadata']['roi_objects']],dtype=torch.float32)
            delta=coordinates[:,None]-coordinates[None,:]
            scale=(coordinates.max(0).values-coordinates.min(0).values).clamp_min(1)
            delta=delta/scale
            p['relative'][i,:c,:c]=torch.cat([delta,delta.square().sum(-1,keepdim=True).sqrt()],-1)
        if control=='zero':
            if 'pixel_zero' in r:
                p['images'][i,:c]=r['pixel_zero']['image'];p['context'][i,:c]=r['pixel_zero']['image'];p['patch_tokens'][i,:c]=r['pixel_zero']['patch_tokens']
            else:
                for k in ['images','context','patch_tokens']:p[k][i,:c].zero_()
        elif control=='shuffled' and c>1:
            # Same-category cyclic permutation, deterministic and target-blind.
            groups={}
            for j,v in enumerate(r['metadata']['roi_objects']):groups.setdefault(v['object_type'],[]).append(j)
            perm=list(range(c))
            for group in groups.values():
                if len(group)>1:
                    for a,z in zip(group,group[1:]+group[:1]):perm[a]=z
            for k in ['images','context','patch_tokens']:p[k][i,:c]=p[k][i,perm].clone()
        elif control not in ('real','shuffled'):raise ValueError(control)
    return p,pos,judged

def training_unit(rows):
    # One deterministic latest prefix per family; repetitions aren't cases.
    groups={}
    for r in rows:
        key=r['metadata']['family_id'];old=groups.get(key)
        if old is None or r['metadata'].get('cutoff_step',0)>old['metadata'].get('cutoff_step',0):groups[key]=r
    return [groups[k] for k in sorted(groups)]
