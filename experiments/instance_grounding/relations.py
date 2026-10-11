"""Offline conservative relation judgments, never normal Head inputs.

A relation word alone does not make every other identity a semantic negative.
Only a named actually-observed anchor with separated geometry can supply a
weak contradiction; unresolved same-appearance instances remain Unknown.
"""
import re
import numpy as np
from shapely.geometry import Polygon
from graphnav.semantic_geometry import parse_spatial_relations,DIRECTIONS,canonical

def relation_mismatches(instruction,regions,target_id):
    text=canonical(instruction)
    referenced=[r for r in regions if canonical(r.get('name','')) and re.search(r'\b'+re.escape(canonical(r['name']))+r'\b',text)]
    if not referenced:return set()
    entities=[dict(name=r.get('name','')) for r in referenced]
    relations=parse_spatial_relations(instruction,entities);target=next((r for r in regions if r['object_id']==target_id),None)
    result=set()
    if target is None:return result
    for rel in relations:
        if rel.kind not in (*DIRECTIONS,'near'):continue
        anchors=[referenced[k] for k in rel.anchors[0] if referenced[k]['object_id']!=target_id]
        if not anchors:continue
        def status(r,a):
            if r['object_id']==a['object_id']:return 0
            if rel.kind in DIRECTIONS:
                direction=np.asarray(DIRECTIONS[rel.kind],float);direction/=np.linalg.norm(direction)
                signed=float((np.asarray(r['center_world'])-np.asarray(a['center_world']))@direction)
                return 1 if signed>=10 else (-1 if signed<=-10 else 0)
            # Public geometry is weak supervision, not an RGB truth.
            d=Polygon(r['contour']).distance(Polygon(a['contour']))
            return 1 if d<=25 else (-1 if d>=50 else 0)
        if not any(status(target,a)==1 for a in anchors):continue
        for r in regions:
            if all(status(r,a)==-1 for a in anchors):result.add(r['object_id'])
    return result
