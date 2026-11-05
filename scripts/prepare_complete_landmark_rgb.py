"""All Revised TRAIN public-reference crops, GT kept in a separate label file."""
import argparse,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
import shapely
from graphnav.complete_landmark_rgb import tile_plan
from graphnav.landmark_alias import canonical_name
from graphnav.checkpointing import atomic_json,file_sha256


def complete_public_parts(original,objects,bounds):
    refs=list(original);ids={str(x['id']) for x in original};keys={(canonical_name(x['name']),x['object_type']) for x in original}
    b=np.asarray(bounds)
    for oid,obj in sorted(objects.items(),key=lambda x:str(x[0])):
        if str(oid) in ids or (canonical_name(obj.get('name','')),obj['object_type']) not in keys:continue
        poly=shapely.Polygon(obj['contour'])
        if not poly.is_valid or poly.area<=0:continue
        refs.append(dict(id=str(oid),name=obj['name'],object_type=obj['object_type'],contour=((np.asarray(obj['contour'])-b[:2])/(b[2:]-b[:2])).tolist()))
    return refs


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--public-objects',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Keep previous dataset inventory')
    a.output.mkdir(parents=True);objects=json.load(open(a.public_objects));records=[json.loads(x) for x in a.source.read_text().splitlines()];groups={};queries=[];labels=[];invalid=0
    for idx,record in enumerate(records):
        b=np.array(record['world_bounds']);extra=complete_public_parts(record['landmarks'],objects[record['map_name']],b);names=[];types=[];contours=[];bad=[]
        for lm in extra:
            c=b[:2]+np.asarray(lm['contour'])*(b[2:]-b[:2]);poly=shapely.Polygon(c)
            if not poly.is_valid or poly.area<=0:bad.append(lm['name']);invalid+=1;continue
            names.append(lm['name']);types.append(lm['object_type']);contours.append(c.tolist())
        key=None
        if contours:
            payload=dict(map_name=record['map_name'],names=names,types=types,contours=contours,world_bounds=b.tolist())
            key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:24]
            if key not in groups:
                plan=tile_plan(contours)
                groups[key]=dict(**payload,plan=json.loads(json.dumps(plan,default=lambda x:np.asarray(x).tolist())),query_count=0)
            groups[key]['query_count']+=1
        split='dev' if record['map_name'].startswith('birmingham') else 'fit'
        queries.append(dict(index=idx,map_name=record['map_name'],instruction=record['instruction'],subset=split,group=key,invalid_reference_names=bad,missing_reason=None if key else 'no_valid_public_polygon'))
        labels.append(dict(index=idx,target_normalized=record['target_xy'],target_world=(b[:2]+np.array(record['target_xy'])*(b[2:]-b[:2])).tolist()))
    maps={q['map_name']:q['subset'] for q in queries};assert not {q['group'] for q in queries if q['subset']=='fit'}&{q['group'] for q in queries if q['subset']=='dev'}-{None}
    for k,v in [('groups.json',groups),('queries_public.json',queries),('labels_private.json',labels)]:atomic_json(v,a.output/k)
    atomic_json([x for x in labels if queries[x['index']]['subset']=='fit'],a.output/'fit_labels_private.json')
    counts=Counter(q['subset'] for q in queries);tiles=[len(x['plan']['tiles']) for x in groups.values()]
    report=dict(public_objects_SHA256=file_sha256(a.public_objects),reference_expansion='All same-map canonical-name and same-type public pieces, never GT selected',source_SHA256=file_sha256(a.source),source=str(a.source),all_queries=len(queries),split=dict(counts),maps=maps,unique_reference_sets=len(groups),
        unique_raster_views=sum(1+n for n in tiles),max_tiles=max(tiles),median_tiles=float(np.median(tiles)),no_public_polygon=sum(q['group'] is None for q in queries),invalid_public_parts=invalid,
        split_protocol='Cambridge fit / Birmingham Dev: map and city-name physical references disjoint; frozen core has trained on all TRAIN',GT_selects_crops=False,
        scales_m=[128,64],local_context_m=32.,margin_fraction=.25,all_inputs_privileged=True,normal_navigation_allowed=False)
    atomic_json(report,a.output/'manifest.json');print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
