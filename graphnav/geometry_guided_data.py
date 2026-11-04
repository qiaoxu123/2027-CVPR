"""Explicit separation of public visual inputs and public-region weak losses."""
import json
import hashlib
from pathlib import Path
import numpy as np
import shapely
import torch
from .checkpointing import file_sha256
from .landmark_region_supervision import named_union


def load_landmark_data(data,teacher,catalog_path):
    proof=json.loads((data/'manifest.json').read_text());tp=json.loads((teacher/'manifest.json').read_text())
    if tp['data_manifest_sha256']!=file_sha256(data/'manifest.json') or tp['teacher_sha256']!=file_sha256(teacher/'teacher_public.pt'):
        raise ValueError('Teacher/data cache mismatch')
    for filename,key in [('samples_public.json','samples_sha256'),('queries_public.json','queries_sha256'),('region_supervision.pt','labels_sha256')]:
        if file_sha256(data/filename)!=proof[key]:raise ValueError('Landmark data artifact changed: '+filename)
    manifests={}
    for filename,expected in proof['source_manifests'].items():
        path=Path(filename)
        if file_sha256(path)!=expected:raise ValueError('Source feature manifest changed')
        manifests[str(path.parent)]={c['file']:c['sha256'] for c in json.loads(path.read_text())['chunks']}
    samples=json.loads((data/'samples_public.json').read_text());queries=json.loads((data/'queries_public.json').read_text())
    labels=torch.load(data/'region_supervision.pt',weights_only=True)
    teachers=torch.load(teacher/'teacher_public.pt',weights_only=True)
    catalog=json.loads(catalog_path.read_text())
    frames={};needed={}
    for r in samples:
        if r['query_key'] not in teachers:continue
        key=(r['cache'],r['chunk']);needed.setdefault(key,set()).add(r['row'])
    for (cache,chunk),indices in needed.items():
        if file_sha256(Path(cache)/chunk)!=manifests[cache][chunk]:raise ValueError('Feature chunk changed')
        records=torch.load(Path(cache)/chunk,weights_only=True)['records']
        for i in indices:
            r=records[i];frames[r['frame_key']]={k:r[k] for k in ('image_features','local_uv','candidate_world','local_valid','quality')}
    region_cache={};wrong_cache={};result=[]
    for r in samples:
        key=r['query_key']
        if key not in teachers:continue
        q=queries[key];t=teachers[key];f=frames[r['frame_key']]
        if key not in region_cache:region_cache[key]=named_union(q['public_polygons'])
        region=region_cache[key]
        points=shapely.points(f['candidate_world'].numpy())
        distance=shapely.distance(points,region).astype(np.float32)
        valid=f['local_valid'].bool()&(f['local_uv']>=0).all(-1)&(f['local_uv']<=1).all(-1)
        projection_supported=bool(((torch.from_numpy(distance[784:])<=.001)&valid).any())
        wrong_supported=False
        if t['metadata']['wrong_polygon_ids']:
            if key not in wrong_cache:
                ids=t['metadata']['wrong_polygon_ids'];count=len(q['public_ids'])
                contours=[catalog[q['map_name']][ids[i%len(ids)]]['contour'] for i in range(count)]
                wrong_cache[key]=named_union(contours)
            wrong_distance=shapely.distance(shapely.points(f['candidate_world'][784:].numpy()),wrong_cache[key])
            wrong_supported=bool(((torch.from_numpy(wrong_distance)<=.001)&valid).any())
        loss_labels=labels[r['sample_key']]['labels'].long().clone().masked_fill(~valid,-1)
        result.append(dict(metadata=r,query=q,frame=f,teacher=t,labels=loss_labels,
            valid=valid,region_distance=torch.from_numpy(distance[:784]),projection_supported=projection_supported,
            wrong_projection_supported=wrong_supported))
    return result


def batch_landmarks(rows,device='cuda',control='real',replacement=None):
    images=torch.stack([r['frame']['image_features'] for r in rows]).to(device)
    if control in ('wrong_rgb','same_type_rgb'):
        if replacement is None:raise ValueError('Frozen replacement mapping required')
        images=torch.stack([r['frame']['image_features'] for r in replacement]).to(device)
    if control=='zero_rgb':images=torch.zeros_like(images)
    valid=torch.stack([r['valid'] for r in rows]).to(device)
    if control=='shuffle_rgb':
        for i in range(len(images)):
            ids=valid[i].nonzero().flatten();images[i,ids]=images[i,ids.flip(0)].clone()
    geometry_key='wrong_polygon_geometry' if control=='wrong_polygon' else 'geometry'
    return dict(image_features=images,name_text=torch.stack([r['teacher']['name_text'] for r in rows]).to(device),
        type_text=torch.stack([r['teacher']['type_text'] for r in rows]).to(device),valid=valid,
        local_uv=torch.stack([r['frame']['local_uv'] for r in rows]).to(device),
        geometry_logits=torch.stack([r['teacher'].get(geometry_key,r['teacher']['geometry']) for r in rows]).to(device),
        labels=torch.stack([r['labels'] for r in rows]).to(device),
        projection_supported=torch.tensor([r['wrong_projection_supported'] if control=='wrong_polygon' else r['projection_supported'] for r in rows],device=device),
        view_ids=torch.tensor([int.from_bytes(hashlib.sha256(r['metadata']['frame']['image_sha256'].encode()).digest()[:8],'little')%(2**63-1) for r in rows],device=device),
        group_ids=torch.tensor([int.from_bytes(hashlib.sha256(r['metadata']['query_key'].encode()).digest()[:8],'little')%(2**63-1) for r in rows],device=device))
