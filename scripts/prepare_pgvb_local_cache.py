"""Local Anchor PGVB cache v2, retaining all legacy queries and positive-only regions."""
import argparse,collections,hashlib,json,time
from pathlib import Path
import cv2
import numpy as np
import shapely
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.landmark_alias import canonical_name
from graphnav.local_landmark_anchor import local_landmark_anchor,local_anchor_tokens,named_public_polygons
from graphnav.landmark_projection import world_to_frame
from graphnav.landmark_region_supervision import named_union,public_type,weak_region_labels
from graphnav.public_reference_supervision import patch_world_windows
from graphnav.polygon_guided_visual import polygon_shape_tokens
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from scripts.analyze_instruction_landmark_association import GENERIC_NAMES


def main():
    p=argparse.ArgumentParser()
    for k in ('source','old-data','catalog','backbone','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    a.output.mkdir(parents=True);torch.set_num_threads(4);start=time.perf_counter()
    old_proof=json.loads((a.old_data/'manifest.json').read_text())
    assert file_sha256(a.old_data/'cache.pt')==old_proof['cache_sha256']
    old=torch.load(a.old_data/'cache.pt',weights_only=True,map_location='cpu')
    old_keys={(old['frames'][r['frame_id']]['frame']['image_sha256'],r['query_key']) for r in old['records']}
    source=json.loads((a.source/'manifest.json').read_text());samples=json.loads((a.source/'samples_public.json').read_text())
    assert file_sha256(a.source/'samples_public.json')==source['samples_sha256']
    catalog=json.loads(a.catalog.read_text());public={}
    for mp,objects in catalog.items():
        by_name=collections.defaultdict(list)
        for oid,obj in objects.items():
            name=canonical_name(obj.get('name',''))
            if len(name)>=3 and name not in GENERIC_NAMES:by_name[name].append((oid,obj))
        public[mp]=[]
        for name,items in sorted(by_name.items()):
            typ=public_type([x['object_type'] for _,x in items])
            if typ not in ('Building','Road'):continue
            contours=[x['contour'] for _,x in items]
            try:region=named_union(contours)
            except ValueError:continue
            public[mp].append(dict(key=mp+'::'+name,name=items[0][1]['name'],type=typ,contours=contours,region=region))
    unique={}
    for r in samples:unique.setdefault(r['frame_key'],r)
    chunk_rows=collections.defaultdict(list)
    for r in unique.values():chunk_rows[(r['cache'],r['chunk'])].append(r)
    source_manifest={};frames=[];records=[];queries={};seen_hash={};excluded=collections.Counter()
    for chunk_no,((cache,chunk),wanted) in enumerate(sorted(chunk_rows.items())):
        if chunk_no%20==0:print(json.dumps(dict(chunks=chunk_no,total_chunks=len(chunk_rows),frames=len(frames),queries=len(records),seconds=time.perf_counter()-start)),flush=True)
        if cache not in source_manifest:
            path=Path(cache)/'manifest.json';assert file_sha256(path)==source['source_manifests'][str(path)]
            proof=json.loads(path.read_text());assert proof['split']=='train_seen' and proof['camera_yaw_rad']==0
            source_manifest[cache]={r['file']:r['sha256'] for r in proof['chunks']}
        path=Path(cache)/chunk;assert file_sha256(path)==source_manifest[cache][chunk]
        values=torch.load(path,weights_only=True)['records']
        for origin in wanted:
            f=values[origin['row']];meta=origin['frame'];mp=origin['map_name'];city=mp.split('_block_')[0]
            assert city in ('cambridge','birmingham')
            if meta['image_sha256'] in seen_hash:
                assert seen_hash[meta['image_sha256']]==city
                excluded['duplicate_rgb']+=1;continue
            valid=f['local_valid'].bool()&(f['local_uv']>=0).all(-1)&(f['local_uv']<=1).all(-1)
            valid &= f['quality'][:,0]>=.9
            if valid.sum()<8:excluded['insufficient_valid_patches']+=1;continue
            image_path=Path(origin['observations'])/meta['image'];depth_path=Path(origin['observations'])/meta['depth']
            if file_sha256(image_path)!=meta['image_sha256'] or file_sha256(depth_path)!=meta['depth_sha256']:raise ValueError('Real RGB/depth changed')
            bgr=cv2.imread(str(image_path))
            with np.load(depth_path) as z:depth=z['depth_valid'].copy()
            observation=dict(bgr=bgr,depth_valid=depth,footprint_world=np.asarray(meta['footprint_world']))
            windows=patch_world_windows(meta['footprint_world'])[:,0]
            footprint=shapely.Polygon(meta['footprint_world']);possible=[]
            for q in public[mp]:
                if not q['region'].intersects(footprint):continue
                labels,_=weak_region_labels(windows,q['region'],valid.numpy())
                if (labels==1).sum()<2:continue
                anchor=local_landmark_anchor(q['contours'],observation,patch_valid=valid.numpy())
                if not np.array_equal(anchor['labels'],labels):raise ValueError('Weak boundary/validity rules changed')
                if not anchor['train_eligible']:continue
                stats={k:anchor[k] for k in ('whole_fraction','view_fraction','stable_valid_pixels','local_source_support','positive_patches','negative_patches','positive_only')}
                possible.append((q,labels,stats))
            selected=[]
            for typ in ('Building','Road'):
                choices=[x for x in possible if x[0]['type']==typ]
                choices.sort(key=lambda x:hashlib.sha256((meta['frame_key']+'::'+x[0]['key']).encode()).hexdigest())
                # Preserve the exact old cohort. Do not evict old samples when
                # newly eligible full-view regions are added to the same frame.
                kept=[x for x in choices if (meta['image_sha256'],x[0]['key']) in old_keys]
                added=[x for x in choices if (meta['image_sha256'],x[0]['key']) not in old_keys and (x[1]==0).sum()<4]
                selected += kept+added[:2]
            if not selected:excluded['no_eligible_named_region']+=1;continue
            image_path=Path(origin['observations'])/meta['image'];depth_path=Path(origin['observations'])/meta['depth']
            if file_sha256(image_path)!=meta['image_sha256'] or file_sha256(depth_path)!=meta['depth_sha256']:raise ValueError('Real RGB/depth changed')
            seen_hash[meta['image_sha256']]=city;frame_id=len(frames)
            global_world=f['candidate_world'][:784].float();dx=float(global_world[1,0]-global_world[0,0]);dy=float(global_world[28,1]-global_world[0,1])
            bounds=[float(global_world[0,0]-dx/2),float(global_world[0,1]-dy/2),float(global_world[-1,0]+dx/2),float(global_world[-1,1]+dy/2)]
            frames.append(dict(image_features=f['image_features'].clone(),local_uv=f['local_uv'].clone(),valid=valid,
                global_world=global_world,bounds=bounds,image_path=str(image_path),frame=meta,map_name=mp,
                city=city,subset='fit' if city=='cambridge' else 'dev'))
            for q,labels,local_stats in selected:
                if q['key'] not in queries:queries[q['key']]={k:v for k,v in q.items() if k!='region'}
                distance=shapely.distance(shapely.points(global_world.numpy()),q['region']).astype(np.float32)
                area=(dx*dy);cell=shapely.box(global_world[:,0].numpy()-dx/2,global_world[:,1].numpy()-dy/2,global_world[:,0].numpy()+dx/2,global_world[:,1].numpy()+dy/2)
                fraction=np.clip(shapely.area(shapely.intersection(cell,q['region']))/area,0,1).astype(np.float32)
                records.append(dict(frame_id=frame_id,query_key=q['key'],type=q['type'],subset=frames[-1]['subset'],
                    labels=torch.from_numpy(labels).long(),polygon_tokens=torch.from_numpy(local_anchor_tokens(q['contours'],meta['footprint_world'])),
                    legacy_polygon_tokens=torch.from_numpy(polygon_shape_tokens(q['contours'],meta['footprint_world'])),
                    legacy_common=(meta['image_sha256'],q['key']) in old_keys,local_anchor=local_stats,
                    distance=torch.from_numpy(distance),area_fraction=torch.from_numpy(fraction),visibility='geometry_proxy_unknown'))
    # Fresh name vectors; backbone and processor unchanged. No original-language cache reuse.
    vectors={}
    for q in old['queries'].values():vectors[q['name']]=q['name_text'];vectors[q['type']]=q['type_text']
    texts=sorted({x for q in queries.values() for x in (q['name'],q['type'])}-set(vectors))
    native=ObservedSigLIPMatchingEncoder(str(a.backbone)) if texts else None
    with torch.inference_mode():
        for off in range(0,len(texts),32):
            part=texts[off:off+32];v=native.texts(part).cpu().half()
            vectors.update(zip(part,v))
    for q in queries.values():q['name_text']=vectors[q['name']];q['type_text']=vectors[q['type']]
    groups=collections.defaultdict(list)
    for i,r in enumerate(records):groups[(r['frame_id'],r['type'])].append(i)
    pair_counts=collections.Counter()
    for (fid,typ),ids in groups.items():
        for i in ids:
            records[i]['hard_negative_indices']=[j for j in ids if i!=j and not ((records[i]['labels']==1)&(records[j]['labels']!=0)).any() and not ((records[j]['labels']==1)&(records[i]['labels']!=0)).any()]
        if len(ids)>1:pair_counts[frames[fid]['subset']+'::'+typ]+=1
    new_keys={(frames[r['frame_id']]['frame']['image_sha256'],r['query_key']) for r in records}
    if not old_keys<=new_keys:raise ValueError('Legacy cohort was lost: '+str(len(old_keys-new_keys)))
    legacy_lookup={(old['frames'][r['frame_id']]['frame']['image_sha256'],r['query_key']):r for r in old['records']}
    for r in records:
        key=(frames[r['frame_id']]['frame']['image_sha256'],r['query_key'])
        if key in legacy_lookup:
            if not torch.equal(r['labels'],legacy_lookup[key]['labels']):raise ValueError('Legacy labels changed')
    proof=dict(complete=True,cache_version='local-anchor-v2',legacy_retained=len(old_keys),
        old_manifest_sha256=file_sha256(a.old_data/'manifest.json'),
        changes={s:dict(before=old_proof['counts'][s],
            after_queries=sum(r['subset']==s for r in records),
            added_positive_only=sum(r['subset']==s and not r['legacy_common'] and r['local_anchor']['negative_patches']==0 for r in records),
            added_few_negative=sum(r['subset']==s and not r['legacy_common'] and 0<r['local_anchor']['negative_patches']<4 for r in records),
            types_before=old_proof['counts'][s]['types'],
            types_after=dict(collections.Counter(r['type'] for r in records if r['subset']==s))) for s in ('fit','dev')},source_manifest_sha256=file_sha256(a.source/'manifest.json'),catalog_sha256=file_sha256(a.catalog),
        source_feature_manifests=source['source_manifests'],backbone=str(a.backbone),
        backbone_sha256={p.name:file_sha256(p) for p in a.backbone.glob('*.safetensors')},
        counts={s:dict(frames=sum(f['subset']==s for f in frames),queries=sum(r['subset']==s for r in records),
            physical_name_groups=len({r['query_key'] for r in records if r['subset']==s}),
            maps=len({f['map_name'] for f in frames if f['subset']==s}),
            types=dict(collections.Counter(r['type'] for r in records if r['subset']==s))) for s in ('fit','dev')},
        same_image_groups=dict(pair_counts),split_rule='fit Cambridge / dev Birmingham; no shared city, map, public instance or RGB SHA',
        excluded=dict(excluded),seconds=time.perf_counter()-start,teacher_logits_used=False,
        annotations='public Polygon projection weak proxies; actual visibility/identity not human verified',
        code_sha256={str(p):file_sha256(p) for p in (Path(__file__),Path('graphnav/polygon_guided_visual.py'),Path('graphnav/local_landmark_anchor.py'))})
    atomic_save(dict(frames=frames,records=records,queries=queries),a.output/'cache.pt');proof['cache_sha256']=file_sha256(a.output/'cache.pt')
    atomic_json(proof,a.output/'manifest.json');print(json.dumps(proof,indent=2),flush=True)


if __name__=='__main__':main()
