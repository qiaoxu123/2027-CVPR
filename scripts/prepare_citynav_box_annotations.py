"""TRAIN-only auto box PROPOSALS + editable human review page. No model fitting."""
import argparse,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path
import cv2,numpy as np,torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.landmark_projection import frame_to_world
from graphnav.map_guided_grounding import project_polygon,public_landmarks,reference_candidates
from scripts.train_object_query_belief import load


def train_sources(cache):
    manifest=json.loads((Path(cache)/'manifest.json').read_text())
    if manifest.get('partition')!='train' or not manifest.get('complete'):raise ValueError('Only completed TRAIN cache allowed')
    lookup={}
    for source in manifest['sources']:
        root=Path(source['path']);path=root/'manifest.json';m=json.loads(path.read_text())
        if m['partition']!='train' or m['status']!='complete' or file_sha256(path)!=source['manifest_sha']:raise ValueError('TRAIN source mismatch')
        for entry in m['frames']:lookup.setdefault((entry['episode'],entry['step']),root/(entry['key']+'.pt'))
    return manifest,lookup


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['cache','catalog','annotations','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--review-size',type=int,default=64)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('No overwrite')
    if Path(a.annotations).name!='citynav_train_seen.json':raise ValueError('TRAIN annotations only')
    manifest,lookup=train_sources(a.cache);_,rows=load(a.cache)
    objects=json.loads(Path(a.catalog).read_text());source=json.loads(Path(a.annotations).read_text())
    named={m:public_landmarks(objects,m) for m in {r['map_name'] for r in rows}}
    relation_cache={};data=[];counts=Counter();out.mkdir(parents=True)
    torch.set_num_threads(4)
    for number,row in enumerate(rows):
        path=lookup[(row['episode'],row['step'])];frame=torch.load(path,weights_only=True)
        image=path.with_suffix('.png');bgr=cv2.imread(str(image))
        if bgr is None or bgr.shape[:2]!=tuple(frame['depth_valid'].shape):raise ValueError('RGB/depth alignment error')
        ann=source[row['episode']]
        if ann['descriptions'][0]!=row['instruction']:raise ValueError('TRAIN episode/instruction mismatch')
        object_id=str(ann['object_ids'][0]);obj=objects[row['map_name']][object_id]
        if np.linalg.norm(np.array(obj['position'])[:2]-row['labels']['target_world'][:2].numpy())>1e-3:raise ValueError('Instance/location identity mismatch')
        footprint=frame['footprint_world'].numpy();depth=frame['depth_valid'].numpy()
        projection=project_polygon(obj['contour'],footprint,bgr.shape,depth)
        bbox=projection['bbox'];size=[bbox[2]-bbox[0],bbox[3]-bbox[1]] if bbox else [0,0]
        kind=obj['object_type'];minimum=64 if kind=='Car' else 256
        high=bool(projection['valid_roi'] and projection['coverage']>=.95 and projection['depth_support']>=.9 and projection['pixels']>=minimum)
        polygon=np.asarray(obj['contour'],np.float32)
        mapping_error=None
        if bbox:
            center=np.array([[(bbox[0]+bbox[2])/2,(bbox[1]+bbox[3])/2]],np.float32)
            mapping_error=float(np.linalg.norm(frame_to_world(center,footprint,bgr.shape)[0]-np.array(obj['position'])[:2]))
        key=(row['map_name'],row['instruction']);landmarks=named[row['map_name']]
        if key not in relation_cache:
            _,relations,mentioned,_=reference_candidates(landmarks,[dict(valid_roi=False) for _ in landmarks],row['instruction'])
            relation_cache[key]=(relations,mentioned)
        relations,mentioned=relation_cache[key]
        references=[]
        for i in mentioned:
            e=landmarks[i];q=project_polygon(e['geometry_m'],footprint,bgr.shape,depth)
            references.append(dict(public_id=e['key'],name=e['name'],category=e['kind'],world_polygon=e['geometry_m'].tolist(),projection=q))
        identity=f"train:{row['episode']}:{row['step']}"
        record=dict(sample_id=identity,partition='train',episode=row['episode'],step=row['step'],instruction=row['instruction'],instruction_group=row['group'],
            instance_group=row['map_name']+':'+object_id,map=row['map_name'],instance_id=object_id,category=kind,
            image_path=str(image),image_sha=file_sha256(image),raw_frame_path=str(path),image_shape=list(bgr.shape),
            original_world_polygon=polygon.tolist(),world_position=obj['position'],dimension=obj.get('dimension'),
            pose=frame['pose'].tolist(),footprint_world=footprint.tolist(),projection=projection,auto_bbox_xyxy=bbox,
            geometric_visibility_hint=projection['status'],semantic_visibility='unknown_NOT_HUMAN_VERIFIED',
            quality=dict(high_geometric_support=high,minimum_pixels_for_class=minimum,bbox_size=size,
                         tiny=min(size)<8 if bbox else True,clipped=projection['status']=='partial',depth_support=projection['depth_support'],
                         roof_offset_unmodeled=kind=='Building',height_proxy_m=obj.get('dimension',[0,0,0])[2],
                         top_height_proxy_above_camera=bool(float(obj['position'][2])+float(obj.get('dimension',[0,0,0])[2])/2>=float(frame['pose'][2])),
                         box_center_vs_instance_position_m=mapping_error,physical_occlusion='unknown'),
            references=references,relations=[dict(kind=r.kind,anchors=[[landmarks[i]['key'] for i in group] for group in r.anchors]) for r in relations],
            annotation_status='UNREVIEWED_AUTO_PROPOSAL_NOT_PIXEL_GT')
        data.append(record);counts[kind]+=1
        if (number+1)%100==0:print(json.dumps(dict(annotations=number+1)),flush=True)
    with (out/'annotations.jsonl').open('w') as f:
        for r in data:f.write(json.dumps(r)+'\n')
    reviews=[dict(sample_id=r['sample_id'],image_sha=r['image_sha'],status='unreviewed',source=None,reviewer='',human_attested=False,visibility='unknown',bbox=None,notes='') for r in data]
    atomic_json(dict(format='citynav-box-review-v1',reviews=reviews),out/'reviews.unreviewed.json')
    # Round robin over category/geometry/tiny, one physical instance per first batch.
    bins=defaultdict(list)
    for r in data:bins[(r['category'],r['geometric_visibility_hint'],r['quality']['tiny'])].append(r)
    for items in bins.values():items.sort(key=lambda r:hashlib.sha256(r['sample_id'].encode()).hexdigest())
    chosen=[];seen=set()
    while len(chosen)<a.review_size and any(bins.values()):
        for key in sorted(bins):
            while bins[key]:
                r=bins[key].pop(0)
                if r['instance_group'] in seen:continue
                chosen.append(r);seen.add(r['instance_group']);break
            if len(chosen)>=a.review_size:break
    gallery=[];(out/'images').mkdir()
    for r in chosen:
        name=r['sample_id'].replace(':','_')+'.png';shutil.copyfile(r['image_path'],out/'images'/name)
        frame=torch.load(r['raw_frame_path'],weights_only=True)
        features=frame['all_query_features'].float();features[frame['query_ids']]=frame['query_features'].float()
        text=frame['encoded_text'][0].float();mask=frame['text_token_mask'][0].bool()
        scores=(features@text.T)[:,mask].sigmoid().amax(-1)
        gallery.append(dict(r,image='images/'+name,raw_boxes=frame['all_boxes_cxcywh'].tolist(),raw_scores=scores.tolist(),raw_order=scores.argsort(descending=True).tolist()))
    atomic_json(gallery,out/'gallery_data.json')
    from scripts.citynav_box_review_html import write_review_html
    write_review_html(gallery,out/'review.html')
    summary=dict(states=len(data),episodes=len({r['episode'] for r in data}),instructions=len({r['instruction_group'] for r in data}),instances=len({r['instance_group'] for r in data}),
        category_states=dict(counts),geometric_status=dict(Counter(r['geometric_visibility_hint'] for r in data)),
        high_geometric_support=sum(r['quality']['high_geometric_support'] for r in data),human_reviewed=0,human_pass_rate=None,human_category_IoU=None,
        high_support_by_category=dict(Counter(r['category'] for r in data if r['quality']['high_geometric_support'])),
        representative_review_states=len(gallery),representative_instances=len(seen),
        source_hashes=dict(cache_manifest=file_sha256(Path(a.cache)/'manifest.json'),catalog=file_sha256(a.catalog),TRAIN_annotations=file_sha256(a.annotations)),
        visibility_warning='All semantic visibility unknown until human review; projection labels are not visible-instance truth',complete=True)
    atomic_json(summary,out/'summary.json');print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
