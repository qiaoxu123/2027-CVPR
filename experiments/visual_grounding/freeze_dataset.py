"""Freeze an identical, object-balanced image/text evaluation index for all models."""
import argparse, hashlib, json
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import cv2
from types import SimpleNamespace
from graphnav.landmark_projection import make_roi_image


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):
    return [json.loads(l) for l in Path(path).open() if l.strip()]


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    data=Path(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'frozen.json').exists():raise ValueError('already frozen; refusing overwrite')
    regions=read(data/'regions.jsonl');queries=read(data/'candidates.jsonl')
    byid={r['sample_id']:r for r in regions};groups=defaultdict(list)
    for r in regions:
        assert r['observed_step']<=r['observation_cutoff_step']
        if not Path(r['roi_source']).is_file():raise ValueError('missing real observed pixels')
        groups[r['split'],r['map_name'],r['object_id']].append(r)
    ids=set(s for q in queries for s in q['candidate_sample_ids']);retrieval=[];attributes=[]
    # Independent objects count once for attribute classification; two views for retrieval.
    for key,rows in sorted(groups.items()):
        rows=sorted(rows,key=lambda r:(-min(r['roi_pixel_size']),-r['roi_visibility'],r['sample_id']))
        qualified=[r for r in rows if r['roi_quality']['attribute_resolution_eligible'] and r['attributes']]
        if qualified:
            r=qualified[0];ids.add(r['sample_id'])
            for kind,v in sorted(r['attributes'].items()):
                attributes.append(dict(sample_id=r['sample_id'],split=r['split'],map_name=r['map_name'],object_id=r['object_id'],object_type=r['object_type'],attribute_type=kind,label=v['label'],label_status='language_weak_not_visual_verified'))
        distinct={}
        for r in rows:
            fp=tuple(np.round(np.asarray(r['footprint_world']).ravel(),3));distinct.setdefault(fp,r)
        if len(distinct)>=2:
            first,second=list(distinct.values())[:2]
            ids.update([first['sample_id'],second['sample_id']])
            retrieval.append(dict(split=key[0],map_name=key[1],object_id=key[2],object_type=first['object_type'],query_sample_id=first['sample_id'],gallery_sample_id=second['sample_id']))
    selected=[byid[s] for s in sorted(ids)]
    # A manual review file may verify a specific attribute/region. Never infer verification.
    verification_path=data/'verified_attributes.json'
    verified=json.loads(verification_path.read_text()) if verification_path.exists() else []
    for label in verified:
        if label['sample_id'] not in byid:raise ValueError('unknown verified image')
        r=byid[label['sample_id']]
        if r['split']=='val_unseen':raise ValueError('manual dev inspection of Unseen forbidden')
        ids.add(label['sample_id'])
        attributes.append(dict(label,split=r['split'],map_name=r['map_name'],object_id=r['object_id'],object_type=r['object_type'],label_status='agent_image_review_exploratory'))
    selected=[dict(byid[s]) for s in sorted(ids)]
    (out/'rois').mkdir(exist_ok=True)
    for r in selected:
        if r.get('roi_path') and Path(r['roi_path']).is_file():continue
        image=cv2.imread(r['roi_source']);bbox=r['roi_bbox']
        if image is None:raise FileNotFoundError(r['roi_source'])
        path=out/'rois'/f"{r['sample_id']}.png"
        roi=SimpleNamespace(bbox_xyxy=tuple(bbox))
        size=min(384,max(32,int(max(bbox[2]-bbox[0],bbox[3]-bbox[1])*1.5)))
        if not cv2.imwrite(str(path),make_roi_image(image,roi,output_size=size),[cv2.IMWRITE_PNG_COMPRESSION,1]):raise IOError(path)
        r['roi_path']=str(path.resolve())
    source_hashes={}
    for r in selected:
        r['roi_sha256']=sha(r['roi_path'])
        if r['roi_source'] not in source_hashes:source_hashes[r['roi_source']]=sha(r['roi_source'])
        r['source_frame_sha256']=source_hashes[r['roi_source']]
    for name,rows in [('index',selected),('queries',queries),('retrieval',retrieval),('attributes',attributes)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    payload=dict(source_data=str(data.resolve()),source_hashes={n:sha(data/n) for n in ['regions.jsonl','manifest.jsonl','candidates.jsonl']},files={n:sha(out/f'{n}.jsonl') for n in ['index','queries','retrieval','attributes']},protocol_sha256=sha('experiments/visual_grounding/protocol.json'),images=len(selected),queries=len(queries),retrieval_instances=len(retrieval),attributes=len(attributes),by_split=dict(Counter(r['split'] for r in selected)),selection='all causal attribute candidate ROIs; highest-resolution single view per labeled object; two distinct footprints per identity; no model scores')
    (out/'frozen.json').write_text(json.dumps(payload,indent=2));print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
