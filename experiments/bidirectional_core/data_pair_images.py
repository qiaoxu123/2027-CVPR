"""Materialize actual paired nadir ROIs and index exact prior SigLIP2 features."""
import argparse,json,hashlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import cv2
from graphnav.landmark_projection import make_roi_image

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main(args):
    cv2.setNumThreads(1);root=Path(args.data);benchmark=Path(args.benchmark);stats={}
    for split,cachepart in [('train_seen','train'),('val_seen','seen')]:
        frozen=benchmark/'data_audit'/('frozen_train' if cachepart=='train' else 'frozen_seen');index={}
        for i,line in enumerate((frozen/'index.jsonl').open()):
            r=json.loads(line);index[r['sample_id']]=(i,r)
        out=root/split/'paired_rois';out.mkdir(exist_ok=True);rows=[];counts=Counter();sourcehash={}
        for line in (root/split/'visual_pairs.jsonl').open():
            r=json.loads(line);sid=r['roi_sample_id'];prior=index.get(sid)
            if prior is not None:
                offset,entry=prior
                # Exact cache reuse requires both current source and saved crop
                # hashes. Never match feature caches merely by object identity.
                path=Path(entry['roi_path']);assert path.is_file() and sha(path)==entry['roi_sha256']
                source=r['roi_source'];sourcehash.setdefault(source,sha(source));assert sourcehash[source]==entry['source_frame_sha256']
                r['roi_path']=str(path.resolve());r['siglip2_cache']={'path':str((benchmark/'siglip2_large384'/cachepart/'embeddings.pt').resolve()),'images_row':offset,'frozen_sha256':sha(frozen/'frozen.json'),'exact_source_pixels':True};counts['exact_siglip2_cache_hits']+=1
            else:
                path=Path(r['roi_path']) if r.get('roi_path') else out/f'{sid}.png'
                if not path.is_file():
                    image=cv2.imread(r['roi_source']);assert image is not None
                    bbox=r['roi_bbox'];size=min(384,max(32,int(max(bbox[2]-bbox[0],bbox[3]-bbox[1])*1.5)))
                    roi=make_roi_image(image,SimpleNamespace(bbox_xyxy=tuple(bbox)),output_size=size)
                    assert cv2.imwrite(str(path),roi,[cv2.IMWRITE_PNG_COMPRESSION,1]);counts['crops_materialized']+=1
                r['roi_path']=str(path.resolve());r['siglip2_cache']=None
            r['roi_sha256']=sha(r['roi_path']);r['source_frame_sha256']=sourcehash.setdefault(r['roi_source'],sha(r['roi_source']));counts['pairs']+=1;rows.append(r)
        (root/split/'visual_pairs_materialized.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));stats[split]=dict(counts)
    (root/'paired_roi_manifest.json').write_text(json.dumps(dict(stats=stats,embedding_cache_keys='exact source hash + crop hash + frozen ordering',new_visual_encoding=False),indent=2));print(json.dumps(stats),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',default='runs/bidirectional_core_test/data');p.add_argument('--benchmark',default='runs/visual_grounding_benchmark');main(p.parse_args())
