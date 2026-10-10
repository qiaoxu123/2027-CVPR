"""Decoded-pixel/perceptual overlap audit; never claim GPS proof from hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import cv2
import numpy as np
from .evaluate_features import load
from graphnav.renderer import CityNavObservedViewRenderer
from experiments.bidirectional_core.cache import DEFAULT_DATA


def fingerprint(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY);small=cv2.resize(gray,(32,32)).astype(np.float32);dct=cv2.dct(small)[:8,:8].ravel();bits=dct>np.median(dct[1:]);bits[0]=False
    return dict(pixel_sha256=hashlib.sha256(str(image.shape).encode()+image.tobytes()).hexdigest(),phash=int(''.join(str(int(x)) for x in bits),2))


def overlaps(a,b):
    exact=[];near=[];lookup={r['pixel_sha256']:r for r in a}
    for right in b:
        if right['pixel_sha256'] in lookup:exact.append([lookup[right['pixel_sha256']]['id'],right['id']])
        if a:
            left=min(a,key=lambda r:(r['phash']^right['phash']).bit_count());distance=(left['phash']^right['phash']).bit_count()
            if distance<=4:near.append(dict(left=left['id'],right=right['id'],hamming=distance))
    return dict(decoded_pixel_matches=exact,phash_hamming_le4=near,near_match_is_not_geographic_proof=True)


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('refusing overlap audit overwrite')
    output.mkdir(parents=True);cv2.setNumThreads(1);started=time.time();assets=Path(args.assets);result={};aerial={}
    for split in ['train','internal_dev']:
        names=sorted({json.loads(line)['filename'] for line in (Path(args.aerial_partition)/(split+'.jsonl')).read_text().splitlines()});records=[]
        for name in names:
            image=cv2.imread(str(assets/'aerialvg/images'/name))
            if image is None:continue
            records.append(dict(id=name,**fingerprint(image)))
        aerial[split]=records
    result['aerial_train_dev']=overlaps(aerial['train'],aerial['internal_dev'])
    refseg={}
    with zipfile.ZipFile(assets/'refsegrs/RefSegRS.zip') as archive:
        for split in ['train','val']:
            names=[line.split(' ',1)[0] for line in archive.read('RefSegRS/output_phrase_'+split+'.txt').decode().splitlines() if line.strip()];records=[]
            for index,name in enumerate(names):
                image=cv2.imdecode(np.frombuffer(archive.read('RefSegRS/images/'+name+'.tif'),np.uint8),cv2.IMREAD_COLOR);records.append(dict(id=name,**fingerprint(image)))
                if index%500==0:print(json.dumps(dict(source='refseg_'+split,images=index+1,seconds=time.time()-started)),flush=True)
            refseg[split]=records
    result['refseg_official_train_val']=overlaps(refseg['train'],refseg['val'])
    renderer=CityNavObservedViewRenderer(Path(args.data_root)/'rgbd',output/'renderer_private',image_size=384);city=[];seen_ids=set()
    for feature_path in args.city_features:
        for row in load(feature_path):
            m=row['metadata'];identity=(m['episode_id'],m['cutoff_step'])
            if identity in seen_ids:continue
            seen_ids.add(identity);observation=renderer.current_rgbd(m['map_name'],np.asarray(m['actual_camera_pose'],np.float64),float(row['public']['ground_level']))
            city.append(dict(id=m['sample_id'],**fingerprint(observation['bgr'])))
    result['aerial_vs_city_train_dev_observed']=overlaps(city,aerial['train']+aerial['internal_dev'])
    result['refseg_vs_city_train_dev_observed']=overlaps(city,refseg['train']+refseg['val'])
    result.update(counts=dict(aerial_train=len(aerial['train']),aerial_dev=len(aerial['internal_dev']),refseg_train=len(refseg['train']),refseg_val=len(refseg['val']),city_train_dev_observed=len(city)),seconds=time.time()-started,city_seen_or_unseen_read=False,
        scope='decoded images and global pHash in this actual pilot; crop/reprojection/nonidentical overlap can evade this check',geographic_policy='AerialVG official VisDrone source is Chinese cities, CityNav maps UK; RefSegRS SkyScapes selected tiles lack GPS/country mapping, so geographic overlap with UK unresolved and RefSegRS is evaluation-only, not pretraining. DIOR worldwide without GPS is not trained. Foundation pretraining footprint unknown.')
    (output/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--assets',required=True);p.add_argument('--aerial-partition',required=True);p.add_argument('--city-features',action='append',required=True);p.add_argument('--output',required=True);p.add_argument('--data-root',default=DEFAULT_DATA);run(p.parse_args())


if __name__=='__main__':main()
