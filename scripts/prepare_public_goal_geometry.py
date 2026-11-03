"""Compile all public named polygons for fixed world hypotheses, without GT."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from shapely.geometry import Polygon
from graphnav.candidate_geometry import relative_entity_features
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.scoped_between_geometry import between_distance
from scripts.analyze_instruction_landmark_association import PublicNames


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,action='append',required=True);p.add_argument('--observations',type=Path,action='append',required=True)
    for k in ('catalog','corpus','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    if len(a.cache)!=len(a.observations):raise ValueError('paired public sources required')
    objects=json.loads(a.catalog.read_text());names=PublicNames(objects)
    corpus={(r['map_name'],r['instruction']):r for r in json.loads(a.corpus.read_text())}
    output={}
    for cache,observations in zip(a.cache,a.observations):
        meta=json.loads((cache/'manifest.json').read_text());assert meta['complete'] and meta['split']=='train_seen'
        assert meta['public_catalog_sha256']==file_sha256(a.catalog)
        frames={r['frame_key']:r for r in map(json.loads,(observations/'public_frames.jsonl').read_text().splitlines())}
        for chunk in meta['chunks']:
            path=cache/chunk['file'];assert file_sha256(path)==chunk['sha256']
            for r in torch.load(path,weights_only=True)['records']:
                if r['subset'] not in ('fit','dev'):continue
                assert not {'target_world','target_id','physical_group'}&r.keys()
                frame=frames[r['frame_key']];mp=frame['map_name'];text=frame['instruction'];xy=r['candidate_world'].numpy()
                ids=sorted({i for m in names.match(mp,text) for i in m['public_ids']});entities=[]
                for i in ids:
                    contour=np.asarray(objects[mp][i]['contour'],float)
                    if contour.ndim!=2 or contour.shape[-1]!=2 or len(contour)<3 or not np.isfinite(contour).all():continue
                    poly=Polygon(contour)
                    if poly.is_valid and poly.area>0:entities.append(dict(kind='region',geometry_m=contour))
                geom=np.zeros((len(xy),33),np.float32)
                if entities:
                    pair=relative_entity_features(xy,entities,len(entities))
                    geom[:,:31]=np.concatenate((pair.mean(1),pair.min(1),pair.max(1),np.full((len(xy),1),len(entities)/16)),axis=-1)
                costs=[]
                for g in corpus.get((mp,text),{}).get('groups',[]):
                    if g['kind']!='two-public-anchors':continue
                    distance,known=between_distance(xy,g['endpoints'],objects[mp])
                    if known:costs.append(distance)
                if costs:geom[:,-2]=np.mean(costs,axis=0)/100.;geom[:,-1]=1
                if r['frame_key'] in output:raise ValueError('duplicate frame key across caches')
                output[r['frame_key']]=torch.from_numpy(geom).half()
    a.output.mkdir(parents=True);atomic_save(output,a.output/'geometry_public.pt')
    atomic_json(dict(frames=len(output),private_target_labels_read=False,normal_inference_modified=False,
        geometry_sha256=file_sha256(a.output/'geometry_public.pt'),catalog_sha256=file_sha256(a.catalog),
        corpus_sha256=file_sha256(a.corpus),producer_sha256=file_sha256(__file__),
        source_manifests=[file_sha256(c/'manifest.json') for c in a.cache]),a.output/'manifest.json')

if __name__=='__main__':main()
