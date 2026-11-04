"""Audit names, real image hashes and same-city polygon overlap across folds."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from graphnav.landmark_alias import canonical_name
from graphnav.landmark_region_supervision import named_union


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    rows=json.loads((a.data/'samples_public.json').read_text());queries=json.loads((a.data/'queries_public.json').read_text())
    qsets=defaultdict(set);images=defaultdict(set);maps=defaultdict(set)
    for r in rows:
        if not r['train_region_eligible']:continue
        part=r['subset'];qsets[part].add(r['query_key']);images[part].add(r['frame']['image_sha256']);maps[part].add(r['map_name'])
    results={}
    for left,right in [('fit','dev'),('fit','confirm'),('dev','confirm')]:
        shared=[]
        for lk in sorted(qsets[left]):
            l=queries[lk];city=l['map_name'].split('_block_')[0]
            for rk in sorted(qsets[right]):
                r=queries[rk]
                if r['map_name'].split('_block_')[0]!=city or canonical_name(l['name'])!=canonical_name(r['name']):continue
                p1=named_union(l['public_polygons']);p2=named_union(r['public_polygons'])
                shared.append(dict(left=lk,right=rk,intersection_area_m2=float(p1.intersection(p2).area),distance_m=float(p1.distance(p2))))
        results[left+'_'+right]=dict(shared_map_names=sorted(maps[left]&maps[right]),same_map_name_groups=sorted(qsets[left]&qsets[right]),
            exact_eligible_image_hash_overlap=len(images[left]&images[right]),same_city_name_pairs=shared,
            positive_polygon_overlap_pairs=sum(x['intersection_area_m2']>1e-6 for x in shared))
    a.output.write_text(json.dumps(dict(results=results,caution='Map separation alone is not guaranteed physical-road-entity separation. Different cities use separate coordinate frames and are not geometrically compared.'),indent=2))
    print(json.dumps({k:{n:v for n,v in row.items() if n!='same_city_name_pairs'} for k,row in results.items()},indent=2))


if __name__=='__main__':main()
