"""Add private verified negative labels to immutable P1 pairs; no new split."""
import argparse
import json
from pathlib import Path
import numpy as np
from graphnav.bidirectional import source_landmark, verified_hard_negatives
from graphnav.checkpointing import file_sha256, provenance


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('pairs','objects','output'):p.add_argument('--'+key,required=True)
    a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('refusing pair overwrite')
    from shapely.geometry import Polygon
    objects=json.loads(Path(a.objects).read_text());maps={};negative_cache={};count=covered=total=0
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w') as handle:
        for line in Path(a.pairs).read_text().splitlines():
            row=json.loads(line);name=row['source_map_name'];bounds=np.asarray(row['world_bounds'])
            if name not in maps:
                maps[name]={}
                for identity,entry in objects[name].items():
                    lm=source_landmark(identity,entry,bounds)
                    if lm is not None:
                        lm['polygon_area']=Polygon(lm['contour']).area
                        maps[name][str(identity)]=lm
            entries=maps[name];anchor=entries[row['source_anchor_id']];target=entries[row['hidden_target_id']]
            key=(name,row['source_anchor_id'],row['hidden_target_id'],row['relation_direction'])
            if key not in negative_cache:
                negative_cache[key]=verified_hard_negatives(anchor,target,list(entries.values()),row['relation_direction'])
            negative=negative_cache[key]
            shift=np.asarray(row['anchor_center_xy'])-np.asarray(anchor['center_norm'])
            negative=[(np.asarray(n)+shift).tolist() for n in negative if np.all(np.asarray(n)+shift>=.02) and np.all(np.asarray(n)+shift<=.98)]
            row['hard_negative_xy']=negative
            row['hard_negative_protocol']='same source map/type; range .6..1.4 and area .5..2; opposite directional side >=5m; positive separation >40m; loss only'
            handle.write(json.dumps(row)+'\n');count+=1;covered+=bool(negative);total+=len(negative)
    report=dict(samples=count,covered_samples=covered,negative_labels=total,source_sha256={'pairs':file_sha256(a.pairs),'objects':file_sha256(a.objects)},output_sha256=file_sha256(out),provenance=provenance(),note='Same P1 v2 rows, order, source families and augmented splits; only private loss labels added.')
    Path(str(out)+'.manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
