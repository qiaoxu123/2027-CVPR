"""Header-only source resolution audit on explicitly partitioned TRAIN observations."""
import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
from PIL import Image
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('observations','filter','rgbd','feature-manifest','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve raster audit')
    allowed=set(json.loads(a.filter.read_text())['frame_keys'])
    partition=json.loads(a.feature_manifest.read_text())['map_assignment']
    frames=[r for r in map(json.loads,(a.observations/'public_frames.jsonl').read_text().splitlines()) if r['frame_key'] in allowed and partition[r['map_name']]=='fit']
    maps={};rows=[]
    for name in sorted({r['map_name'] for r in frames}):
        with rasterio.open(a.rgbd/(name+'.tif')) as source:
            png_size=Image.open(a.rgbd/(name+'.png')).size
            maps[name]=dict(height_size=[source.width,source.height],rgb_size=list(png_size),same_pixel_dimensions=png_size==(source.width,source.height),
                ground_sample_distance_m=[float(np.hypot(source.transform.a,source.transform.d)),float(np.hypot(source.transform.b,source.transform.e))])
            if not maps[name]['same_pixel_dimensions']:raise ValueError('RGB/height source pixel grids differ: '+name)
            for frame in frames:
                if frame['map_name']!=name:continue
                corners=np.asarray([[source.index(float(x),float(y))[1],source.index(float(x),float(y))[0]] for x,y in frame['footprint_world']],float)
                lengths=np.linalg.norm(np.roll(corners,-1,axis=0)-corners,axis=-1)
                rows.append(dict(frame_key=frame['frame_key'],map_name=name,source_pixels_per_view_edge=float(np.mean(lengths)),
                    source_to_384_ratio=float(np.mean(lengths)/383),current_rgb_size=[384,384]))
    ratios=np.array([r['source_to_384_ratio'] for r in rows])
    summary=dict(frames=len(rows),maps=len(maps),subset='fit',feature_manifest_sha256=file_sha256(a.feature_manifest),source_rgb_height_grids_match=True,
        source_to_384_edge_ratio_percentiles=dict(zip(['min','p25','median','p75','max'],np.quantile(ratios,[0,.25,.5,.75,1]).tolist())),
        more_than_384_source_samples=int((ratios>1).sum()),more_than_768_source_samples=int((ratios>2).sum()),
        warning='Geometric raster sampling ratio, not observed object recognizability or visual performance. Out-of-raster portions contain no extra information.')
    a.output.mkdir(parents=True);(a.output/'results.json').write_text(json.dumps(dict(summary=summary,maps=maps,rows=rows,source_sha256=file_sha256(__file__)),indent=2));print(json.dumps(summary))


if __name__=='__main__':main()
