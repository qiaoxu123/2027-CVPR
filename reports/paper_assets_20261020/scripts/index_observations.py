"""Index actual RGB sources/crops; publish metadata, never raw raster pixels."""
from common import *
import numpy as np
import rasterio

def main():
    cases=list(jsonl(OUT/'cases.jsonl'));maps=sorted({r['map_name'] for r in cases});assets=[];frames=[]
    for name in maps:
        for suffix in ('.png','.tif'):
            p=WORKSPACE/'DATA/rgbd'/f'{name}{suffix}'
            row=dict(map_name=name,path=str(p),bytes=p.stat().st_size,sha256=sha(p),redistribution='Not included; underlying raster terms not established by the code MIT license')
            if suffix=='.tif':
                with rasterio.open(p) as ds:row.update(shape=[ds.height,ds.width],affine=list(ds.transform),crs=str(ds.crs),nodata=ds.nodata)
            assets.append(row)
    for case in cases:
        m=read(OUT/'fields'/f"{case['case_id']}.json")
        for frame in m['frames']:
            path=OUT/'local_only/observations'/f"{case['case_id']}_step{frame['step']}.npz"
            with np.load(path) as a:shape=list(a['bgr'].shape)
            frames.append(dict(case_id=case['case_id'],episode_id=case['episode_id'],split=case['split'],map_name=case['map_name'],**frame,local_crop=str(path),crop_sha256=sha(path),shape=shape,
                renderer=str(ARCHIVE/'graphnav/renderer.py'),renderer_sha256=sha(ARCHIVE/'graphnav/renderer.py'),source_rgb=str(WORKSPACE/'DATA/rgbd'/f"{case['map_name']}.png"),
                command='python reports/paper_assets_20261020/scripts/replay_cases.py',public=False))
    dump(OUT/'observation_index.json',dict(source_assets=assets,frames=frames))
    source=WORKSPACE/'DATA/hett/release/refined_citynav/cityrefer/landmarks.json';allmaps=read(source)
    dump(OUT/'public_map_context.json',dict(source=str(source),source_sha256=sha(source),maps={k:list(allmaps[k].values()) for k in maps},
        note='All named public map landmarks are faint display context, not additional inputs retroactively supplied to the model. Referenced input polygons remain highlighted. No target-object catalog or raster is used as background.',attribution='CityNav / OpenStreetMap contributors, https://www.openstreetmap.org/copyright'))
    print('source rasters',len(assets),'observed frames',len(frames))

if __name__=='__main__':main()
