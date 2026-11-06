"""Inspect local raster georeference; never infer WGS84 from local XY."""
import argparse,json
from pathlib import Path
import rasterio
p=argparse.ArgumentParser();p.add_argument('--rgbd',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
rows=[]
for f in sorted(a.rgbd.glob('*.tif')):
 with rasterio.open(f) as r:
  rows.append(dict(map_name=f.stem,crs=str(r.crs) if r.crs else None,bounds=list(r.bounds),transform=list(r.transform),resolution=list(r.res),shape=[r.height,r.width],tags=r.tags(),geographic_alignment='unverified' if r.crs is None else 'requires_validation'))
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(maps=rows,n=len(rows),missing_crs=sum(x['crs'] is None for x in rows),note='MAP_BOUNDS alone cannot define local XY to WGS84. No GT used.'),indent=2));print(len(rows))
