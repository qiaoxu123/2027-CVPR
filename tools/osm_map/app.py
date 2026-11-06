"""Read-only local-XY map viewer; no model inference or target-driven retrieval."""
import argparse,json,io
from pathlib import Path
import numpy as np,cv2,rasterio
from fastapi import FastAPI,HTTPException
from fastapi.responses import FileResponse,Response
from functools import lru_cache

def create_app(data,overlays):
 data=Path(data); overlays=Path(overlays); app=FastAPI(); objects=json.loads((data/'cityrefer_original/objects.json').read_text())
 maps={p.stem:p for p in (data/'rgbd').glob('*.tif')}
 @app.get('/')
 def index():return FileResponse(Path(__file__).parent/'web/index.html')
 @app.get('/api/maps')
 def listing():return sorted(maps)
 @app.get('/api/scene/{name}')
 def scene(name:str):
  if name not in maps:raise HTTPException(404)
  with rasterio.open(maps[name]) as r:meta=dict(bounds=list(r.bounds),affine=list(r.transform),shape=[r.height,r.width],crs=str(r.crs) if r.crs else None)
  p=overlays/(name+'.json'); overlay=json.loads(p.read_text()) if p.exists() else None
  return dict(name=name,**meta,georegistration='地理配准未验证',osm_enabled=False,objects=[dict(id=str(k),name=v.get('name',''),type=v.get('object_type'),polygon=v.get('contour',[])) for k,v in objects.get(name,{}).items() if v.get('name')],overlay=overlay)
 @lru_cache(maxsize=6)
 def image(name):
  im=cv2.imread(str(data/'rgbd'/(name+'.png')));h,w=im.shape[:2];scale=min(1,2048/max(h,w));im=cv2.resize(im,(round(w*scale),round(h*scale)),interpolation=cv2.INTER_AREA);return cv2.imencode('.png',im)[1].tobytes()
 @app.get('/raster/{name}')
 def raster(name:str):
  if name not in maps:raise HTTPException(404)
  return Response(image(name),media_type='image/png')
 return app
if __name__=='__main__':
 import uvicorn
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--overlays',required=True);p.add_argument('--port',type=int,default=8792);a=p.parse_args();uvicorn.run(create_app(a.data,a.overlays),host='0.0.0.0',port=a.port)
