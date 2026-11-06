"""Authenticated manual registration UI; persistent city-level controls, no model changes."""
import argparse,os,json,secrets,time,fcntl,tempfile,hashlib
from pathlib import Path
from functools import lru_cache
from urllib.parse import urlparse
import requests,numpy as np
from fastapi import Request,HTTPException
from fastapi.responses import FileResponse,Response,JSONResponse
from fastapi.staticfiles import StaticFiles
from rasterio.warp import transform
from tools.osm_map.app import create_app
from tools.osm_map.registration import fit_controls

def build(data,overlays,store,user,password):
    if not user or len(password)<16:raise ValueError('Set MAP_REVIEW_USER/PASSWORD (>=16 chars)')
    app=create_app(data,overlays);store=Path(store);store.mkdir(parents=True,exist_ok=True);os.chmod(store,0o700)
    import base64
    token=base64.b64encode((user+':'+password).encode()).decode()
    @app.middleware('http')
    async def auth(request:Request,call_next):
        if not secrets.compare_digest(request.headers.get('authorization',''),'Basic '+token):return Response(status_code=401,headers={'WWW-Authenticate':'Basic realm="CityNav Control Points"'})
        if request.method=='POST' and request.headers.get('x-control-write')!='1':return Response('Missing write confirmation header',status_code=403)
        response=await call_next(request)
        if not request.url.path.startswith('/osm/'):response.headers['Cache-Control']='no-store'
        response.headers['X-Content-Type-Options']='nosniff';return response
    app.mount('/vendor',StaticFiles(directory=Path(__file__).parent/'web/vendor'),name='vendor')
    @app.get('/control')
    def page():return FileResponse(Path(__file__).parent/'web/control.html')
    def path(city):
        if city not in ['cambridge','birmingham']:raise HTTPException(404)
        return store/(city+'.json')
    def read(city):
        p=path(city);return json.loads(p.read_text()) if p.exists() else dict(city=city,revision=0,points=[],registration=None)
    def validate(city,points):
        if len(points)>500:raise HTTPException(400,'Too many points')
        from math import isfinite
        ids=[]
        for p in points:
            if not isinstance(p.get('id'),str) or len(p['id'])>100:raise HTTPException(400,'Invalid point id')
            ids.append(p['id'])
            if p.get('split') not in ['fit','validation'] or p.get('source') not in ['public_building_corner','public_road_intersection','other_public_feature']:raise HTTPException(400,'Invalid type')
            if not str(p.get('map_name','')).startswith(city+'_block_'):raise HTTPException(400,'Wrong city')
            if any(not isinstance(p.get(k),(int,float)) or not isfinite(p[k]) for k in ['x','y','lon','lat']):raise HTTPException(400,'Invalid coordinate')
            if not (-180<=p['lon']<=180 and -90<=p['lat']<=90):raise HTTPException(400,'Invalid lon/lat')
        if len(set(ids))!=len(ids):raise HTTPException(400,'Duplicate IDs')
    def atomic(p,obj):
        fd,tmp=tempfile.mkstemp(dir=store,prefix='.write-');
        try:
            with os.fdopen(fd,'w') as f:json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
            os.replace(tmp,p)
        finally:
            if os.path.exists(tmp):os.unlink(tmp)
    @app.get('/native-raster/{name}')
    def native(name:str):
        if '/' in name or not name.startswith(('cambridge_block_','birmingham_block_')):raise HTTPException(404)
        path=Path(data)/'rgbd'/(name+'.png')
        if not path.is_file():raise HTTPException(404)
        return FileResponse(path,media_type='image/png')
    @app.get('/api/controls/{city}')
    def get(city:str):return read(city)
    @app.post('/api/controls/{city}')
    async def save(city:str,request:Request):
        payload=await request.json();validate(city,payload.get('points',[]))
        with (store/(city+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);old=read(city)
            if payload.get('revision')!=old['revision']:raise HTTPException(409,'其他窗口已更新；请先重新加载')
            mode=payload.get('mode','similarity')
            if mode not in ['similarity','affine']:raise HTTPException(400,'Invalid mode')
            try:reg=fit_controls(payload['points'],mode)
            except ValueError as e:reg=dict(verified=False,pending_reason=str(e))
            obj=dict(city=city,revision=old['revision']+1,points=payload['points'],mode=mode,registration=reg,updated_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            if path(city).exists():atomic(store/(city+f'.history-{old["revision"]}.json'),old)
            atomic(path(city),obj);return obj
    @app.get('/api/controls/{city}/export')
    def export(city:str):return JSONResponse(read(city),headers={'Content-Disposition':f'attachment; filename="{city}_controls.json"'})
    @app.get('/api/project-corners/{city}/{name}')
    def corners(city:str,name:str):
        import rasterio
        reg=read(city).get('registration') or {}
        if 'matrix' not in reg:raise HTTPException(400,'尚无变换')
        if not name.startswith(city+'_block_') or '/' in name:raise HTTPException(400)
        p=Path(data)/'rgbd'/(name+'.tif')
        if not p.exists():raise HTTPException(404)
        with rasterio.open(p) as r:b=r.bounds
        xy=np.array([[b.left,b.top],[b.right,b.top],[b.right,b.bottom],[b.left,b.bottom]]);M=np.array(reg['matrix']);en=xy@M[:,:2].T+M[:,2];lon,lat=transform('EPSG:27700','EPSG:4326',en[:,0],en[:,1]);return dict(corners=list(zip(lat,lon)),verified=reg.get('verified',False))
    session=requests.Session();proxy=os.environ.get('OSM_HTTP_PROXY');session.proxies={'http':proxy,'https':proxy} if proxy else {};session.headers['User-Agent']='CityNavManualControlPoint/1.0 (+https://github.com/qiaoxu123/2027-CVPR)'
    session.headers['Referer']=os.environ.get('CONTROL_PUBLIC_URL','http://101.43.173.9:42024/')
    @lru_cache(maxsize=512)
    def tile(z,x,y):
        r=session.get(f'https://tile.openstreetmap.org/{z}/{x}/{y}.png',timeout=20);r.raise_for_status()
        if r.headers.get('x-blocked') or not r.headers.get('content-type','').startswith('image/'):raise requests.RequestException('Blocked or non-image tile')
        return r.content
    @app.get('/osm/{z}/{x}/{y}.png')
    def osm(z:int,x:int,y:int):
        if not 0<=z<=20 or not 0<=x<2**z or not 0<=y<2**z:raise HTTPException(400)
        try:return Response(tile(z,x,y),media_type='image/png',headers={'Cache-Control':'private, max-age=86400'})
        except requests.RequestException:raise HTTPException(502,'OSM瓦片暂不可用')
    last_search=[0.]
    @app.get('/api/search')
    def search(q:str):
        if not 3<=len(q)<=200:raise HTTPException(400)
        # Explicit user search only; no autocomplete or bulk geocoding.
        if time.monotonic()-last_search[0]<1.1:raise HTTPException(429,'请稍后搜索')
        last_search[0]=time.monotonic()
        try:
            r=session.get('https://nominatim.openstreetmap.org/search',params=dict(q=q,format='json',limit=5),timeout=20);r.raise_for_status();return r.json()
        except requests.RequestException:raise HTTPException(502,'搜索暂不可用，可直接拖动OSM')
    return app
if __name__=='__main__':
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--overlays',required=True);p.add_argument('--store',required=True);p.add_argument('--port',type=int,default=8793);a=p.parse_args();uvicorn.run(build(a.data,a.overlays,a.store,os.environ.get('MAP_REVIEW_USER',''),os.environ.get('MAP_REVIEW_PASSWORD','')),host='127.0.0.1',port=a.port)
