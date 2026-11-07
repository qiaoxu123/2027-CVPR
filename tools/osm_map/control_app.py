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
    if not user or len(password)<12:raise ValueError('Set MAP_REVIEW_USER/PASSWORD (>=12 chars)')
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
    def corners(city:str,name:str,preview:str="active"):
        import rasterio
        record=read(city);active=record.get('active_registration') or {};reg=record.get('scene_registrations',{}).get(name) or (active if active.get('map_name',name)==name else {}) or record.get('registration') or {}
        if preview=='auto':
            candidate=store/'auto_proposals'/name/'result.json'
            if not candidate.exists():raise HTTPException(400,'此场景没有自动候选')
            reg=json.loads(candidate.read_text())
            if reg.get('status')=='rejected':raise HTTPException(400,'自动候选已拒绝')
        elif preview!='active':raise HTTPException(400,'Invalid preview source')
        if 'matrix' not in reg:raise HTTPException(400,'尚无变换')
        if not name.startswith(city+'_block_') or '/' in name:raise HTTPException(400)
        p=Path(data)/'rgbd'/(name+'.tif')
        if not p.exists():raise HTTPException(404)
        with rasterio.open(p) as r:b=r.bounds
        xy=np.array([[b.left,b.top],[b.right,b.top],[b.right,b.bottom],[b.left,b.bottom]]);M=np.array(reg['matrix']);en=xy@M[:,:2].T+M[:,2];lon,lat=transform('EPSG:27700','EPSG:4326',en[:,0],en[:,1]);fp=hashlib.sha256(json.dumps(record['points'],sort_keys=True).encode()).hexdigest();stale=bool(reg.get('controls_fingerprint') and reg['controls_fingerprint']!=fp)
        return dict(corners=list(zip(lat,lon)),verified=bool(reg.get('verified',False) and name in reg.get('per_map_validation',{}) and not stale),controls_stale=stale,preview_source=preview)
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
    # Experimental structural proposals are separate from manual controls.
    from concurrent.futures import ThreadPoolExecutor
    from tools.osm_map.auto_registration import run_job
    import re
    executor=ThreadPoolExecutor(max_workers=1);running={}
    def auto_dir(city,name):
        path(city)
        if not re.fullmatch(city+r'_block_\d+',name) or not (Path(data)/'rgbd'/(name+'.tif')).is_file():raise HTTPException(404)
        d=store/'auto_proposals'/name;d.mkdir(parents=True,exist_ok=True);return d
    def auto_summary(d):
        status=d/'status.json'
        if status.exists():
            state=json.loads(status.read_text())
            if state.get('status') in ['running','failed']:return state
        result=d/'result.json'
        if result.exists():
            obj=json.loads(result.read_text());return {k:v for k,v in obj.items() if k not in ['source_polygons','osm_features']}
        status=d/'status.json';return json.loads(status.read_text()) if status.exists() else dict(status='not_started')
    @app.get('/api/registration-status')
    def registration_status():
        p=store/'batch34_summary.json';summary=json.loads(p.read_text()) if p.exists() else dict(n=34,rows=[])
        for row in summary.get('rows',[]):
            f=store/'auto_proposals'/row['map_name']/'result.json'
            if f.exists():
                r=json.loads(f.read_text());row['review_status']=r.get('status','proposal')
                if r.get('grade'):
                    row.update(grade=r['grade'],validation=r.get('validation'),selected=r.get('selected'),precise_2m=bool(r.get('grade')=='verified' and r.get('validation') and r['validation']['n']>=5 and r['validation']['max_m']<=2))
                current=read(row['city']);fp=hashlib.sha256(json.dumps(current['points'],sort_keys=True).encode()).hexdigest();row['controls_stale']=bool(r.get('controls_fingerprint') and r['controls_fingerprint']!=fp)
        if summary.get('rows'):
            summary.update(verified=sum(r['grade']=='verified' for r in summary['rows']),pending=sum(r['grade']=='pending_validation' for r in summary['rows']),unreliable=sum(r['grade']=='unreliable' for r in summary['rows']),precise_2m=sum(r.get('precise_2m',False) for r in summary['rows']))
        return summary
    @app.get('/api/osm-export/{name}')
    def osm_export(name:str):
        if not re.fullmatch(r'(cambridge|birmingham)_block_\d+',name):raise HTTPException(404)
        f=store/'validated_osm_exports'/(name+'.json');rp=store/'auto_proposals'/name/'result.json'
        if not f.exists() or not rp.exists():raise HTTPException(403,'此场景尚无独立验证通过的OSM导出')
        reg=json.loads(rp.read_text());v=reg.get('validation') or {};current=read(name.split('_')[0]);fp=hashlib.sha256(json.dumps(current['points'],sort_keys=True).encode()).hexdigest()
        if reg.get('status')=='rejected' or not reg.get('verified') or v.get('n',0)<5 or v.get('max_m',1e9)>5 or reg.get('controls_fingerprint')!=fp:raise HTTPException(403,'验证不足、控制点已变化或候选已被拒绝')
        exported=json.loads(f.read_text())
        if not np.allclose(np.asarray(exported.get('registration_matrix')),np.asarray(reg['matrix']),rtol=0,atol=1e-8):raise HTTPException(403,'配准已变化，需重新生成OSM导出')
        return FileResponse(f,media_type='application/json',filename=name+'_osm_local_xy.json')
    @app.get('/api/auto/{city}/{name}')
    def auto_get(city:str,name:str):return auto_summary(auto_dir(city,name))
    @app.get('/api/auto/{city}/{name}/image')
    def auto_image(city:str,name:str):
        p=auto_dir(city,name)/'alignment.png'
        if not p.exists():raise HTTPException(404)
        return FileResponse(p,media_type='image/png')
    @app.post('/api/auto/{city}/{name}/start')
    def auto_start(city:str,name:str):
        d=auto_dir(city,name);key=name
        if key in running and not running[key].done():raise HTTPException(409,'自动配准正在运行')
        snapshot=read(city)
        if len([p for p in snapshot['points'] if p['split']=='fit'])<3:raise HTTPException(400,'至少需要3个已有拟合点作为初始约束')
        if (d/'result.json').exists():
            archive=d/('result-'+str(int(time.time()))+'.json');os.replace(d/'result.json',archive)
        atomic(d/'controls_snapshot.json',snapshot);atomic(d/'status.json',dict(status='running',base_revision=snapshot['revision']))
        def work():
            try:
                from tools.osm_map.batch_policy import select_public_transform,grade_independent
                from tools.osm_map.auto_registration import metrics,render_preview
                import rasterio
                fingerprint=hashlib.sha256(json.dumps(snapshot['points'],sort_keys=True).encode()).hexdigest();seedfile=store/'city_seeds'/(city+'.json');seed=json.loads(seedfile.read_text()) if seedfile.exists() else {}
                initial=seed.get('matrix') if seed.get('controls_fingerprint')==fingerprint else None
                r=run_job(data,city,name,snapshot,d,os.environ.get('OSM_HTTP_PROXY'),initial_matrix=initial)
                with rasterio.open(Path(data)/'rgbd'/(name+'.tif')) as tif:bounds=list(tif.bounds)
                choice=select_public_transform(r['initial_matrix'],r['matrix'],bounds,r['structure_cost_initial'],r['structure_cost_final'],r['edge_samples'],r['building_proxy_polygons'])
                r['scene_local_candidate_matrix']=r['matrix'];r.update(choice);atomic(d/'selected_prediction_frozen.json',choice)
                val=[p for p in snapshot['points'] if p['split']=='validation' and p['map_name']==name];r['validation']=metrics(r['matrix'],val);r['grade']=grade_independent(r['validation'],r['risk_flags'],sum(p['split']=='fit' for p in snapshot['points']));r['verified']=r['grade']=='verified';r['per_map_validation']={name:r['validation']} if val else {};r['controls_fingerprint']=fingerprint;r['initialization_source']='frozen city seed' if initial else 'current manual fit';render_preview(data,r,d/'alignment.png');atomic(d/'result.json',r)
                if r['verified']:
                    from scripts.summarize_citynav_osm_batch import validated_export
                    exdir=store/'validated_osm_exports';exdir.mkdir(exist_ok=True);atomic(exdir/(name+'.json'),validated_export(r,dict(map_name=name,grade=r['grade'],matrix=r['matrix'],validation=r['validation']),data))
                atomic(d/'status.json',dict(status='proposal'))
            except Exception as e:atomic(d/'status.json',dict(status='failed',error=str(e)[:500]))
        running[key]=executor.submit(work);return dict(status='running',base_revision=snapshot['revision'])
    @app.post('/api/auto/{city}/{name}/accept')
    def auto_accept(city:str,name:str):
        d=auto_dir(city,name);r=auto_summary(d)
        if 'matrix' not in r or r.get('status')=='rejected':raise HTTPException(400,'无可接受候选')
        with (store/(city+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);current=read(city)
            fingerprint=hashlib.sha256(json.dumps(current['points'],sort_keys=True).encode()).hexdigest()
            if (r.get('controls_fingerprint') and r['controls_fingerprint']!=fingerprint) or (not r.get('controls_fingerprint') and current['revision']!=r['base_revision']):raise HTTPException(409,'控制点已变化，请重新运行自动配准')
            atomic(store/(city+f'.history-{current["revision"]}.json'),current)
            # Accept preview only; verification remains false without independent evidence.
            current.setdefault('scene_registrations',{})[name]=r;current['active_registration']=r;current['revision']+=1;atomic(path(city),current)
            full=json.loads((d/'result.json').read_text());full['status']='accepted';atomic(d/'result.json',full)
        return dict(status='accepted',verified=r.get('verified',False),revision=current['revision'])
    @app.post('/api/auto/{city}/{name}/reject')
    def auto_reject(city:str,name:str):
        d=auto_dir(city,name)
        if not (d/'result.json').exists():raise HTTPException(400,'无候选')
        r=json.loads((d/'result.json').read_text());r['status']='rejected';atomic(d/'result.json',r)
        with (store/(city+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);current=read(city);active=current.get('active_registration',{});scenes=current.get('scene_registrations',{})
            if name in scenes and scenes[name].get('base_revision')==r.get('base_revision'):scenes.pop(name);current['revision']+=1;atomic(path(city),current)
            if active.get('map_name')==name and active.get('base_revision')==r.get('base_revision'):
                atomic(store/(city+f'.history-{current["revision"]}.json'),current);current.pop('active_registration',None);current['revision']+=1;atomic(path(city),current)
        return dict(status='rejected',revision=current['revision'])
    return app
if __name__=='__main__':
    import uvicorn
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--overlays',required=True);p.add_argument('--store',required=True);p.add_argument('--port',type=int,default=8793);a=p.parse_args();uvicorn.run(build(a.data,a.overlays,a.store,os.environ.get('MAP_REVIEW_USER',''),os.environ.get('MAP_REVIEW_PASSWORD','')),host='127.0.0.1',port=a.port)
