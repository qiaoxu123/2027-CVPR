"""Experimental RGB/height structural registration. Fit-only public controls initialize;
independent validation is performed AFTER selecting a transform by structural cost.
No navigation target annotations, private object labels, or validation feedback used.
"""
import json,hashlib,time,xml.etree.ElementTree as ET
from pathlib import Path
import cv2,numpy as np,requests,rasterio
from rasterio.warp import transform
from scipy.ndimage import distance_transform_edt,map_coordinates
from scipy.optimize import least_squares

CONFIG=dict(seed=42,ransac_m=3,max_initial_match_m=25,resolution_m=1,robust_scale_m=3,max_distance_m=20,translation_bound_m=20,rotation_bound_rad=.07,logscale_bound=.06,max_nfev=200,version='structural-proxy-v1')

def metrics(M,points):
 if not points:return None
 xy=np.array([[p['x'],p['y']] for p in points]);e,n=transform('EPSG:4326','EPSG:27700',[p['lon'] for p in points],[p['lat'] for p in points]);en=np.array([e,n]).T;pred=xy@np.array(M)[:,:2].T+np.array(M)[:,2];errs=np.linalg.norm(pred-en,axis=1)
 return dict(n=len(points),mean_m=float(errs.mean()),rmse_m=float(np.sqrt((errs**2).mean())),p95_m=float(np.quantile(errs,.95)),max_m=float(errs.max()),errors=[dict(id=p['id'],split=p['split'],map_name=p['map_name'],error_m=float(er)) for p,er in zip(points,errs)])

def initialize(fit,mode='similarity'):
 if len(fit)<3:raise ValueError('自动初始定位至少需要3个已有拟合点；验证点不参与初始化')
 xy=np.array([[p['x'],p['y']] for p in fit],float);e,n=transform('EPSG:4326','EPSG:27700',[p['lon'] for p in fit],[p['lat'] for p in fit]);en=np.array([e,n]).T
 if np.linalg.matrix_rank(xy-xy.mean(0))<2:raise ValueError('初始点共线')
 cv2.setRNGSeed(42);fn=cv2.estimateAffinePartial2D if mode=='similarity' else cv2.estimateAffine2D;M,mask=fn(xy,en,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=5000,confidence=.999,refineIters=20)
 if M is None:raise ValueError('无法初始化')
 return M,mask.ravel().astype(bool)

def fetch_osm(bbox,cache,proxy):
 cache=Path(cache);cache.mkdir(parents=True,exist_ok=True);key=hashlib.sha256(json.dumps(bbox).encode()).hexdigest()[:16];path=cache/(key+'.osm')
 if not path.exists():
  s=requests.Session();s.proxies={'https':proxy} if proxy else {};s.headers['User-Agent']='CityNavManualControlPoint/1.0 (+https://github.com/qiaoxu123/2027-CVPR)'
  r=s.get('https://www.openstreetmap.org/api/0.6/map',params={'bbox':','.join(map(str,bbox))},timeout=60);r.raise_for_status();ET.fromstring(r.content);path.write_bytes(r.content)
 root=ET.fromstring(path.read_bytes());nodes={n.attrib['id']:(float(n.attrib['lon']),float(n.attrib['lat'])) for n in root.findall('node')};features=[];incomplete=0
 for way in root.findall('way'):
  tags={t.attrib['k']:t.attrib['v'] for t in way.findall('tag')};kind='building' if 'building' in tags else 'road' if 'highway' in tags else None
  if not kind:continue
  refs=[n.attrib['ref'] for n in way.findall('nd')]
  if any(n not in nodes for n in refs):incomplete+=1;continue
  ll=np.array([nodes[n] for n in refs]);e,n=transform('EPSG:4326','EPSG:27700',ll[:,0],ll[:,1]);features.append(dict(id=way.attrib['id'],kind=kind,xy=np.array([e,n]).T.tolist(),tags=tags))
 meta=dict(url='https://www.openstreetmap.org/api/0.6/map',bbox=bbox,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),fetched_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime(path.stat().st_mtime)),copyright='© OpenStreetMap contributors; ODbL',features=len(features),incomplete_ways=incomplete,historical_alignment_warning='Current OSM may differ from older CityNav scans; multipolygon relations not reconstructed in this pilot')
 return features,meta

def estimate(data,map_name,fit,cache,proxy=None):
 """Only FIT controls accepted. Validation is deliberately absent from signature."""
 if any(p['split']!='fit' for p in fit):raise ValueError('Validation points forbidden in optimizer')
 M0,inliers=initialize(fit);data=Path(data)
 with rasterio.open(data/'rgbd'/(map_name+'.tif')) as r:height=r.read(1);T=r.transform;b=r.bounds
 rgb=cv2.imread(str(data/'rgbd'/(map_name+'.png')));orig_h,orig_w=rgb.shape[:2];scale=min(1,1200/max(orig_h,orig_w));rgb=cv2.resize(rgb,(round(orig_w*scale),round(orig_h*scale)),interpolation=cv2.INTER_AREA);height=cv2.resize(height,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_NEAREST);valid=np.isfinite(height)&(height>=0)&(rgb.max(-1)>0)
 if not valid.any():raise ValueError('无有效RGB/高度区域')
 ground=float(np.quantile(height[valid],.1));bb,gg,rr=[rgb[:,:,i].astype(float) for i in range(3)];green=(gg>1.12*rr)&(gg>1.05*bb);building=((height>ground+3)&valid&~green).astype('uint8');building=cv2.morphologyEx(building,cv2.MORPH_CLOSE,np.ones((5,5),np.uint8));building=cv2.morphologyEx(building,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
 hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV);road=((height<ground+2.5)&valid&~green&(hsv[:,:,1]<65)&(hsv[:,:,2]>60)).astype('uint8');road=cv2.morphologyEx(road,cv2.MORPH_OPEN,np.ones((7,7),np.uint8))
 # Classical road-surface proxy skeleton; junctions are candidates, not semantic GT.
 skel=np.zeros_like(road);tmp=road.copy();cross=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
 for _ in range(1000):
  eroded=cv2.erode(tmp,cross);opened=cv2.dilate(eroded,cross);skel|=tmp&~opened;tmp=eroded
  if not tmp.any():break
 degree=cv2.filter2D(skel.astype('float32'),-1,np.ones((3,3),np.float32))-skel
 junction_mask=((skel>0)&(degree>=3)).astype('uint8');count,labels,stats,centroids=cv2.connectedComponentsWithStats(junction_mask)

 sx,sy=orig_w/rgb.shape[1],orig_h/rgb.shape[0]
 def world(p):
  p=np.asarray(p).reshape(-1,2);return np.array([T.c+(p[:,0]+.5)*sx*T.a,T.f+(p[:,1]+.5)*sy*T.e]).T
 contours,_=cv2.findContours(building,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);polys=[]
 for p in contours:
  if cv2.contourArea(p)*abs(T.a*T.e)*sx*sy>=8:polys.append(world(p))
 if not polys:raise ValueError('未提取足够建筑结构；不强行配准')
 corners=np.array([[b.left,b.bottom],[b.right,b.bottom],[b.right,b.top],[b.left,b.top]]);pc=corners@M0[:,:2].T+M0[:,2];lon,lat=transform('EPSG:27700','EPSG:4326',pc[:,0],pc[:,1]);bbox=[min(lon)-.001,min(lat)-.001,max(lon)+.001,max(lat)+.001];features,osm_meta=fetch_osm(bbox,cache,proxy)
 if not any(f['kind']=='building' for f in features):raise ValueError('OSM缺少建筑几何')
 lo=pc.min(0)-50;hi=pc.max(0)+50;size=np.ceil(hi-lo).astype(int)+1;mask=np.zeros((size[1],size[0]),np.uint8)
 def rasterp(en):return np.rint(np.array([en[:,0]-lo[0],hi[1]-en[:,1]]).T).astype('int32')
 for f in features:
  if f['kind']=='building':cv2.polylines(mask,[rasterp(np.array(f['xy']))],True,1,1)
 if mask.sum()<50:raise ValueError('OSM建筑边界不足')
 dt=distance_transform_edt(mask==0);edge=cv2.Canny(building*255,50,100);ys,xs=np.where(edge>0);src=world(np.array([xs,ys]).T);rng=np.random.default_rng(42)
 if len(src)>2500:src=src[rng.choice(len(src),2500,replace=False)]
 pivot=pc.mean(0)
 def dist(M):
  en=src@M[:,:2].T+M[:,2];return np.minimum(20,map_coordinates(dt,np.array([hi[1]-en[:,1],en[:,0]-lo[0]]),order=1,mode='constant',cval=20))
 # Automatic centroid correspondences, one-to-one and size compatible, never using labels.
 source=[];target=[];areas=[]
 for p in polys:
  area=abs(cv2.contourArea(p.astype('float32')))
  if area>=20:source.append(p.mean(0));areas.append(area)
 objects=[]
 for f in features:
  if f['kind']=='building':
   p=np.array(f['xy']);area=abs(cv2.contourArea((p-p.mean(0)).astype('float32')))
   if area>=20:objects.append((p.mean(0),area))
 pairs=[];used=set()
 for i,p in enumerate(source):
  pred=p@M0[:,:2].T+M0[:,2];cand=sorted([(np.linalg.norm(t-pred),j) for j,(t,a) in enumerate(objects) if .25< a/max(areas[i],1)<4])
  if cand and cand[0][0]<25 and cand[0][1] not in used:pairs.append((p,objects[cand[0][1]][0]));used.add(cand[0][1])
 options=[('manual_fit_initial',M0)]
 auto_inliers=0
 if len(pairs)>=4:
  A=np.array([v[0] for v in pairs]);B=np.array([v[1] for v in pairs]);cv2.setRNGSeed(42);M,im=cv2.estimateAffinePartial2D(A,B,method=cv2.RANSAC,ransacReprojThreshold=3)
  if M is not None and im.sum()>=4:options.append(('centroid_ransac',M));auto_inliers=int(im.sum())
 costs=[float(np.minimum(dist(M),10).mean()) for _,M in options];initial_index=int(np.argmin(costs));chosen_name,base=options[initial_index];best_initial=costs[initial_index]
 def make(q):
  dx,dy,ang,ls=q;R=np.array([[np.cos(ang),-np.sin(ang)],[np.sin(ang),np.cos(ang)]])*np.exp(ls);lin=R@base[:,:2];shift=R@(base[:,2]-pivot)+pivot+np.array([dx,dy]);return np.column_stack([lin,shift])
 def residual(q):return np.r_[dist(make(q)),np.array([q[0]/10,q[1]/10,q[2]/.03,q[3]/.03])*.4]
 fitres=least_squares(residual,np.zeros(4),bounds=([-20,-20,-.07,-.06],[20,20,.07,.06]),loss='soft_l1',f_scale=3,diff_step=1e-4,max_nfev=200,x_scale=[5,5,.02,.02]);M=make(fitres.x);newcost=float(np.minimum(dist(M),10).mean())
 if newcost>best_initial:M=base;newcost=best_initial
 return dict(matrix=M.tolist(),initial_matrix=M0.tolist(),config=CONFIG,optimizer_success=bool(fitres.success),fit_control_inliers=int(inliers.sum()),fit_control_inlier_ids=[p['id'] for p,keep in zip(fit,inliers) if keep],fit_control_n=len(fit),auto_centroid_pairs=len(pairs),auto_centroid_inliers=auto_inliers,building_proxy_polygons=len(polys),road_proxy_pixels=int(road.sum()),edge_samples=len(src),structure_cost_initial=float(np.minimum(dist(M0),10).mean()),structure_cost_final=newcost,initialization_selected=chosen_name,osm=osm_meta,source_polygons=[p.tolist() for p in polys],road_junction_proxy_xy=world(centroids[1:]).tolist() if len(centroids)>1 else [],osm_features=features,warning='RGB+height structural proxies, not semantic segmentation truth. No independent validation used for selection.'),rgb,building,road

def run_job(data,city,map_name,controls,output,proxy=None):
 output=Path(output);output.mkdir(parents=True,exist_ok=True);fit=[p for p in controls['points'] if p['split']=='fit'];result,rgb,building,road=estimate(data,map_name,fit,output/'osm_cache',proxy)
 # Freeze prediction before touching private human validation coordinates.
 (output/'prediction_frozen.json').write_text(json.dumps({k:v for k,v in result.items() if k not in ['source_polygons','osm_features']},indent=2))
 val=[p for p in controls['points'] if p['split']=='validation'];M=result['matrix'];before=result['initial_matrix'];result.update(city=city,map_name=map_name,base_revision=controls['revision'],fit_metrics=metrics(M,fit),validation=metrics(M,val),manual_validation=metrics(before,val),status='proposal',verified=False,created_at=time.time())
 if len(fit)>=10 and len(val)>=5 and result['validation']['max_m']<=5:result['verified']=True
 result['per_map_validation']={m:metrics(M,[p for p in val if p['map_name']==m]) for m in {p['map_name'] for p in val}}
 result['manual_modes']={}
 for mode in ['similarity','affine']:
  mm,mask=initialize(fit,mode);result['manual_modes'][mode]=dict(matrix=mm.tolist(),inliers=int(mask.sum()),inlier_ids=[p['id'] for p,keep in zip(fit,mask) if keep],determinant=float(np.linalg.det(mm[:,:2])),scales=np.linalg.svd(mm[:,:2])[1].tolist(),fit=metrics(mm,fit),validation=metrics(mm,val))
 cv2.imwrite(str(output/'building_proxy.png'),building*255);cv2.imwrite(str(output/'road_proxy.png'),road*255)
 render_preview(data,result,output/'alignment.png');(output/'result.json').write_text(json.dumps(result,indent=2));return result

def render_preview(data,r,dest):
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 with rasterio.open(Path(data)/'rgbd'/(r['map_name']+'.tif')) as t:b=t.bounds
 rgb=cv2.cvtColor(cv2.imread(str(Path(data)/'rgbd'/(r['map_name']+'.png'))),cv2.COLOR_BGR2RGB);ext=[b.left,b.right,b.bottom,b.top];fig,axes=plt.subplots(1,3,figsize=(18,6))
 for ax,mat,title in zip(axes[:2],[r['initial_matrix'],r['matrix']],['Manual-fit initial','Automatic structural proposal']):
  ax.imshow(rgb,extent=ext);M=np.array(mat);inv=np.linalg.inv(M[:,:2])
  for f in r['osm_features']:
   p=(np.array(f['xy'])-M[:,2])@inv.T;ax.plot(*p.T,color='cyan' if f['kind']=='building' else 'yellow',lw=.7,alpha=.8)
  ax.set(xlim=ext[:2],ylim=ext[2:],title=title);ax.set_aspect('equal')
 axes[2].imshow(rgb,extent=ext);M=np.array(r['matrix']);inv=np.linalg.inv(M[:,:2]);p=np.concatenate(r['source_polygons']);en=p@M[:,:2].T+M[:,2]
 # Display proximity residual against OSM building boundaries (proxy, not validation error).
 from scipy.spatial import cKDTree
 segments=[]
 for f in r['osm_features']:
  if f['kind']=='building':
   q=np.array(f['xy'])
   for a,b in zip(q[:-1],q[1:]):segments.append(a+(b-a)*np.linspace(0,1,max(2,int(np.ceil(np.linalg.norm(b-a)))+1))[:,None])
 target=np.concatenate(segments);dist=cKDTree(target).query(en)[0];sc=axes[2].scatter(*p.T,c=np.minimum(dist,20),cmap='inferno',s=2,vmin=0,vmax=20);fig.colorbar(sc,ax=axes[2],label='nearest OSM boundary proxy distance (m), not independent validation')
 axes[2].set(xlim=ext[:2],ylim=ext[2:],title='Structural residual heatmap');axes[2].set_aspect('equal');fig.suptitle(r['map_name']+' | RGB-height proxies; proposal not auto-accepted | © OpenStreetMap contributors');fig.tight_layout();fig.savefig(dest,dpi=130);plt.close(fig)
