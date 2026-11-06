"""Public manual control-point registration. Labels/target GT are never inputs."""
import cv2,numpy as np
from rasterio.warp import transform

def fit_controls(points,mode='similarity'):
    fit=[p for p in points if p['split']=='fit'];val=[p for p in points if p['split']=='validation']
    if len(fit)<10 or len(val)<5:raise ValueError('至少需要10个拟合点、5个独立验证点')
    if len({p['id'] for p in points})!=len(points):raise ValueError('控制点编号重复')
    xy=np.array([[p['x'],p['y']] for p in points],float)
    for i in range(len(points)):
        if np.any(np.linalg.norm(xy[i+1:]-xy[i],axis=1)<.5):raise ValueError('控制点重复或间距不足0.5m')
    if not np.isfinite(xy).all():raise ValueError('坐标非有限值')
    lon=np.array([p['lon'] for p in points]);lat=np.array([p['lat'] for p in points]);e,n=transform('EPSG:4326','EPSG:27700',lon,lat);en=np.array([e,n]).T
    for i in range(len(points)):
        if np.any(np.linalg.norm(en[i+1:]-en[i],axis=1)<.5):raise ValueError('OSM控制点重复或间距不足0.5m')
    nf=len(fit);ix=[i for i,p in enumerate(points) if p['split']=='fit'];iv=[i for i,p in enumerate(points) if p['split']=='validation']
    if np.linalg.matrix_rank(xy[ix]-xy[ix].mean(0))<2:raise ValueError('拟合点共线')
    cv2.setRNGSeed(42)
    fn=cv2.estimateAffinePartial2D if mode=='similarity' else cv2.estimateAffine2D
    M,mask=fn(xy[ix],en[ix],method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=5000,confidence=.999,refineIters=20)
    if M is None or abs(np.linalg.det(M[:,:2]))<1e-8:raise ValueError('拟合退化')
    pred=xy@M[:,:2].T+M[:,2];err=np.linalg.norm(pred-en,axis=1)
    def stats(a):return dict(n=len(a),mean_m=float(a.mean()),rmse_m=float(np.sqrt(np.mean(a*a))),p95_m=float(np.quantile(a,.95)),max_m=float(a.max()))
    permap={m:stats(err[[i for i in iv if points[i]['map_name']==m]]) for m in {points[i]['map_name'] for i in iv}}
    plon,plat=transform('EPSG:27700','EPSG:4326',pred[:,0],pred[:,1])
    return dict(mode=mode,matrix=M.tolist(),projected_crs='EPSG:27700',fit=stats(err[ix]),validation=stats(err[iv]),validation_n=len(iv),validation_errors_m=err[iv].tolist(),per_map_validation=permap,ransac_inliers=int(mask.sum()),verified=bool(err[iv].max()<=5),scope='city transform; only independently validated maps approved, not all scenes',points=[dict(id=p['id'],split=p['split'],map_name=p['map_name'],error_m=float(err[i]),pred_lon=plon[i],pred_lat=plat[i]) for i,p in enumerate(points)],datum_note='Rasterio/GDAL selected datum operation; OSTN15 grid availability not independently verified. Validation errors include conversion effects.')
