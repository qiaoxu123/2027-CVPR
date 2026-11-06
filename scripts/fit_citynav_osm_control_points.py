"""Fit ONLY public control points; independent validation required for approval."""
import argparse,json
import cv2,numpy as np
p=argparse.ArgumentParser();p.add_argument('--controls',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=json.load(open(a.controls));train=[v for v in d['points'] if v['split']=='fit'];val=[v for v in d['points'] if v['split']=='validation']
if len(train)<4 or len(val)<3:raise ValueError('Require >=4 fit and >=3 independent validation public control points')
fit_xy={tuple(v['world_xy']) for v in train}; validation_xy={tuple(v['world_xy']) for v in val}
if fit_xy & validation_xy:raise ValueError('Fit/validation control points overlap')
if len(fit_xy)!=len(train) or len(validation_xy)!=len(val):raise ValueError('Duplicate controls')
if any(v.get('source') not in ['public_building_corner','public_road_intersection'] for v in train+val):raise ValueError('Only visually confirmed public geometry controls allowed')
M,inliers=cv2.estimateAffine2D(np.array([v['world_xy'] for v in train],float),np.array([v['projected_osm_xy_m'] for v in train],float),method=cv2.RANSAC,ransacReprojThreshold=3)
if M is None:raise ValueError('Fit failed')
x=np.array([v['world_xy']+[1] for v in val]);y=np.array([v['projected_osm_xy_m'] for v in val]);err=np.linalg.norm(x@M.T-y,axis=1);result=dict(validation_n=len(val),matrix=M.tolist(),validation_errors_m=err.tolist(),rmse_m=float(np.sqrt((err**2).mean())),max_error_m=float(err.max()),verified=bool(err.max()<=5),projected_crs=d['projected_crs'],fit_inliers=int(inliers.sum()),note='5m max error is fixed visualization acceptance; not navigational correctness')
json.dump(result,open(a.output,'w'),indent=2)
