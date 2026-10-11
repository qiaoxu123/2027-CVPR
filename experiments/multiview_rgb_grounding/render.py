"""Small measured-point perspective renderer; invalid pixels remain invalid.

Z-buffer implements occlusion among measured points, not missing surfaces.
Radius-one point splats are explicit, with original point IDs retained.
"""
import argparse, hashlib, json, time
from pathlib import Path
import cv2, numpy as np, torch
from matplotlib.path import Path as Polygon

ROOT=Path('/home/rental/20260922_1/Workspace/DATA/SensatUrban_Dataset/SensatUrban_Dataset/ply')
OBJECTS=Path('/home/rental/20260922_1/Workspace/DATA/sbfnav-data/cityrefer/objects.json')

def read_cloud(path):
    types={'float32':'<f4','float':'<f4','uint8':'u1','uchar':'u1','int32':'<i4','int':'<i4','double':'<f8'};fields=[];count=None
    with Path(path).open('rb') as f:
        assert f.readline().strip()==b'ply'
        while True:
            line=f.readline().decode('ascii').strip()
            if line.startswith('format '):assert line=='format binary_little_endian 1.0'
            if line.startswith('element vertex '):count=int(line.split()[-1])
            if line.startswith('property '):
                _,typ,name=line.split();fields.append((name,types[typ]))
            if line=='end_header':break
        offset=f.tell()
    data=np.memmap(path,mode='r',dtype=np.dtype(fields),offset=offset,shape=(count,))
    xyz=np.stack([data[k] for k in ['x','y','z']],axis=1).astype(np.float32)
    rgb=np.stack([data[k] for k in ['red','green','blue']],axis=1).astype(np.uint8)
    return xyz,rgb,data

def basis(yaw,tilt):
    p=np.deg2rad(tilt)-np.pi/2
    forward=np.array([np.cos(yaw)*np.cos(p),np.sin(yaw)*np.cos(p),np.sin(p)])
    right=np.array([np.sin(yaw),-np.cos(yaw),0.]);up=np.cross(right,forward)
    return np.stack([right,-up,forward])

def project(xyz,pose,tilt,width=512,hfov=90,near=.5,far=400):
    cam=(np.asarray(xyz,dtype=np.float64)-np.asarray(pose[:3]))@basis(pose[3],tilt).T
    focal=width/(2*np.tan(np.deg2rad(hfov)/2));z=cam[:,2]
    uv=width/2+focal*cam[:,:2]/np.maximum(z[:,None],1e-10)
    good=(z>near)&(np.linalg.norm(cam,axis=1)<=far)&(uv[:,0]>=0)&(uv[:,1]>=0)&(uv[:,0]<width)&(uv[:,1]<width)
    return uv,z,good

def render(xyz,rgb,pose,tilt,width=512,radius=0):
    uv,z,good=project(xyz,pose,tilt,width);ids=np.flatnonzero(good);pix=np.floor(uv[ids]).astype(np.int32)
    # All points in one native pixel cover exactly the same uniform splat.
    # Keeping its closest point before spreading is exactly equivalent to
    # spreading tens of millions of points then sorting every duplicate.
    raw_pixel=pix[:,1]*width+pix[:,0];nearest=np.full(width*width,np.inf)
    np.minimum.at(nearest,raw_pixel,z[ids]);at_min=z[ids]==nearest[raw_pixel]
    base=np.full(width*width,np.iinfo(np.int32).max,np.int32)
    np.minimum.at(base,raw_pixel[at_min],ids[at_min].astype(np.int32));occupied=np.flatnonzero(np.isfinite(nearest))
    projected_points=len(ids);ids=base[occupied];yy,xx=np.divmod(occupied,width);pix=np.stack([xx,yy],1)
    pixels=[];points=[]
    for dy in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            shifted=pix+np.array([dx,dy]);ok=(shifted>=0).all(1)&(shifted<width).all(1)
            pixels.append(shifted[ok,1]*width+shifted[ok,0]);points.append(ids[ok])
    pixel=np.concatenate(pixels);point=np.concatenate(points)
    dep=np.full(width*width,np.inf);np.minimum.at(dep,pixel,z[point]);best=z[point]==dep[pixel]
    winner=np.full(width*width,np.iinfo(np.int32).max,np.int32);np.minimum.at(winner,pixel[best],point[best].astype(np.int32));winner[~np.isfinite(dep)]=-1;valid=winner>=0
    image=np.zeros((width*width,3),np.uint8);image[valid]=rgb[winner[valid]]
    depth=np.full(width*width,np.nan,np.float32);depth[valid]=z[winner[valid]]
    yy,xx=np.divmod(np.flatnonzero(valid),width)
    error=np.linalg.norm(np.stack([xx+.5,yy+.5],1)-uv[winner[valid]],axis=1)
    return dict(rgb=image.reshape(width,width,3),depth=depth.reshape(width,width),valid=valid.reshape(width,width),winner=winner.reshape(width,width),reprojection_max=float(error.max()) if len(error) else 0.,projected_points=projected_points)

def associate(xyz,objects):
    """Public polygon and 3D box weak association; overlap is explicitly unknown."""
    names=sorted(objects,key=lambda x:int(x));assignment=np.full(len(xyz),-1,np.int32);ambiguous=np.zeros(len(xyz),bool);counts=[]
    origin=xyz[:,:2].min(0);cell=8.;bins=np.floor((xyz[:,:2]-origin)/cell).astype(np.int32);nx=int(bins[:,0].max()+1);ny=int(bins[:,1].max()+1)
    keys=bins[:,1]*nx+bins[:,0];order=np.argsort(keys,kind='stable');starts=np.r_[0,np.cumsum(np.bincount(keys,minlength=nx*ny))]
    for i,oid in enumerate(names):
        o=objects[oid];poly=np.asarray(o.get('contour',[]));pos=np.asarray(o.get('position',[]));dim=np.asarray(o.get('dimension',[]))
        if poly.ndim!=2 or len(poly)<3 or pos.shape!=(3,) or dim.shape!=(3,):counts.append(0);continue
        low=poly.min(0);high=poly.max(0)
        lo=np.maximum(np.floor((low-origin)/cell).astype(int),0);hi=np.minimum(np.floor((high-origin)/cell).astype(int),[nx-1,ny-1])
        ids=np.concatenate([order[starts[y*nx+lo[0]]:starts[y*nx+hi[0]+1]] for y in range(lo[1],hi[1]+1)]) if (hi>=lo).all() else np.zeros(0,dtype=np.int64)
        p=xyz[ids];ok=(p[:,0]>=low[0])&(p[:,0]<=high[0])&(p[:,1]>=low[1])&(p[:,1]<=high[1])&(np.abs(p[:,2]-pos[2])<=dim[2]/2+.5)
        ids=ids[ok];ids=ids[Polygon(poly).contains_points(xyz[ids,:2],radius=1e-7)];counts.append(len(ids))
        ambiguous[ids]|=assignment[ids]>=0;assignment[ids]=i
    assignment[ambiguous]=-1
    return assignment,names,counts,int(ambiguous.sum())

def regions(frame,assignment,names,objects,gate):
    win=frame['winner'];inst=np.full(win.shape,-1,np.int32);valid=win>=0;inst[valid]=assignment[win[valid]];result=[]
    for index in np.unique(inst[inst>=0]):
        yy,xx=np.where(inst==index);a,b,c,d=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1);side=min(c-a,d-b)
        support=float(valid[b:d,a:c].mean());oid=names[index]
        result.append(dict(object_id=oid,object_type=objects[oid]['object_type'],bbox=[a,b,c,d],native_min_side=side,measured_visible_pixels=len(xx),bbox_valid_fraction=support,
          credible=bool(side>=gate['instance_min_native_side_pixels'] and len(xx)>=gate['instance_visible_measured_pixels_min'] and support>=gate['instance_bbox_valid_fraction_min']),annotation_confidence='public Polygon+3D box weak association; overlaps Unknown'))
    return result,inst

def run(a):
    root=Path(a.output);assert not root.exists();root.mkdir(parents=True);start=time.time();cv2.setNumThreads(1)
    config_bytes=Path(__file__).with_name('protocol.json').read_bytes();config=json.loads(config_bytes);g=config['quality_gates'];objects=json.loads(OBJECTS.read_text())
    plan=torch.load(Path(a.plan)/'plan.pt',weights_only=True);latest={}
    for s in plan['states']:
        m=s['metadata'];key=m['family_id']
        better=key not in latest or (m['cutoff_step']>latest[key]['metadata']['cutoff_step'] if a.prefix_policy=='latest' else m['cutoff_step']<latest[key]['metadata']['cutoff_step'])
        if better:latest[key]=s
    states=sorted(latest.values(),key=lambda s:hashlib.sha256(s['metadata']['family_id'].encode()).hexdigest())
    if a.limit:states=states[:a.limit]
    states.sort(key=lambda s:(s['metadata']['map_name'],s['metadata']['family_id']));allframes=[];clouds={};current=None
    for si,s in enumerate(states):
        m=s['metadata'];name=m['map_name'];pose=s['public']['pose_world']
        if name!=current:
            paths=list(ROOT.glob('*/'+name+'.ply'));assert len(paths)==1
            xyz,rgb,data=read_cloud(paths[0]);assignment,names,counts,ambiguous=associate(xyz,objects[name]);current=name
            clouds[name]=dict(path=str(paths[0]),bytes=paths[0].stat().st_size,points=len(xyz),properties=list(data.dtype.names),bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],ambiguous_points=ambiguous,association_counts=counts,object_ids=names)
        key=hashlib.sha256(m['sample_id'].encode()).hexdigest()[:16];directory=root/key;directory.mkdir()
        for tilt in config['renderer']['tilt_from_nadir_degrees']:
            for radius in (a.radii if a.radii is not None else config['renderer']['splat_radius_pixels']):
                t=time.time();frame=render(xyz,rgb,pose,tilt,radius=radius);roi,instance=regions(frame,assignment,names,objects[name],g)
                prefix=f'tilt{tilt}_r{radius}';cv2.imwrite(str(directory/(prefix+'_rgb.png')),frame['rgb'][...,::-1]);cv2.imwrite(str(directory/(prefix+'_valid.png')),frame['valid'].astype(np.uint8)*255)
                np.savez_compressed(directory/(prefix+'_geometry.npz'),depth=frame['depth'],winner=frame['winner'],instance=instance)
                summary=dict(metadata=m,frame_key=key,tilt=tilt,splat_radius=radius,camera_pose=pose,world_to_camera_basis=basis(pose[3],tilt).tolist(),intrinsics=[[256,0,256],[0,256,256],[0,0,1]],view_origin='new measured-point counterfactual sensor exposure at actual current pose; not original CityFlight capture',source_step=m['cutoff_step'],observation_cutoff_step=m['cutoff_step'],valid_fraction=float(frame['valid'].mean()),reprojection_max=frame['reprojection_max'],projected_points=frame['projected_points'],regions=roi,seconds=time.time()-t,frame_quality_pass=bool(frame['valid'].mean()>=g['frame_valid_fraction_min'] and frame['reprojection_max']<=g['reprojection_error_max_pixels']))
                (directory/(prefix+'_metadata.json')).write_text(json.dumps(summary,indent=2)+'\n');allframes.append(summary)
        print(json.dumps({'states':si+1,'total':len(states),'map':name,'seconds':time.time()-start}),flush=True)
    oblique=[f for f in allframes if f['tilt']>0 and f['splat_radius']==1];fraction=np.mean([f['frame_quality_pass'] for f in oblique]) if oblique else 0.
    result=dict(complete=True,states=len(states),independent_families=len(states),prefix_policy=a.prefix_policy,source_plan=str(a.plan),protocol_sha256=hashlib.sha256(config_bytes).hexdigest(),clouds=clouds,frames=allframes,seconds=time.time()-start,oblique_frame_pass_fraction=float(fraction),numeric_quality_gate_pass=bool(fraction>=g['pilot_oblique_frames_passing_fraction_min']),requires_visual_facade_review=True,GT_conditioned_camera=False)
    (root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=0);p.add_argument('--prefix-policy',choices=['latest','earliest'],default='latest');p.add_argument('--radii',type=int,nargs='+',choices=[0,1]);run(p.parse_args())
