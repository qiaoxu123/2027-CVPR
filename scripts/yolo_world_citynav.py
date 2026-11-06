"""Frozen TRAIN-only detector experiment. Public annotations used only posthoc.

No instruction, target ID/position, old proposals or task-specific prompt enters YOLO.
All registry agreement metrics are geometry proxies, never human accuracy.
"""
from __future__ import annotations
import argparse, hashlib, json, time, platform, random
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

BANKS={'base':['car','van','truck','parking lot','parking space','building','road'],
       'color':['car','van','truck','red car','white car','black car'],
       'parking':['parking lot','parking area','parking space']}

def family(s):
    if s in ('car','van','truck','red car','white car','black car'): return 'vehicle'
    if s in ('parking lot','parking area','parking space'): return 'parking'
    return s

def tile_windows(w,h,size=640,overlap=.2):
    stride=int(size*(1-overlap))
    def starts(n): return sorted(set(list(range(0,max(n-size,0)+1,stride))+[max(n-size,0)]))
    return [(x,y,min(x+size,w),min(y+size,h)) for y in starts(h) for x in starts(w)]

def iou_matrix(a,b):
    a=np.asarray(a,float).reshape(-1,4);b=np.asarray(b,float).reshape(-1,4)
    lo=np.maximum(a[:,None,:2],b[None,:,:2]);hi=np.minimum(a[:,None,2:],b[None,:,2:])
    area=np.prod(np.maximum(hi-lo,0),axis=-1)
    return area/np.maximum(np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=-1)[:,None]+np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=-1)[None,:]-area,1e-9)

def match_boxes(pred,gt,threshold=.5):
    if not len(pred) or not len(gt):return []
    iou=iou_matrix(pred,gt);valid=iou>=threshold
    r,c=linear_sum_assignment(-(valid*(min(iou.shape)+1)+valid*iou))
    return [(int(a),int(b),float(iou[a,b])) for a,b in zip(r,c) if valid[a,b]]

def family_nms(rows,threshold=.5):
    keep=[]
    for r in sorted(rows,key=lambda x:-x['score']):
        if not any(family(r['label'])==family(k['label']) and iou_matrix([r['box']],[k['box']])[0,0]>threshold for k in keep):keep.append(r)
    return keep

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)

def ensure_config(path,metadata):
    path=Path(path)
    if path.exists():
        old=json.loads(path.read_text())
        for key in ['seed','partition','weights_sha256','queue_sha256','catalog_sha256','ultralytics','config','samples','smoke']:
            if old.get(key)!=metadata.get(key):
                raise ValueError(f"resume configuration mismatch: {key}; use a separate output directory")
    else:
        save(path,metadata)

def project_public(objects,footprint,shape,depth,rgb):
    import cv2
    from shapely.geometry import Polygon
    from graphnav.landmark_projection import world_to_frame
    h,w=shape;view=Polygon(footprint);rows=[]
    valid=cv2.resize(depth.astype('uint8'),(w,h),interpolation=cv2.INTER_NEAREST).astype(bool)&(rgb.max(axis=2)>0)
    types={'Car':'vehicle','Parking':'parking','Building':'building','TrafficRoad':'road'}
    for oid,o in objects.items():
        if o.get('object_type') not in types:continue
        try:
            poly=Polygon(np.asarray(o['contour'])[:,:2]);poly=poly if poly.is_valid else poly.buffer(0)
            if poly.is_empty or poly.area<=0:continue
            inter=poly.intersection(view)
            if inter.is_empty or inter.area<=0:continue
            parts=[inter] if inter.geom_type=='Polygon' else [q for q in inter.geoms if q.geom_type=='Polygon']
            contours=[world_to_frame(np.asarray(q.exterior.coords)[:,:2],footprint,(h,w)) for q in parts]
            if not contours:continue
            pts=np.concatenate(contours);mask=np.zeros((h,w),np.uint8)
            for q in contours:cv2.fillPoly(mask,[np.rint(q).astype('int32')],1)
            count=int(mask.sum());support=float(valid[mask.astype(bool)].mean()) if count else 0
            xy=np.r_[pts.min(0),pts.max(0)].tolist()
            area=float(sum(abs(cv2.contourArea(q.astype('float32'))) for q in contours))
            rows.append(dict(object_id=oid,name=o.get('name',''),category=types[o['object_type']],box=xy,contours=[q.tolist() for q in contours],area=area,coverage=float(inter.area/poly.area),support=support,eligible=area>=16 and support>=.9,visibility='unknown_geometry_only'))
        except (ValueError,TypeError,KeyError):continue
    return rows

def evaluate(dets,proxies,threshold):
    from shapely.geometry import Polygon,box
    result={}
    for f in ['vehicle','parking','building','road']:
        pi=[i for i,d in enumerate(dets) if family(d['label'])==f];gi=[i for i,g in enumerate(proxies) if g['category']==f and g['eligible']]
        matches=match_boxes([dets[i]['box'] for i in pi],[proxies[i]['box'] for i in gi],threshold)
        out=[]
        for a,b,iou in matches:
            g=proxies[gi[b]];rect=box(*dets[pi[a]]['box']);polys=[Polygon(q) for q in g['contours']]
            from shapely.ops import unary_union
            poly=unary_union(polys);union=rect.union(poly).area
            out.append(dict(detection=pi[a],proxy=gi[b],bbox_iou=iou,rectangle_polygon_iou=rect.intersection(poly).area/max(union,1e-9)))
        result[f]=dict(detections=len(pi),proxies=len(gi),matches=out,tp=len(matches),unverified=len(pi)-len(matches),unmatched_proxies=len(gi)-len(matches))
    return result

def main():
    import cv2,torch,ultralytics
    from ultralytics import YOLOWorld
    from graphnav.landmark_projection import frame_to_world
    p=argparse.ArgumentParser();p.add_argument('--queue',required=True);p.add_argument('--raw',required=True);p.add_argument('--catalog',required=True);p.add_argument('--weights',required=True);p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=120);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True);random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    queue=json.load(open(a.queue));assert queue['partition']=='train';tasks=list(queue['tasks']);random.Random(42).shuffle(tasks);tasks=tasks[:a.limit]
    index=json.load(open(Path(a.raw)/'manifest.json'));index={(str(r['episode']),int(r['step'])):r['key'] for r in index['frames']}
    catalog=json.load(open(a.catalog));assert torch.cuda.is_available()
    metadata=dict(seed=42,partition='train',weights_sha256=sha(a.weights),queue_sha256=sha(a.queue),catalog_sha256=sha(a.catalog),torch=torch.__version__,cuda=torch.version.cuda,ultralytics=ultralytics.__version__,python=platform.python_version(),gpu=torch.cuda.get_device_name(),config=dict(imgsz=640,tile=640,overlap=.2,confidence=.25,iou=.5,max_det=1000,half=False,banks=BANKS),samples=len(tasks),smoke=a.smoke)
    ensure_config(out/'config.json',metadata)
    model=YOLOWorld(a.weights).to('cuda');model.model.eval()
    # Warmup is separate from measured inference and uses no scene labels.
    model.set_classes(BANKS['base']);model.predict(np.zeros((640,640,3),np.uint8),imgsz=640,device=0,verbose=False,conf=.25,iou=.5,max_det=1000,half=False)
    for bank,labels in BANKS.items():
        t=time.perf_counter();model.set_classes(labels);torch.cuda.synchronize();save(out/f'text_{bank}.json',dict(seconds=time.perf_counter()-t,labels=labels))
        for no,task in enumerate(tasks):
            name=task['sample_id'].replace(':','_');dest=out/'results'/f'{name}_{bank}.json'
            if dest.exists():continue
            assert task['partition']=='train';assert sha(task['image_path'])==task['image_sha']
            bgr=cv2.imread(task['image_path']);h,w=bgr.shape[:2]
            raw=torch.load(Path(a.raw)/(index[(str(task['episode']),int(task['step']))]+'.pt'),map_location='cpu',weights_only=True)
            assert raw['map_name']==task['map_name']
            fp=np.asarray(raw['footprint_world']);depth=np.asarray(raw['depth_valid'])
            proxies=project_public(catalog[task['map_name']],fp,(h,w),depth,bgr)
            methods={}
            for method in ['whole','tile']:
                windows=[(0,0,w,h)] if method=='whole' else tile_windows(w,h)
                rows=[];torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter();speeds=[]
                for ti,(x,y,x2,y2) in enumerate(windows):
                    pred=model.predict(bgr[y:y2,x:x2],imgsz=640,device=0,verbose=False,conf=.25,iou=.5,max_det=1000,half=False)[0]
                    speeds.append(pred.speed)
                    for bb,score,cl in zip(pred.boxes.xyxy.cpu().tolist(),pred.boxes.conf.cpu().tolist(),pred.boxes.cls.cpu().tolist()):
                        box_=np.asarray(bb)+[x,y,x,y];rows.append(dict(box=box_.tolist(),score=score,label=labels[int(cl)],tile=ti))
                torch.cuda.synchronize();elapsed=time.perf_counter()-t;peak=torch.cuda.max_memory_allocated()/2**20
                kept=family_nms(rows)
                for d in kept:
                    x,y,x2,y2=d['box'];corners=np.array([[x,y],[x2,y],[x2,y2],[x,y2]],float);corners=np.clip(corners,[0,0],[w-1,h-1]);d['world_quad']=frame_to_world(corners,fp,(h,w)).tolist();d['world_center']=frame_to_world(np.array([[(x+x2)/2,(y+y2)/2]]),fp,(h,w))[0].tolist()
                methods[method]=dict(detections=kept,raw_tile_proposals=rows,tiles=windows,seconds=elapsed,peak_allocated_mib=peak,speeds_ms=speeds,proxy_iou50=evaluate(kept,proxies,.5),proxy_iou30=evaluate(kept,proxies,.3))
            native=np.linalg.norm(fp[1]-fp[0])/.1
            record=dict(sample_id=task['sample_id'],episode=task['episode'],map_name=task['map_name'],image_path=task['image_path'],image_sha=task['image_sha'],width=w,height=h,pose=np.asarray(raw['pose']).tolist(),footprint=fp.tolist(),source_native_samples=native,render_sampling_ratio=w/native,proxies=proxies,bank=bank,methods=methods)
            save(dest,record);print(bank,no+1,len(tasks),name,[(m,len(v['detections']),round(v['seconds'],3)) for m,v in methods.items()],flush=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()
