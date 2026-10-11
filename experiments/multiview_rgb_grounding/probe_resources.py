"""Bounded endpoint probes and actual SkyFind annotation/image checks; no training."""
import concurrent.futures,json,time,subprocess,base64
from pathlib import Path
import cv2
from .resource_audit import fetch

URLS={
 '3dg_data':'https://huggingface.co/api/datasets/fanfuxue/UAV-VLN-FOV',
 '3dg_weights':'https://huggingface.co/api/models/fanfuxue/3DG-VLN',
 'travel_envs':'https://huggingface.co/api/datasets/wangxiangyu0814/TravelUAV_env',
 'travel_data':'https://huggingface.co/api/datasets/wangxiangyu0814/TravelUAV',
 '3dg_instructions':'https://huggingface.co/datasets/fanfuxue/UAV-VLN-FOV/resolve/f296bbdcf3a3cb4203d973abe5c05714de5c2247/UAV-VLN-FOV/meta/instructions.json',
 '3dg_train_meta':'https://huggingface.co/datasets/fanfuxue/UAV-VLN-FOV/resolve/f296bbdcf3a3cb4203d973abe5c05714de5c2247/UAV-VLN-FOV/meta/train_scene_trajectory_instruction_direction.json',
 '3dg_weight_index':'https://huggingface.co/fanfuxue/3DG-VLN/resolve/02e161fffd70fe8899a3b065dd41e45707221fcf/model.safetensors.index.json',
 'qwen_weight_index':'https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/model.safetensors.index.json',
 'anyvisloc_portal':'https://1drv.ms/f/c/fef40436afe2e5bb/IgC0Zat90dsgTY-2lNlDliGXAfuTZOYEeKcllr7VPD_F3eY?e=LJ5fAD',
 'aerialvln_data':'https://www.kaggle.com/api/v1/datasets/list?search=shuboliu%2Faerialvln',
 'qwen_config':'https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/ebb281ec70b05090aa6165b016eac8ec08e71b17/config.json',
 'sam2_checkpoint':'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt'}

def run():
 root=Path('runs/multiview_rgb_grounding/resource_audit');start=time.time()
 def check(kv):
  key,url=kv
  if key=='sam2_checkpoint':
   # HEAD checks availability without downloading a model unused behind a quality gate.
   import requests
   attempts=[];response=None
   for direct in [True,False]:
    session=requests.Session();session.trust_env=not direct
    try:
     r=session.head(url,timeout=15,allow_redirects=True);attempts.append({'direct':direct,'status':r.status_code})
     if r.ok:response=r;break
    except requests.RequestException as e:attempts.append({'direct':direct,'error':type(e).__name__})
   return key,{'url':url,'available':response is not None,'trials':attempts,'check':'HEAD only, checkpoint not downloaded'}
  r,trials=fetch(url);out={'url':url,'trials':trials,'available':r is not None}
  if r is not None:
   (root/(key+'_probe.txt')).write_bytes(r.content)
   try:
    data=r.json()
    if isinstance(data,dict):
     out.update(revision=data.get('sha'),gated=data.get('gated'),card_license=data.get('cardData',{}).get('license'),files=[v['rfilename'] for v in data.get('siblings',[])])
     if 'weight_map' in data:out.update(weight_bytes=data.get('metadata',{}).get('total_size'),weight_shards=sorted(set(data['weight_map'].values())))
     if key in ['3dg_instructions','3dg_train_meta']:out.update(records=len(data),top_level_keys=list(data)[:10])
    elif isinstance(data,list):out.update(records=len(data),sample_schema=list(data[0]) if data and isinstance(data[0],dict) else None)
   except ValueError:out['response_type']='HTML portal, not proof of downloadable dataset'
  return key,out
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:checks=dict(pool.map(check,URLS.items()))
 p=subprocess.run(['gh','api','repos/UAV-AVL/Benchmark/contents/docs/LICENSE_DATASET.md'],capture_output=True)
 if p.returncode==0:
  d=json.loads(p.stdout);(root/'anyvisloc_dataset_license.txt').write_bytes(base64.b64decode(d['content']));checks['anyvisloc_license']={'available':True,'url':d['html_url']}
 image=cv2.imread(str(root/'skyfind_sample_image'));rows=json.loads((root/'skyfind_sample.json').read_text())['rows'];boxes=[]
 for row in rows:
  for ann in row['row']['annotations']:boxes.append(ann)
 audit={'rows':len(rows),'objects':len(boxes),'first_actual_image_shape':list(image.shape) if image is not None else None,
        'first_image_decoded':image is not None,'annotation_keys':sorted(boxes[0]),
        'positive_area_bbox_fraction':sum(v['bbox'][2]>v['bbox'][0] and v['bbox'][3]>v['bbox'][1] for v in boxes)/len(boxes),
        'expressions':sum(len(v['expressions']) for v in boxes),
        'geographic_leakage_verified':False,'use':'inspection only; no training or redistribution of image'}
 destination=root/'probes.json'
 if destination.exists():destination=root/'probes_corrected.json'
 destination.write_text(json.dumps({'checks':checks,'skyfind_sample_audit':audit,'seconds':time.time()-start},indent=2)+'\n')
 print(json.dumps({'availability':{k:v['available'] for k,v in checks.items()},'seconds':time.time()-start}),flush=True)

if __name__=='__main__':run()
