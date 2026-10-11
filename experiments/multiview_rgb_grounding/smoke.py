"""Precision smoke on real rendered Train ROI; never reranks or trains."""
import json,time
from pathlib import Path
import cv2,numpy as np,torch
from torch.nn import functional as F
from experiments.rgb_grounding.city_features import CropEncoder
from experiments.visual_grounding.adapters.remoteclip import RemoteCLIPEncoder

def run():
 root=Path('runs/multiview_rgb_grounding');features=torch.load(root/'matching_train_pilot_world/features.pt',weights_only=True);cov=json.loads((root/'matching_train_pilot_world/coverage.json').read_text());c=next(v for v in cov if v['scope']=='pair20' and len(v['candidate_ids'])>=2);manifest=json.loads((root/'render_pilot_train_scatter/manifest.json').read_text());f=next(v for v in manifest['frames'] if v['metadata']['sample_id']==c['sample_id'] and v['tilt']==0 and v['splat_radius']==1);roi=next(v for v in f['regions'] if v['object_id']==c['candidate_ids'][0]);x0,y0,x1,y1=roi['bbox'];crop=cv2.imread(str(root/'render_pilot_train_scatter'/f['frame_key']/'tilt0_r1_rgb.png'))[y0:y1,x0:x1];start=time.time();out={}
 for name in ['SigLIP2-Large384','RemoteCLIP-ViT-L14']:
  if name.startswith('SigLIP'):
   enc=CropEncoder();batch=enc.native.processor(images=[cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)],return_tensors='pt').to('cuda')
   with torch.inference_mode():hidden=enc.native.model.vision_model(**batch).last_hidden_state;exact=F.normalize(enc.native.model.vision_model.head(hidden).float(),dim=-1).cpu()[0]
  else:
   from PIL import Image
   enc=RemoteCLIPEncoder('ViT-L-14','/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/weights/RemoteCLIP-ViT-L-14.pt');x=enc.preprocess(Image.fromarray(crop[:,:,::-1].copy())).unsqueeze(0).to('cuda')
   with torch.inference_mode():exact=F.normalize(enc.model.encode_image(x).float(),dim=-1).cpu()[0]
  old=features[(name,c['sample_id'],'pair20')]['vectors'][(0,'tight')][0];cosine=float(F.cosine_similarity(old,exact,dim=0));assert cosine>.995;out[name]={'fp32_reference_cosine':cosine,'finite':bool(torch.isfinite(exact).all()),'dimension':len(exact),'actual_train_roi':c['sample_id'],'source_step':f['source_step'],'cutoff_step':f['observation_cutoff_step']};del enc;torch.cuda.empty_cache()
 out['seconds']=time.time()-start;(root/'cuda_smoke.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)

if __name__=='__main__':run()
