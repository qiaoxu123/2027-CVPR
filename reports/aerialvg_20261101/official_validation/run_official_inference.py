"""Official checkpoint/demo inference, with the repository's own PyTorch deformable core."""
import argparse,json,sys,time,hashlib,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT/'AerialVG';sys.path.insert(0,str(REPO))
from util.slconfig import SLConfig
from util.utils import clean_state_dict
from model import build_model
from model.AerialVG import ms_deform_attn
from demo import load_image,plot_boxes_to_image


class TorchDeformableFallback:
 @staticmethod
 def apply(value,shapes,start,locations,weights,step):
  return ms_deform_attn.multi_scale_deformable_attn_pytorch(value,shapes,locations,weights)


def main():
 p=argparse.ArgumentParser();p.add_argument('--output',default=str(ROOT/'aerialvg_validation/demo_results'));a=p.parse_args();out=Path(a.output)
 if out.exists():raise ValueError('Refusing overwrite')
 out.mkdir(parents=True)
 ms_deform_attn.MultiScaleDeformableAttnFunction=TorchDeformableFallback
 args=SLConfig.fromfile(str(REPO/'config/config_cfg.py'))
 args.text_encoder_type='/home/20261024_1/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594';args.device='cuda'
 model=build_model(args)
 checkpoint=ROOT/'aerialvg_assets/aerialvg.pth'
 with torch.serialization.safe_globals([argparse.Namespace]):s=torch.load(checkpoint,map_location='cpu',weights_only=True)
 load=model.load_state_dict(clean_state_dict(s['model']),strict=True)
 model.to('cuda').eval();model.requires_grad_(False)
 pil,image=load_image(str(REPO/'images/demo.JPG'));image=image.to('cuda')
 caption='A blue car is parking on the right road with a white sedan above it.'
 cases=[('full',caption,False,False),('no_relation',caption,True,False),('no_hierarchy',caption,False,True),('neither',caption,True,True),('counterfactual',caption.replace('above','below'),False,False)]
 original=model.relation_transformer.forward;sim=model.sim_param;results=[]
 for tag,text,disable_relation,disable_hierarchy in cases:
  model.relation_transformer.forward=(lambda features,td:model.class_embed[-1](features,td)) if disable_relation else original
  model.sim_param=sim*0 if disable_hierarchy else sim
  torch.cuda.synchronize();started=time.perf_counter()
  with torch.inference_mode():prediction=model(image[None],captions=[text.lower()])
  torch.cuda.synchronize();seconds=time.perf_counter()-started
  logits=prediction['pred_logits'][0];boxes=prediction['pred_boxes'][0]
  valid=prediction['text_mask'][0]
  assert torch.isfinite(logits[:,valid]).all() and torch.isfinite(boxes).all()
  score=logits[:,valid].sigmoid().amax(-1);ids=score.topk(5).indices
  result=dict(case=tag,caption=text,seconds=seconds,top5_query_ids=ids.cpu().tolist(),top5_scores=score[ids].cpu().tolist(),top5_boxes_cxcywh=boxes[ids].cpu().tolist())
  results.append(result);print(json.dumps(result),flush=True)
  plot_boxes_to_image(pil.copy(),dict(size=[pil.height,pil.width],boxes=boxes[ids[:1]].cpu(),labels=[f'{tag}: {score[ids[0]]:.3f}']))[0].save(out/(tag+'.jpg'))
  torch.save(dict(logits=logits.cpu(),boxes=boxes.cpu(),text_mask=valid.cpu()),out/(tag+'.pt'))
 metadata=dict(commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),checkpoint_sha256=hashlib.file_digest(checkpoint.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(checkpoint.read_bytes()).hexdigest(),strict_load=str(load),operator='Official PyTorch deformable core dispatched on CUDA tensors; no compiled extension',image_shape=list(image.shape),results=results,limitations='One repository demo image, no GT annotations; not benchmark accuracy. Ablations are same-checkpoint interventions.')
 (out/'results.json').write_text(json.dumps(metadata,indent=2))


if __name__=='__main__':main()
