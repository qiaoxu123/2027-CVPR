"""Execute unmodified AerialVG relation and hierarchical fusion modules."""
import json,sys,time
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AerialVG'))
from model.AerialVG.relation import CrossSelfRelationTransformer
from model.AerialVG.fuse_modules import similarity_update
from model.AerialVG.aerialvg import SharedKernelConv


def main():
 torch.manual_seed(42);device='cuda';checks={};start=time.perf_counter()
 module=CrossSelfRelationTransformer(d_model=256,num_heads=8,num_layers=3).to(device).eval()
 image=torch.randn(2,15,256,device=device,requires_grad=True)
 text=torch.randn(2,24,256,device=device,requires_grad=True)
 mask=torch.ones(2,24,device=device,dtype=torch.bool);mask[:,-3:]=False
 td=dict(encoded_text=text,text_token_mask=mask)
 out=module(image,td)
 assert out.shape==(2,15,256)
 assert torch.isfinite(out[:,:,:21]).all() and torch.isneginf(out[:,:,21:]).all()
 out[:,:,:21].square().mean().backward()
 assert image.grad.abs().sum()>0 and text.grad.abs().sum()>0
 checks['relation_forward_backward']=True
 with torch.no_grad():
  padded=text.detach().clone();padded[:,-3:]=10000
  other=module(image.detach(),dict(encoded_text=padded,text_token_mask=mask))
  assert torch.allclose(out[:,:,:21],other[:,:,:21],atol=2e-4,rtol=1e-4)
  checks['text_padding_ignored']=True
  altered=text.detach().clone();altered[:,3]+=3
  changed=module(image.detach(),dict(encoded_text=altered,text_token_mask=mask))
  checks['text_intervention_max_change']=float((out[:,:,:21]-changed[:,:,:21]).abs().max())
  assert checks['text_intervention_max_change']>1e-3
  permutation=torch.randperm(15,device=device)
  permuted=module(image.detach()[:,permutation],td)
  assert torch.allclose(out[:,permutation,:21],permuted[:,:,:21],atol=2e-4,rtol=2e-4)
  checks['candidate_permutation_equivariance']=True
 shapes=torch.tensor([[16,16],[8,8],[4,4],[2,2]],device=device)
 similarity=torch.randn(8,int(shapes.prod(-1).sum()),12,device=device,requires_grad=True)
 down=SharedKernelConv().to(device);up=torch.nn.ConvTranspose2d(4,4,3,stride=2,padding=1,output_padding=1,groups=4).to(device)
 same=similarity_update(similarity,shapes,0.,0.,up,down)
 assert torch.equal(same,similarity)
 checks['hierarchy_zero_coeff_identity']=True
 updated=similarity_update(similarity,shapes,.3,.3,up,down)
 assert updated.shape==similarity.shape and torch.isfinite(updated).all()
 assert not torch.equal(updated,similarity)
 updated.square().mean().backward()
 assert similarity.grad.abs().sum()>0 and down.shared_conv.weight.grad.abs().sum()>0
 checks['hierarchy_forward_backward']=True
 checks['seconds']=time.perf_counter()-start
 path=Path(__file__).with_name('official_module_results.json');path.write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
