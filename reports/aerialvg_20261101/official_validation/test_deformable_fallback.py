import json,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'AerialVG'))
from model.AerialVG.ms_deform_attn import multi_scale_deformable_attn_pytorch

torch.manual_seed(17)
value=torch.randn(1,20,2,4,device='cuda',requires_grad=True)
shapes=torch.tensor([[4,4],[2,2]],device='cuda')
locations=torch.rand(1,5,2,2,4,2,device='cuda',requires_grad=True)
weights=torch.rand(1,5,2,2,4,device='cuda',requires_grad=True)
a=multi_scale_deformable_attn_pytorch(value,shapes,locations,weights)
b=multi_scale_deformable_attn_pytorch(value.detach().cpu(),shapes.cpu(),locations.detach().cpu(),weights.detach().cpu())
error=float((a.detach().cpu()-b).abs().max());assert torch.allclose(a.detach().cpu(),b,atol=2e-5,rtol=2e-5)
a.square().mean().backward();assert value.grad.abs().sum()>0 and locations.grad.abs().sum()>0 and weights.grad.abs().sum()>0
result=dict(cpu_cuda_max_abs_error=error,gradient_to_values_locations_weights=True,compiled_extension_tested=False)
Path(__file__).with_name('deformable_fallback_results.json').write_text(json.dumps(result,indent=2));print(result)
