"""No-GT counterexample: distinguish zero evidence from resampled prior mass."""
import argparse,json,math
from pathlib import Path
import torch
from graphnav.local_chain_raster import rasterized_goal_logprob
from graphnav.world_likelihood_evidence import world_evidence_logits


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    prior=torch.zeros(1,784);prior[0,0]=3;prior[0,400]=2.8;base=prior.log_softmax(-1)
    # Deliberately dense local sample: mathematical counterexample, not a flight.
    uv=torch.tensor([[(400%28+.5)/28,(400//28+.5)/28]]).repeat(1,144,1);valid=torch.ones(1,144,dtype=torch.bool)
    local=base[:,400:401].repeat(1,144)-3+math.log(784)-math.log(144)
    joint=torch.cat((base,local),1);v2=rasterized_goal_logprob(joint,uv,valid,global_cells=784)
    v3=world_evidence_logits(prior,uv,torch.zeros(1,144),valid)
    result=dict(prior_top1=int(base.argmax()),v2_zero_new_evidence_top1=int(v2.argmax()),v3_zero_new_evidence_top1=int(v3.argmax()),
                v2_local_probability_mass=float(joint.softmax(-1)[:,784:].sum()),v2_total_variation=float((v2.exp()-base.exp()).abs().sum()/2),
                v3_total_variation=float((v3.softmax(-1)-base.exp()).abs().sum()/2),
                warning='Artificial repeated-point neutral-evidence counterexample; not a legal flight frame, not measured frequency or SR.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':main()
