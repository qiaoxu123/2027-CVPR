"""TRAIN-only BF16/FP32 batch invariance; no tuning on navigation results."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from graphnav.direct_target_data import load_model,batch_inputs
from scripts.train_direct_target import sample_rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();s=a.source
    tasks=json.loads((s/'inventory_validated/public_tasks.json').read_text());labels=json.loads((s/'inventory_validated/private_labels.json').read_text())
    # Labels are loaded by this shared evaluation loader but are NOT consumed
    # by any forward, comparison, or precision decision in this experiment.
    rows=sample_rows([t for t in tasks if t['subset']=='dev'],labels,s/'cache_aligned')
    model=load_model(torch.load(s/'training_aligned/best.pt',map_location='cpu',weights_only=True)).eval().requires_grad_(False);torch.set_num_threads(4)
    result={}
    with torch.inference_mode():
        for precision in ('bf16','fp32'):
            values={}
            for size in (1,8):
                ids=[];logits=[]
                for start in range(0,len(rows),size):
                    inputs=batch_inputs(rows[start:start+size],visual=False);model.reset_episode()
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                        fields=model.forward_direct_target(**inputs)
                    ids.extend(torch.stack([fields[k].belief_logits.argmax(-1) for k in ('geometry_field','relation_field')],-1).cpu().tolist())
                    logits.append(torch.stack([fields[k].belief_logits.float().cpu() for k in ('geometry_field','relation_field')],1))
                values[size]=(np.array(ids),torch.cat(logits))
            result[precision]=dict(geometry_different_states=int((values[1][0][:,0]!=values[8][0][:,0]).sum()),
                relation_different_states=int((values[1][0][:,1]!=values[8][0][:,1]).sum()),
                max_abs_logit_difference=float((values[1][1]-values[8][1]).abs().max()))
    result.update(states=len(rows),queries=80,GT_used_for_precision_choice=False,
                  float32_matmul_precision=torch.get_float32_matmul_precision(),
                  allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    assert result['fp32']['geometry_different_states']==result['fp32']['relation_different_states']==0
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
