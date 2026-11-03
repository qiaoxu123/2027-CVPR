"""TRAIN-fit registered-core ranking proxy; not pixel-instance accuracy."""
import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from scripts.train_local_chain_position import records,merge_fit_observations,collate,batch_inputs
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from graphnav.checkpointing import file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','labels','extra-cache','extra-labels','fit-observation-filter','role-vectors','instance-labels','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--checkpoints',nargs='+',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve prior audit')
    torch.set_num_threads(4);roles=torch.load(a.role_vectors,weights_only=True);allowed=set(json.loads(a.fit_observation_filter.read_text())['frame_keys'])
    fit=merge_fit_observations(records(a.cache,a.labels,'fit',roles,'reference',allowed),records(a.extra_cache,a.extra_labels,'fit',roles,'reference'));masks=torch.load(a.instance_labels,weights_only=True)
    data=[r for r in fit if r['frame_key'] in masks]
    for r in data:r['instance_labels']=masks[r['frame_key']]
    result={}
    for path in a.checkpoints:
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['architecture']!='world-likelihood-v3':raise ValueError('V3 only')
        model=WorldLikelihoodPositionHead().cuda().eval().requires_grad_(False);model.load_state_dict(state['model']);rows=[]
        with torch.no_grad():
            for batch in DataLoader(data,batch_size=16,collate_fn=collate):
                x=batch_inputs(batch,'cuda');_,visual=model(**x,return_visual=True);visual=visual.cpu();valid=batch['local_valid'];labels=batch['instance_labels']
                for i in range(len(visual)):
                    pos=(labels[i]==1)&valid[i];neg=(labels[i]==0)&valid[i]
                    if not pos.any() or not neg.any():continue
                    known=pos|neg;ids=visual[i].masked_fill(~known,float('-inf')).argmax();all_ids=visual[i].masked_fill(~valid[i],float('-inf')).argmax()
                    rows.append(dict(frame_key=batch['frame_key'][i],task_key=batch['task_key'][i],known_core_target_top1=bool(pos[ids]),all_candidates_top1_label=int(labels[i,all_ids]),
                                     target_minus_other_score=float(visual[i,pos].logsumexp(0)-pos.sum().float().log()-visual[i,neg].logsumexp(0)+neg.sum().float().log())))
        summary=dict(frames=len(rows),physical_targets=len({r['task_key'] for r in rows}),target_core_top1=sum(r['known_core_target_top1'] for r in rows),all_candidate_target=sum(r['all_candidates_top1_label']==1 for r in rows),all_candidate_other=sum(r['all_candidates_top1_label']==0 for r in rows),all_candidate_unknown=sum(r['all_candidates_top1_label']==-1 for r in rows),mean_score_margin=sum(r['target_minus_other_score'] for r in rows)/max(1,len(rows)))
        result[str(path)]=dict(checkpoint_sha256=file_sha256(path),summary=summary,rows=rows)
        print(path.parent.name,json.dumps(summary))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(dict(warning='TRAIN-fit only, weak registered world cores; NOT pixel identity gold or independent result',results=result),indent=2))


if __name__=='__main__':main()
