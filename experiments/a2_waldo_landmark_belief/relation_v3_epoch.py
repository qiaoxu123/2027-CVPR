"""All-row V3 epoch execution and same-checkpoint real/teacher DEV metrics."""
import json,time
import numpy as np
import torch
from torch.utils.data import SequentialSampler,RandomSampler
from .batching import to_device
from .relation_v3_loss import relation_v3_loss,LOSS_WEIGHTS
from .relation_v3_runtime import evidence_forward
from .relation_v3_protocol import CONFIG,safe_dev_points

LOSS_FIELDS=('ranking',*LOSS_WEIGHTS)
COUNT_FIELDS=('rank_views','utility_candidates','positive_candidates','proposal_available_views','proposal_covered_views',
              'a0_correct','map_correct','corrected','broken','a0_ne_sum','map_ne_sum','proposal_correct','proposal_ne_sum')


class Accumulator:
    def __init__(self):
        self.views=0;self.loss=0.;self.losses={k:0. for k in LOSS_FIELDS};self.counts={k:0. for k in COUNT_FIELDS}

    def add(self,size,loss,stats,proposal_ne,available):
        self.views+=size;self.loss+=float(loss.detach())*size
        for key in self.losses:self.losses[key]+=stats[key]*size
        for key in self.counts:
            if key not in ('proposal_correct','proposal_ne_sum'):self.counts[key]+=stats[key]
        self.counts['proposal_correct']+=int(((proposal_ne<=20)&available).sum())
        self.counts['proposal_ne_sum']+=float(proposal_ne[available].sum())

    def result(self):
        n=max(self.views,1);a=self.counts['proposal_available_views']
        return dict(views=self.views,loss=self.loss/n,**{k:v/n for k,v in self.losses.items()},**self.counts,
                    proposal_r1=self.counts['proposal_correct']/n,proposal_ne=self.counts['proposal_ne_sum']/a if a else 1e6,
                    a0_r1=self.counts['a0_correct']/n,map_r1=self.counts['map_correct']/n,
                    a0_ne=self.counts['a0_ne_sum']/n,map_ne=self.counts['map_ne_sum']/n)


def run_epoch(model,batches,*,optimizer=None,refine=False,device='cuda'):
    training=optimizer is not None
    if not training and not isinstance(batches.sampler,SequentialSampler):raise ValueError('DEV packet identities require sequential sampler')
    if training and (not isinstance(batches.sampler,(SequentialSampler,RandomSampler)) or getattr(batches.sampler,'replacement',False)):
        raise ValueError('Training must visit each observation without replacement')
    model.train(training);start=time.perf_counter();overall=Accumulator();parts={k:Accumulator() for k in ('teacher','real')}
    updates=0;identities=[];saved={k:[] for k in ('scores','available','before_ne','after_ne','proposal_ne','is_real')}
    with torch.set_grad_enabled(training):
        for index,raw in enumerate(batches,1):
            batch=to_device(raw,device);out=model(**batch['network']);loss,stats=relation_v3_loss(model,out,batch,refine=refine)
            if not torch.isfinite(loss):raise ValueError('Nonfinite V3 epoch loss')
            size=len(out['scores']);offset=overall.views
            with torch.no_grad():
                packet=evidence_forward(batch['spatial'],out,model)
                distance=(batch['spatial']['grid_world'].float()-batch['labels']['target_xy'].float()[:,None]).norm(dim=-1)
                def ne(logits):return distance.gather(1,logits.argmax(-1,keepdim=True)).squeeze(-1)
                proposal_ne=ne(packet['q']);before=ne(batch['spatial']['base_logits']);after=ne(packet['fixed_action_logits'])
                real=batch['labels']['source_is_real'].bool();overall.add(size,loss,stats,proposal_ne,packet['available'])
                for name,mask in (('teacher',~real),('real',real)):
                    if not mask.any():continue
                    sub={group:{k:v[mask] for k,v in values.items()} for group,values in batch.items()}
                    subout={k:v[mask] for k,v in out.items()}
                    subtotal,substats=relation_v3_loss(model,subout,sub,refine=refine)
                    parts[name].add(int(mask.sum()),subtotal,substats,proposal_ne[mask],packet['available'][mask])
                if not training:
                    entries=batches.dataset.entries[offset:offset+size]
                    expected=[e['source_kind']!='teacher_snapshot' for e in entries]
                    if real.cpu().tolist()!=expected:raise ValueError('DEV source/identity order changed')
                    identities.extend(dict(e) for e in entries)
                    values=dict(scores=packet['scores'],available=packet['available'],before_ne=before,after_ne=after,proposal_ne=proposal_ne,is_real=real)
                    for key,value in values.items():saved[key].append(value.detach().cpu().numpy())
            if training:
                optimizer.zero_grad(set_to_none=True);loss.backward()
                torch.nn.utils.clip_grad_norm_(filter(lambda p:p.requires_grad,model.parameters()),CONFIG['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();updates+=1
            if index%100==0:print(json.dumps(dict(batch=index,views=overall.views,training=training,seconds=time.perf_counter()-start)),flush=True)
    if overall.views!=len(batches.dataset):raise ValueError('Incomplete V3 epoch population')
    result=dict(overall.result(),optimizer_updates=updates,teacher=parts['teacher'].result(),real=parts['real'].result(),seconds=time.perf_counter()-start)
    if training:return result,None
    packets={k:np.concatenate(values) for k,values in saved.items()};packets['identities']=identities
    populations={}
    for name,is_real in (('real',True),('teacher',False)):
        mask=packets['is_real']==is_real
        if not mask.any():raise ValueError('DEV must report both real and teacher populations')
        populations[name]={k:packets[k][mask] for k in ('scores','available','before_ne','after_ne')}
        populations[name]['identities']=[e for e,keep in zip(identities,mask) if keep]
    result.update(safe_dev_points(**populations))
    return result,packets
