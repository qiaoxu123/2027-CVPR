"""Small TRAIN-only angle/Polygon/Belief gallery, with explicitly offline Oracle."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,atomic_save
from graphnav.full_relation_training import load_full_relation_model
from graphnav.token_language import FrozenBERTTokens
from graphnav.candidate_geometry import collate_full
from graphnav.unified_angle_only_belief import AngleOnlyBelief
from scripts.train_unified_angle_only import build_data,IndexedDataset,forward_frozen,labels_after_forward,target_error


def main():
 p=argparse.ArgumentParser()
 for k in ('run','source','baseline','public-objects','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
 base=torch.load(a.baseline,weights_only=True,map_location='cpu');checkpoint=torch.load(a.run/'full/last.pt',weights_only=True,map_location='cpu')
 core=load_full_relation_model(base,'cuda').eval().requires_grad_(False);head=AngleOnlyBelief().cuda().eval();head.load_state_dict(checkpoint['head']);head.requires_grad_(False)
 cfg=base['config']['geometry_args'];bert=FrozenBERTTokens(cfg['text_backbone'],device='cuda',max_length=cfg['max_text_length']);bert.load_cache(a.run/'full/bert_cache.pt')
 ds=build_data(a.source,base,public_objects=a.public_objects);indexed=IndexedDataset(ds)
 candidates=[i for i,r in enumerate(ds.records) if r['map_name'].startswith('birmingham_') and r['landmarks']]
 ids=[candidates[i] for i in np.linspace(0,len(candidates)-1,24,dtype=int)];cases=[];payload=[];cf=[]
 with torch.no_grad():
  for ordinal,i in enumerate(ids):
   row=ds.records[i];batch=collate_full([indexed[i]]);x,pred=forward_frozen(core,head,bert,batch,ds.records,'cuda')
   bins,valid=labels_after_forward(batch,dict(nodes=batch['nodes'].cuda(),named_mask=x['landmark_valid']),head,'cuda')
   oracle=head.apply_angle(base_logits=x['base_logits'],candidate_bearings=x['candidate_bearings'],landmark_valid=valid,bearing_probabilities=F.one_hot(bins,8).float())
   bounds=np.asarray(row['world_bounds']);target=np.asarray(row['target_xy'])*(bounds[2:]-bounds[:2])+bounds[:2]
   errors=[float(target_error(z,batch)[0]) for z in (x['base_logits'],pred['logits'],oracle['logits'])]
   probs=pred['angle_probs'][0,0].cpu().numpy();theta=np.arange(8)*np.pi/4;angle=float(np.arctan2((probs*np.sin(theta)).sum(),(probs*np.cos(theta)).sum()))
   fig,axs=plt.subplots(1,4,figsize=(15,4));center=None
   for j,(ax,z,title) in enumerate(zip(axs[:3],(x['base_logits'],pred['logits'],oracle['logits']),('BASE','PREDICTED angle','ORACLE angle: TRAIN OFFLINE'))):
    ax.imshow(z[0].softmax(-1).reshape(28,28).cpu().numpy(),origin='lower',extent=[bounds[0],bounds[2],bounds[1],bounds[3]])
    for lm in row['landmarks']:
     c=np.asarray(lm['contour'])*(bounds[2:]-bounds[:2])+bounds[:2];ax.plot(*np.vstack((c,c[0])).T,'c-',lw=.7)
     if center is None:center=c.mean(0)
    ax.plot(*target,'r*');ax.arrow(*center,40*np.cos(angle),40*np.sin(angle),color='orange',width=.8,length_includes_head=True);ax.set_title(f'{title}\nNE={errors[j]:.2f}m',fontsize=8)
   axs[3].bar(np.arange(8),probs);axs[3].axvline(int(bins[0,0]) if valid[0,0] else -1,c='r',label='GT bin: offline; -1 invalid');axs[3].set(xlabel='Map bearing bin: +X=0, CCW',ylabel='Predicted P(k)',ylim=(0,1));axs[3].legend(fontsize=7)
   fig.suptitle(row['instruction'][:160],fontsize=8);fig.tight_layout();filename=f'train_{i:05d}.png';fig.savefig(a.output/filename,dpi=140);plt.close(fig)
   case=dict(TRAIN_index=i,original_instruction=row['instruction'],map_name=row['map_name'],public_anchors=len(row['landmarks']),first_anchor_probs=probs.tolist(),GT_first_anchor_bin_OFFLINE=(int(bins[0,0]) if valid[0,0] else None),angle_delta_max=float(pred['delta'].abs().max()),NE_baseline_predicted_oracle=errors,figure=filename)
   cases.append(case);payload.append(dict(index=i,baseline=x['base_logits'].cpu(),predicted=pred['logits'].cpu(),oracle_OFFLINE=oracle['logits'].cpu(),angle_probs=pred['angle_probs'].cpu(),label_OFFLINE=bins.cpu(),valid=valid.cpu()))
   if ordinal<12:
    name=row['landmarks'][0]['name'];texts=[f'A building {rel} of {name}.' for rel in ('north','south','east','west')]
    tokens,mask=bert.encode_tokens(texts);names=x['node_emb'].expand(4,-1,-1);named=x['landmark_valid'].expand(4,-1)
    probabilities=head.predict_angle(tokens=tokens.cuda(),token_mask=mask.cuda(),node_emb=names,landmark_valid=named).softmax(-1)[:,0]
    cf.append(dict(index=i,name=name,texts=texts,probabilities=probabilities.cpu().tolist(),NS_mass_response_under_engineering_axis=float(probabilities[0,2]-probabilities[1,2]),EW_mass_response_under_engineering_axis=float(probabilities[2,0]-probabilities[3,0]),GT_used=False,geographic_axis_verified=False))
 atomic_json(dict(cases=cases,selection='24 evenly spaced anchored Birmingham TRAIN cases, not independent accuracy',arrow='fixed40m display length only, not an input or predicted distance'),a.output/'cases.json');atomic_json(cf,a.output/'counterfactual.json');atomic_save(payload,a.run/'diagnostic_fields.pt')
 counts={};allrows=ds.records
 for r in allrows:
  k=len(r['landmarks']);counts['with_public_anchor']=counts.get('with_public_anchor',0)+int(k>0);counts['no_public_anchor']=counts.get('no_public_anchor',0)+int(k==0);counts['anchor_queries']=counts.get('anchor_queries',0)+k
 # GT labels below are post-hoc ambiguity audit only; not a model input.
 conflict_rows=0;duplicate_rows=0;conflict_groups=0
 for row in allrows:
  groups=defaultdict(list);scale=np.asarray(row['physical_map_sizes']);target=np.asarray(row['target_xy'])
  for lm in row['landmarks']:
   center=np.asarray(lm['contour']).mean(0);v=(target-center)*scale
   if np.linalg.norm(v)<=1:continue
   k=int(np.round((np.arctan2(v[1],v[0])%(2*np.pi))*8/(2*np.pi)))%8
   groups[f"{lm.get('object_type','')} {lm['name']}"] .append(k)
  duplicate_rows+=any(len(x)>1 for x in groups.values())
  conflicts=sum(len(x)>1 and len(set(x))>1 for x in groups.values());conflict_rows+=conflicts>0;conflict_groups+=conflicts
 counts.update(queries_with_identical_name_inputs_multiple_anchors=int(duplicate_rows),queries_with_conflicting_GT_angle_bins_for_identical_head_inputs=int(conflict_rows),conflicting_name_groups=int(conflict_groups),conflict_audit_GT_offline_only=True)
 atomic_json(counts,a.output/'data_coverage.json')
if __name__=='__main__':main()
