"""Family/object independence, weak supervision, causal source audit."""
import argparse,json
from pathlib import Path
from collections import Counter
import torch,numpy as np
from transformers import AutoProcessor
from .dataset import city_rows,training_unit,RELATION
from .data import write


def run(a):
    root=Path(a.output)
    if root.exists():raise ValueError('no overwrite')
    root.mkdir(parents=True);processor=AutoProcessor.from_pretrained('google/siglip2-large-patch16-384',use_fast=False,local_files_only=True);sets={};result={}
    for split,path in [('train',a.train),('dev',a.dev),('seen',a.seen)]:
        if not path:continue
        rows=city_rows(path);fam={r['metadata']['family_id'] for r in rows};maps={r['metadata']['map_name'] for r in rows};objects={(r['metadata']['map_name'],r['sealed']['target_object_id']) for r in rows};sets[split]=dict(families=fam,maps=maps,objects=objects)
        count=Counter();future=[];supervised_hard=[];perfamily={}
        for r in rows:
            m=r['metadata'];lab=r['labels'];n=len(r['public']['roi_valid']);seen=bool(lab['identity_positive'].any());count['states']+=1;count['instance_entries']+=n;count['target_visible_states']+=seen;count['target_credible_states']+=lab['target_credible'];count['usable_rank_states']+=bool(lab['positive_roi_mask'].any() and (lab['semantic_valid_mask']&~lab['positive_roi_mask']).any());count['judged_mismatch_rois']+=int((lab['semantic_valid_mask']&~lab['positive_roi_mask']).sum());count['unknown_rois']+=int((~lab['semantic_valid_mask']).sum());count['same_color_weak_negative_rois']+=int((lab['same_color']&lab['semantic_valid_mask']&~lab['positive_roi_mask']).sum());count['relation_in_instruction_states']+=bool(RELATION.search(m['instruction']));tokens=len(processor.tokenizer(m['instruction'],truncation=False)['input_ids']);count['full_language_truncated_states']+=tokens>64
            if seen:perfamily.setdefault(m['family_id'],dict(map_name=m['map_name'],target_credible=False));perfamily[m['family_id']]['target_credible']|=lab['target_credible']
            for rg in m['roi_objects']:
                if rg['source_step']>m['cutoff_step']:future.append(m['sample_id'])
            if (lab['same_color']&lab['semantic_valid_mask']&~lab['positive_roi_mask']).any():supervised_hard.append(dict(sample_id=m['sample_id'],instruction=m['instruction'],target_id=r['sealed']['target_object_id'],same_color_mismatch_ids=[v['object_id'] for j,v in enumerate(m['roi_objects']) if lab['same_color'][j] and lab['semantic_valid_mask'][j] and not lab['positive_roi_mask'][j]],annotation_status='weak geometric contradiction to explicit named relation; colors language-derived, not human RGB verified'))
        units=training_unit(rows);count['training_family_units']=len(units);count['usable_rank_training_units']=sum(bool(r['labels']['positive_roi_mask'].any() and (r['labels']['semantic_valid_mask']&~r['labels']['positive_roi_mask']).any()) for r in units)
        result[split]=dict(counts=dict(count),families=len(fam),maps=sorted(maps),objects=len(objects),visible_families=len(perfamily),credible_visible_families=sum(v['target_credible'] for v in perfamily.values()),future_source_violations=future,human_verified_visual_labels=0)
        write(root/(split+'_hard_negatives.json'),supervised_hard)
    if 'train' in sets and 'dev' in sets:
        leak={k:len(sets['train'][k]&sets['dev'][k]) for k in ['families','maps','objects']};assert all(v==0 for v in leak.values());result['train_dev_overlap']=leak
    if 'train' in sets and 'seen' in sets:result['train_seen_shared_maps_expected']={k:len(sets['train'][k]&sets['seen'][k]) for k in ['families','maps','objects']}
    assert all(not v['future_source_violations'] for k,v in result.items() if 'future_source_violations' in v)
    write(root/'metrics.json',result);print(json.dumps(result),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--train');p.add_argument('--dev');p.add_argument('--seen');p.add_argument('--output',required=True);run(p.parse_args())
if __name__=='__main__':main()
