"""Frozen-cache P2.5 diagnostics. No optimizer or image encoding is present."""
import argparse,copy,gzip,json,time,re
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from graphnav.checkpointing import file_sha256
from graphnav.appearance_matching import ObservedSigLIPMatchingEncoder
from experiments.rgb_grounding.city_features import visual_phrase
from experiments.bidirectional_core.cache import conservative_attributes
from experiments.instance_grounding.dataset import city_rows,collate
from experiments.instance_grounding.evaluate import load_model,bind_contours,summary,family_rows
from experiments.instance_grounding.model import InstanceVEN
from experiments.bidirectional_core.stats import paired_difference,summarize_rows
from .core import percentile,public_relations,compose_scores,offline_residual,auc,reliability,risk_curve

SEED=20261011
COLORS={'white':'gray','gray':'white','red':'white','black':'white','blue':'red','green':'white','brown':'white','yellow':'white'}

def write(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True);Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def save_rows(path,rows):
    with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:
        for r in rows:f.write((json.dumps(r,separators=(',',':'),allow_nan=False)+'\n').encode())


def query_spec(instruction):
    phrase,supported=visual_phrase(instruction);n=re.search(r'\b(building|car|vehicle|truck|van)\b',phrase.lower());typ='Building' if n and n[0]=='building' else ('Car' if n else None)
    attrs=conservative_attributes(phrase,typ) if supported and typ else {};swapped=None
    for field in ['roof_color','building_color','vehicle_color']:
        value=attrs.get(field)
        if value in COLORS:
            # Phrase was conservatively bound before this replacement; never
            # replace a nearby reference object's color in the full instruction.
            opposite=COLORS[value];changed=re.sub(r'\b'+re.escape(value)+r'\b',opposite,phrase,count=1,flags=re.I)
            if changed!=phrase:swapped=dict(field=field,original=value,replacement=opposite,phrase=changed,attributes={**attrs,field:opposite});break
    return dict(phrase=phrase,supported=supported,typ=typ,attributes=attrs,swap=swapped)


def text_cache(a):
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True);started=time.time();base={};specs={};source={}
    for split in ['dev','seen']:
        p=Path(a.p2)/('features_'+split);m=json.loads((p/'manifest.json').read_text());assert file_sha256(p/'features.pt')==m['feature_sha256'];source[split]=file_sha256(p/'manifest.json')
        data=torch.load(p/'features.pt',weights_only=True,mmap=True)
        for instruction,q in data['queries'].items():
            base[instruction]=dict(vector=q['texts']['full'],tokens=q['language_tokens'],valid=q['language_valid']);base[q['phrase']]=dict(vector=q['texts']['phrase'],tokens=q['language_tokens'],valid=q['language_valid']) if q['phrase']==instruction else dict(vector=q['texts']['phrase']);specs[instruction]=query_spec(instruction)
        del data
    needed=sorted({s['phrase'] for s in specs.values() if s['phrase'] not in base or 'tokens' not in base[s['phrase']]}|{s['swap']['phrase'] for s in specs.values() if s['swap'] and s['swap']['phrase'] not in base})
    protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());assert len(needed)<=protocol['budget']['text_only_max_unique'];encoder=ObservedSigLIPMatchingEncoder(device='cuda');assert encoder.model.config._commit_hash=='1b426889ea62b5a72bf9839009a1b184bfc9c178';torch.set_num_threads(2)
    for off in range(0,len(needed),8):
        words=needed[off:off+8];batch=encoder.processor(text=words,padding='max_length',truncation=True,max_length=64,return_tensors='pt').to('cuda')
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):tokens=encoder.model.text_model(**batch).last_hidden_state.cpu().half()
        vectors=encoder.texts(words).cpu().half()
        for j,text in enumerate(words):base[text]=dict(vector=base[text]['vector'] if text in base else vectors[j],tokens=tokens[j],valid=batch['input_ids'][j].cpu()!=encoder.processor.tokenizer.pad_token_id)
    assert encoder.encoded_images==0 and all(not p.requires_grad for p in encoder.model.parameters());torch.save(dict(texts=base,specs=specs),out/'text_features.pt');write(out/'manifest.json',dict(new_text_queries=len(needed),cached_texts=len(base),image_encodings=0,training_steps=0,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),feature_sha256=file_sha256(out/'text_features.pt'),source_manifest_sha256=source,siglip_revision=encoder.model.config._commit_hash,protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json'))))


@torch.inference_mode()
def frozen_predict(model,rows,texts,mode,control):
    results=[]
    for off in range(0,len(rows),4):
        chunk=rows[off:off+4];adapt=[]
        for r in chunk:
            q={**r,'texts':dict(r['texts'])}
            if mode!='full':
                spec=query_spec(r['metadata']['instruction']);text=spec['swap']['phrase'] if mode=='swap' and spec['swap'] else spec['phrase'];t=texts[text];q.update(language_tokens=t['tokens'],language_valid=t['valid']);q['texts']=dict(full=t['vector'],phrase=t['vector'])
            adapt.append(q)
        p,_,_=collate(adapt,control);p={k:v.cuda() for k,v in p.items()};p['relative'].zero_() # RGB-only: geometry kept for separate public ablations.
        if isinstance(model,InstanceVEN):scores,_,tri=model(p)
        elif model is None:scores=(F.normalize(p['images'],dim=-1)*F.normalize(p['full'],dim=-1)[:,None]).sum(-1)*20;tri=None
        else:scores=model(dict(text=p['full'],images=p['images'],patch_tokens=p['patch_tokens'],language_tokens=p['language_tokens'],language_valid=p['language_valid'],quality=p['quality'],valid=p['valid'])).score;tri=None
        for j,r in enumerate(chunk):
            n=len(r['public']['roi_valid']);results.append(dict(score=scores[j,:n].cpu(),tri=tri[j,:n].cpu() if tri is not None else None))
    return results


def offline_labels(r):
    state=r['state']['public'];bounds=np.array(state['bounds']);xy=bounds[:2]+np.array(state['grid_xy'])*(bounds[2:]-bounds[:2]);distance=np.linalg.norm(xy-np.array(r['sealed']['target_world']),axis=1);static=np.array(state['static_logits']);top=int(np.argsort(static,kind='stable')[-1]);positive=r['labels']['identity_positive'].numpy();visible=bool(positive.any());wrong=bool(distance[top]>20)
    return dict(distance=distance,static=static,positive=positive,visible=visible,wrong=wrong,credible_error=wrong and bool(r['labels']['target_credible']),focus=wrong and bool(r['labels']['target_credible']))


def metric_record(r,score,language,labels):
    n=len(score);order=np.argsort(np.asarray(score),kind='stable')[::-1];pos=labels['positive'];eligible=labels['visible'] and n>=2;target=np.flatnonzero(pos);size=r['metadata']['roi_objects'][int(target[0])]['native_min_side'] if len(target) else None
    def masked(mask,positive):
        ids=np.flatnonzero(np.asarray(mask,bool))
        if len(ids)<2 or not np.any(positive[ids]):return None
        tied=ids[np.asarray(score)[ids]==np.asarray(score)[ids].max()];return float(np.mean(positive[tied]))
    rank=float(np.sum(np.asarray(score)>np.asarray(score)[pos].max())+(np.sum(np.asarray(score)==np.asarray(score)[pos].max())+1)/2) if eligible else None;identity=float(np.mean(pos[np.asarray(score)==np.asarray(score).max()])) if eligible else None;attr=masked(r['labels']['attribute_judged'],r['labels']['attribute_positive'].numpy())
    m=dict(identity_r1=identity,hard_r1=masked(r['labels']['same_class'],pos),same_color_r1=masked(r['labels']['same_color'],pos),attribute_r1=attr,target_rank=float(rank) if eligible else None,normalized_target_rank=(rank-1)/(n-1) if eligible else None,candidate_count=n,focus_identity_r1=identity if labels['focus'] else None,focus_hard_r1=masked(r['labels']['same_class'],pos) if labels['focus'] else None,focus_rank=float(rank) if eligible and labels['focus'] else None)
    m['supported_phrase_identity_r1']=identity if labels.get('phrase_supported',False) else None;m['supported_relation_identity_r1']=identity if labels.get('relation_available',False) else None
    meta=r['metadata'];typ=r['metadata']['roi_objects'][int(target[0])]['object_type'] if len(target) else None
    return dict(sample_id=meta['sample_id'],family_id=meta['family_id'],map_name=meta['map_name'],cutoff_step=meta['cutoff_step'],metrics=m,target_native_side=size,target_type=typ,credible_error=labels['focus'],phrase_supported=labels.get('phrase_supported',False),relation_supported=labels.get('relation_available',False),target_visible=labels['visible'],candidate_object_ids=[v['object_id'] for v in meta['roi_objects']],candidate_source_steps=[v['source_step'] for v in meta['roi_objects']],scores=np.asarray(score).tolist(),target_object_id=r['sealed']['target_object_id'],language_mode=language,labels_status='original identity and attributes weak; no human semantic verification')


def grouped(records):
    result={}
    for k,fn in [('pixels_8_15',lambda r:r['target_native_side'] is not None and r['target_native_side']<16),('pixels_16_31',lambda r:r['target_native_side'] is not None and 16<=r['target_native_side']<32),('pixels_32_63',lambda r:r['target_native_side'] is not None and 32<=r['target_native_side']<64),('pixels_64_plus',lambda r:r['target_native_side'] is not None and r['target_native_side']>=64)]+[('type_'+t,lambda r,t=t:r['target_type']==t) for t in sorted({r['target_type'] for r in records if r['target_type']})]:
        selected=[r for r in records if fn(r)];result[k]={key:summarize_rows(selected,key,seed=SEED) for key in ['identity_r1','hard_r1','focus_identity_r1','target_rank']};result[k]['objects']=len({(r['map_name'],r['target_object_id']) for r in selected if r['target_visible']})
    return result


def confidence(rows,p2,out,protocol):
    result={}
    for model in ['city_only','aerial_city','external_aerial']:
        predictions=torch.load(p2/'calibration_dev'/(model+'_scores.pt'),weights_only=True);cfg=json.loads((p2/'calibration_dev/frozen_config.json').read_text())['configs'][model];top=[];alltri=[];alltarget=[];raw=[]
        for r,p in zip(rows,predictions):
            n=len(p['score']);assert n==len(r['public']['roi_valid'])
            if not n:continue
            gt=torch.full((n,),2,dtype=torch.long);gt[r['labels']['semantic_valid_mask']&~r['labels']['positive_roi_mask']]=1;gt[r['labels']['positive_roi_mask']]=0
            tri=p['tri'].numpy();prob=torch.softmax(p['tri']/cfg['temperature'],-1).numpy();alltri.append(tri);alltarget.extend(gt.tolist());best=int(p['score'].argmax());sprob=p['score'].softmax(-1).numpy();gap=float(p['score'].topk(2).values.diff().neg()[0]) if n>=2 else 0
            top.append(dict(sample_id=r['metadata']['sample_id'],family_id=r['metadata']['family_id'],map_name=r['metadata']['map_name'],eligible=n>=2 and r['metadata']['roi_objects'][best]['depth_support']>=.9,correct=bool(r['labels']['identity_positive'][best]),target_visible=bool(r['labels']['identity_positive'].any()),match=float(prob[best,0]),match_vs_mismatch=float(torch.sigmoid(p['tri'][best,0]-p['tri'][best,1])),rank_softmax=float(sprob[best]),score_gap_sigmoid=float(torch.sigmoid(torch.tensor(gap))),raw_match_logit=float(tri[best,0]),raw_match_unknown_margin=float(tri[best,0]-tri[best,2])))
            raw.append(dict(sample_id=r['metadata']['sample_id'],family_id=r['metadata']['family_id'],map_name=r['metadata']['map_name'],candidate_object_ids=[v['object_id'] for v in r['metadata']['roi_objects']],raw_logits=tri.tolist(),labels=gt.tolist(),temperature=cfg['temperature']))
        tri=np.concatenate(alltri);target=np.array(alltarget);prob=torch.softmax(torch.tensor(tri)/cfg['temperature'],-1).numpy();known=target!=2;true=target[known]==0;confusion=np.zeros((3,3),int)
        for y,z in zip(target,prob.argmax(-1)):confusion[y,z]+=1
        metrics=dict(original_config=cfg,weak_class_counts=np.bincount(target,minlength=3).tolist(),confusion=confusion.tolist(),multiclass_brier=float(np.mean(np.sum((prob-np.eye(3)[target])**2,axis=1))),multiclass_confidence_ece=reliability(target==prob.argmax(-1),prob.max(-1)),classwise={str(c):reliability(target==c,prob[:,c]) for c in range(3)},known_match_mismatch_auc={k:auc(true,values) for k,values in dict(raw_match=tri[known,0],raw_match_minus_mismatch=tri[known,0]-tri[known,1],raw_match_minus_unknown=tri[known,0]-tri[known,2],calibrated_match=prob[known,0]).items()},raw_logit_quantiles={str(c):np.quantile(tri[target==c],[0,.25,.5,.75,1],axis=0).tolist() for c in range(3) if np.any(target==c)},top_candidate={},thresholds={},scope='Internal Dev only, reused original raw logits/temperature; class labels weak; target-absent means wrong acceptance, never training mismatch labels')
        chosen=[r for r in top if r['eligible']]
        for key in ['match','match_vs_mismatch','rank_softmax','score_gap_sigmoid']:
            metrics['top_candidate'][key]=reliability([r['correct'] for r in chosen],[r[key] for r in chosen]);metrics['thresholds'][key]=risk_curve(top,key,protocol['confidence_thresholds'])
        result[model]=metrics;save_rows(out/(model+'_confidence_dev.jsonl.gz'),top);save_rows(out/(model+'_raw_logits_dev.jsonl.gz'),raw)
    write(out/'confidence_dev.json',result);return result


def residual_records(rows,predictions,amplitude,oracle=False,credible_oracle=False):
    result=[]
    for r,p in zip(rows,predictions):
        labels=offline_labels(r);n=len(p['score']);w=r['state']['_association'];selected=None
        if oracle:
            ids=np.flatnonzero(labels['positive']);selected=int(ids[0]) if len(ids) and (not credible_oracle or r['labels']['target_credible']) else None
        elif n:
            best=int(p['score'].argmax());selected=best if r['metadata']['roi_objects'][best]['depth_support']>=.9 else None
        updated,res=offline_residual(labels['static'],w,selected,amplitude,offline_only=True);rank=np.argsort(updated,kind='stable')[::-1];before=not labels['wrong'];after=labels['distance'][rank[0]]<=20;correct_selected=selected is not None and bool(labels['positive'][selected]);meta=r['metadata']
        result.append(dict(sample_id=meta['sample_id'],family_id=meta['family_id'],map_name=meta['map_name'],cutoff_step=meta['cutoff_step'],candidate_object_ids=[v['object_id'] for v in meta['roi_objects']],metrics=dict(goal_r1=float(after),goal_r4=float((labels['distance'][rank[:4]]<=20).any()),goal_r8=float((labels['distance'][rank[:8]]<=20).any()),ne=float(labels['distance'][rank[0]]),recovery=float(not before and after),damage=float(before and not after),net_recovery=float(after)-float(before),identity_recovery=float(not before and after and correct_selected),incidental_recovery=float(not before and after and not correct_selected)),selected_index=selected,correct_selected=correct_selected,static_top1=int(np.argsort(labels['static'],kind='stable')[-1]),theory_top1=int(rank[0]),max_residual=float(res.max()),amplitude=amplitude,offline_only=True,confidence_bypassed=not oracle,oracle=oracle,credible_oracle=credible_oracle))
    return result


def margin_records(rows,raw):
    result=[]
    for r,p in zip(rows,raw):
        lab=offline_labels(r);w=r['state']['_association'];support=(w[lab['positive']]>0).any(0) if lab['visible'] else np.zeros(784,bool);good=lab['distance']<=20;good_support=support&good;s=lab['static'];pos=lab['positive'];score=p['score'].numpy();n=len(score)
        gap=float(s.max()-s[support].max()) if support.any() else None;required=float(s.max()-s[good_support].max()) if good_support.any() else None
        weakattrs=r['labels']['attribute_positive'].numpy()&r['labels']['attribute_judged'].numpy();m=r['metadata']
        result.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],target_object_id=r['sealed']['target_object_id'],wrong=lab['wrong'],visible=lab['visible'],credible_error=lab['focus'],associated_centroid_margin=gap,associated_correct_cell_margin=required,gt_correct_grid_margin=float(s.max()-s[good].max()),supported_target_cell_count=int(support.sum()),supported_cell_within_success_radius=bool(good_support.any()),rgb_identity_margin=float(score[pos].max()-score[~pos].max()) if pos.any() and (~pos).any() else None,attribute_multi_positive_count=int(weakattrs.sum()),target_native_side=next((v['native_min_side'] for v,z in zip(m['roi_objects'],pos) if z),None)))
    return result


def swaps(rows,original,changed):
    records=[]
    for r,p,q in zip(rows,original,changed):
        spec=query_spec(r['metadata']['instruction']);sw=spec['swap'];n=len(p['score'])
        if not sw or n<2:continue
        regs=r['metadata']['roi_objects'];field=sw['field'];typ=spec['typ'];known=np.array([v['object_type']==typ and all(k in v['weak_attrs'] for k in sw['attributes']) for v in regs]);before=np.array([all(v['weak_attrs'].get(k)==value for k,value in spec['attributes'].items()) for v in regs])&known;after=np.array([all(v['weak_attrs'].get(k)==value for k,value in sw['attributes'].items()) for v in regs])&known
        if not before.any() or not after.any():continue
        ids=np.flatnonzero(known);a=max(ids,key=lambda j:float(p['score'][j]));b=max(ids,key=lambda j:float(q['score'][j]));d=q['score'].numpy()-p['score'].numpy();direction=float(d[after].mean()-d[before].mean());m=r['metadata']
        records.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],phrase=spec['phrase'],swapped_phrase=sw['phrase'],field=field,multi_positive_before=np.flatnonzero(before).tolist(),multi_positive_after=np.flatnonzero(after).tolist(),judged_indices=ids.tolist(),scores_original=p['score'].tolist(),scores_swapped=q['score'].tolist(),metrics=dict(original_attribute_correct=float(before[a]),swapped_attribute_correct=float(after[b]),counterfactual_both_correct=float(before[a] and after[b]),score_direction_correct=float(direction>0),score_direction=direction),label_status='language-derived weak attributes; no unique-target or human visual truth claim'))
    return records


def run(a):
    out=Path(a.output);assert not out.exists();out.mkdir(parents=True);started=time.time();p2=Path(a.p2);protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());frozen=json.loads((p2/'calibration_dev/frozen_config.json').read_text());cache=torch.load(Path(a.text)/'text_features.pt',weights_only=True);texts=cache['texts'];torch.manual_seed(SEED);torch.set_num_threads(2)
    lock=Path(a.lock) if a.lock else None
    if a.split=='seen':
        assert lock and lock.exists();binding=json.loads(lock.read_text());assert binding['protocol_sha256']==file_sha256(Path(__file__).with_name('protocol.json'))
        for name,sha in binding['source_sha256'].items():assert file_sha256(Path(__file__).with_name(name))==sha
    for name in protocol['models']:
        path=frozen['heads'][name]
        if path!='native':assert file_sha256(path)==frozen['checkpoint_sha256'][name]
    rows=city_rows(p2/('features_'+a.split));bind_contours(rows);labels=[offline_labels(r) for r in rows];relations=[public_relations(r['metadata']['instruction'],r['metadata']['roi_objects']) for r in rows];geometry=[]
    for r,lab,rel in zip(rows,labels,relations):lab.update(phrase_supported=cache['specs'][r['metadata']['instruction']]['supported'],relation_available=bool(rel[2]) and bool(rel[1].any()))
    for r in rows:
        w=r['state']['_association'];s=np.array(r['state']['public']['static_logits']);geometry.append(np.array([s[v>0].max() if (v>0).any() else s.min() for v in w]))
    confidence_metrics=confidence(rows,p2,out,protocol) if a.split=='dev' else None;metrics={};groupstats={};allrecords={};allraw={};swapmetrics={}
    for name in protocol['models']:
        model=load_model(frozen['heads'][name]);raw={}
        for control in protocol['controls']:
            for mode in ['full','attribute']:
                raw[(control,mode)]=frozen_predict(model,rows,texts,mode,control)
            for mode in protocol['language']:
                source=raw[(control,'full' if mode=='full' else 'attribute')]
                for modality in ['rgb','combined']:
                    key='__'.join([name,mode,modality,control]);records=[metric_record(r,compose_scores(p['score'].numpy(),g,rel[0],mode,modality,bool(rel[2])),mode,lab) for r,p,g,rel,lab in zip(rows,source,geometry,relations,labels)];metrics[key]=summary(records);groupstats[key]=grouped(records);allrecords[key]=records;save_rows(out/(key+'_identity.jsonl.gz'),records)
            if control=='real':
                changed=frozen_predict(model,rows,texts,'swap',control);sr=swaps(rows,raw[(control,'attribute')],changed);save_rows(out/(name+'_language_swap.jsonl.gz'),sr);swapmetrics[name]={key:summarize_rows(sr,key,seed=SEED) for key in ['original_attribute_correct','swapped_attribute_correct','counterfactual_both_correct','score_direction_correct','score_direction']}
        allraw[name]=raw;torch.save(raw,out/(name+'_raw_scores.pt'));del model;torch.cuda.empty_cache();print(json.dumps(dict(split=a.split,model=name,seconds=time.time()-started)),flush=True)
    geometry_records=[metric_record(r,g,'full',lab) for r,g,lab in zip(rows,geometry,labels)];metrics['geometry']=summary(geometry_records);groupstats['geometry']=grouped(geometry_records);save_rows(out/'geometry_identity.jsonl.gz',geometry_records)
    principal=protocol['principal_model'];reference=allrecords[principal+'__full__rgb__real'];paired={}
    for name in [principal+'__full__rgb__'+c for c in ['zero','shuffled']]+['native__full__rgb__real',principal+'__attribute__rgb__real',principal+'__attribute_spatial__rgb__real',principal+'__full__combined__real']:
        other=allrecords[name];paired[name]={key:paired_difference(other,reference,key,seed=SEED) for key in ['identity_r1','focus_identity_r1','hard_r1','supported_phrase_identity_r1','supported_relation_identity_r1']};paired[name]['identity_family_equal']=paired_difference(family_rows(other,'identity_r1'),family_rows(reference,'identity_r1'),'identity_r1',seed=SEED)
    residual={};respaired={};base=None
    for policy in [principal+'_real',principal+'_zero',principal+'_shuffled','native_real','external_aerial_real','identity_oracle','credible_identity_oracle']:
        oracle='oracle' in policy;name,control=(principal,'real') if oracle else policy.rsplit('_',1);pred=allraw[name][(control,'full')]
        for amplitude in protocol['offline_amplitudes']:
            key=policy+'__'+str(amplitude);records=residual_records(rows,pred,amplitude,oracle=oracle,credible_oracle=policy=='credible_identity_oracle');residual[key]=summary(records);save_rows(out/(key+'_residual.jsonl.gz'),records)
            if base is None:base=records
            respaired[key]={k:paired_difference(base,records,k,seed=SEED) for k in ['goal_r1','recovery','damage','net_recovery']}
    principal_raw=allraw[principal][('real','full')];margins=margin_records(rows,principal_raw);save_rows(out/'margins.jsonl.gz',margins);credible=[r for r in margins if r['credible_error']];visible_errors=[r for r in margins if r['wrong'] and r['visible']];coverage=dict(states=len(rows),families=len({r['metadata']['family_id'] for r in rows}),maps=len({r['metadata']['map_name'] for r in rows}),credible_error_states=len(credible),credible_error_families=len({r['family_id'] for r in credible}),credible_error_objects=len({(r['map_name'],r['target_object_id']) for r in credible}),visible_error_states=len(visible_errors),visible_error_families=len({r['family_id'] for r in visible_errors}),relation_states=sum(bool(x[2]) for x in relations),relation_decidable_candidate_states=sum(bool(x[1].any()) for x in relations),supported_correct_error_states=sum(r['supported_cell_within_success_radius'] for r in credible),margin_le_025_credible_errors=sum(r['associated_correct_cell_margin'] is not None and r['associated_correct_cell_margin']<.025 for r in credible),weak_attribute_ambiguous_states=sum(r['attribute_multi_positive_count']>=2 for r in margins),human_verified_semantic_cases=0)
    margin_summary={}
    for subset,values in [('credible_errors',credible),('all_visible_errors',visible_errors)]:
        margin_summary[subset]={}
        for key in ['associated_centroid_margin','associated_correct_cell_margin','gt_correct_grid_margin','rgb_identity_margin']:
            x=[r[key] for r in values if r[key] is not None];margin_summary[subset][key]=dict(n=len(x),quantiles=np.quantile(x,[0,.1,.25,.5,.75,.9,1]).tolist() if x else None)
    write(out/'metrics.json',dict(identity=metrics,groups=groupstats,paired=paired,language_swap=swapmetrics,residual=residual,residual_paired=respaired,coverage=coverage,margin_summary=margin_summary,seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),seed=SEED,protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json')),checkpoint_sha256={n:frozen['checkpoint_sha256'].get(n) for n in protocol['models']},original_gain_unchanged=.025,formal_threshold_unchanged=frozen['configs'][principal]['threshold'],training_steps=0,image_encodings=0,split=a.split,confidence_analyzed=a.split=='dev',navigation_rollouts=0,unseen_runs=0))
    write(out/'relations.json',[dict(sample_id=r['metadata']['sample_id'],relations=rel[2],scores=rel[0].tolist(),decidable=rel[1].tolist()) for r,rel in zip(rows,relations)])
    if a.split=='dev':write(out/'seen_lock.json',dict(protocol_sha256=file_sha256(Path(__file__).with_name('protocol.json')),source_sha256={p.name:file_sha256(p) for p in Path(__file__).parent.glob('*.py') if p.name in ['run.py','core.py']},checkpoint_sha256=frozen['checkpoint_sha256'],text_sha256=file_sha256(Path(a.text)/'text_features.pt'),selection='No model or threshold chosen in P2.5; all declared diagnostic arms retained. Principal city_only inherited P2. Seen curves offline only.'))


def main():
    p=argparse.ArgumentParser();s=p.add_subparsers(dest='cmd',required=True);t=s.add_parser('text');t.add_argument('--p2',required=True);t.add_argument('--output',required=True);r=s.add_parser('diagnose');r.add_argument('--p2',required=True);r.add_argument('--text',required=True);r.add_argument('--split',choices=['dev','seen'],required=True);r.add_argument('--lock');r.add_argument('--output',required=True);a=p.parse_args();(text_cache if a.cmd=='text' else run)(a)
if __name__=='__main__':main()
