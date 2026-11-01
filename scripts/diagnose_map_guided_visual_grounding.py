"""Frozen current-frame map/RGB diagnostic with instance labels isolated to evaluation."""
import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoTokenizer

from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.map_guided_grounding import (project_polygon,public_landmarks,target_word_mask,
    crop_reference,reference_crop_bounds,reference_caption,reference_candidates,geometry_scores,
    permute_visual_correspondence,score_candidates,registered_instance_evaluation)
from graphnav.native_object_runtime import NativeObjectEncoder
from graphnav.landmark_projection import frame_to_world
from graphnav.workspace_paths import WORKSPACE
from graphnav.object_proposals import select_proposals
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.semantic_geometry import canonical
from scripts.train_object_query_belief import load
from scripts.train_native_gated_fusion import split_train_by_group

TOKENIZER='/home/20261024_1/.cache/huggingface/hub/models--bert-base-uncased/snapshots/86b5e0934494bd15c9632b12f734a8a67f723594'


def raw_lookup(manifest,partition,aerialvg_sha):
    lookup={}
    for source in manifest['sources']:
        directory=Path(source['path']);p=directory/'manifest.json';m=json.loads(p.read_text())
        if m['partition']!=partition or m['status']!='complete' or file_sha256(p)!=source['manifest_sha'] or m['official_checkpoint_sha']!=aerialvg_sha:
            raise ValueError('Raw cache partition/hash mismatch')
        for entry in m['frames']:lookup.setdefault((entry['episode'],entry['step']),directory/(entry['key']+'.pt'))
    return lookup


def projections_for(frame,landmarks):
    return [project_polygon(e['geometry_m'],frame['footprint_world'].numpy(),frame['depth_valid'].shape,frame['depth_valid'].numpy()) for e in landmarks]


def aligned_boxes(frame,row,device):
    f=frame['all_query_features'].to(device).float();f[frame['query_ids'].to(device)]=frame['query_features'].to(device).float()
    text=frame['encoded_text'][0].to(device).float();mask=frame['text_token_mask'][0].to(device).bool()
    score=(f@text.T)[:,mask].sigmoid().amax(-1)
    centers=frame['all_world_centers'].to(device).float();grid=row['grid_world'].to(device).float();base=row['base_logits'].to(device).float()
    nearest=torch.cat([torch.cdist(grid,part).argmin(0) for part in centers.split(64)])
    ids=select_proposals(frame['all_boxes_cxcywh'].to(device).float(),score,frame['all_valid_pixel_fraction'].to(device).float(),15,base[nearest],True)
    count=int(row['valid'].sum())
    if count!=len(ids) or not torch.allclose(centers[ids].cpu(),row['query_world'][:count],rtol=0,atol=1e-4):
        raise ValueError('Frozen candidate alignment failed')
    if not torch.equal(f[ids].half().cpu(),row['features'][:count]):raise ValueError('Frozen feature alignment failed')
    boxes=torch.zeros(15,4);boxes[:count]=frame['all_boxes_cxcywh'][ids.cpu()]
    return boxes


class ReferenceExtractor:
    """Encode actual current polygon crops using the SAME frozen AerialVG path."""
    def __init__(self,directory,tokenizer):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.encoder=None;self.tokenizer=tokenizer;self.calls=0

    @torch.inference_mode()
    def extract(self,bgr,landmark,projection,parent_footprint,parent_depth):
        caption=reference_caption(landmark)
        crop=crop_reference(bgr,projection)
        x0,y0,x1,y1=reference_crop_bounds(bgr.shape,projection)
        crop_depth=np.asarray(parent_depth,bool)[y0:y1,x0:x1].copy()
        crop_corners=frame_to_world(np.array([[x0,y0],[x1-1,y0],[x1-1,y1-1],[x0,y1-1]],np.float32),parent_footprint,bgr.shape)
        key=hashlib.sha256(crop.tobytes()+crop_depth.tobytes()+caption.encode()).hexdigest();path=self.directory/(key+'.pt')
        if path.exists():return torch.load(path,weights_only=True),0.,0.,True
        if self.encoder is None:self.encoder=NativeObjectEncoder('cuda')
        # Current crop only; no global image or target annotations. Proxy validity
        # was already checked on the exact projected parent polygon.
        observation=dict(bgr=crop,depth_valid=crop_depth,footprint_world=crop_corners)
        torch.cuda.synchronize();baseline=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
        encoded=self.encoder.encode(observation,caption)
        language_mask=target_word_mask(caption.split(' called ')[0]+' called '+landmark['name'],self.tokenizer,len(encoded['text']),encoded['text_mask'].cpu())
        # Only object-type words, not a presumed learned identity for a proper name.
        tokens=self.tokenizer(caption.lower().strip().rstrip('.')+'.',return_offsets_mapping=True)
        cutoff=caption.index(' called ')
        language_mask &= torch.tensor([stop>start and stop<=cutoff for start,stop in tokens['offset_mapping']])
        if not language_mask.any():raise ValueError('Reference type language empty')
        text=encoded['text'][language_mask.to(encoded['text'].device)].mean(0)
        scores=encoded['features']@text
        scores=scores.masked_fill(encoded['quality']<.6,-1e4);index=int(scores.argmax())
        supported=bool((encoded['quality']>=.6).any())
        feature=encoded['features'][index].cpu().float() if supported else torch.zeros(256)
        result=dict(feature=feature,language=text.cpu().float(),caption=caption,supported=supported,
                    selected_box=encoded['boxes'][index].cpu(),crop_shape=list(crop.shape),score=float(scores[index]) if supported else None)
        torch.cuda.synchronize();duration=time.perf_counter()-start;memory=(torch.cuda.max_memory_allocated()-baseline)/(1024**2)
        atomic_save(result,path);self.calls+=1
        return result,duration,memory,False


def summarize(records):
    result={}
    for label in ['all','target_projected','reference_only_projected','neither_projected','unknown']:
        rows=records if label=='all' else [r for r in records if r['visibility']==label]
        n=len(rows);methods={}
        for name in ['A0','native','map_only','map_rgb','shuffled_rgb']:
            values=[r['methods'][name] for r in rows]
            methods[name]=dict(states=n,R1=sum(v['error']<=20 for v in values)/n if n else None,
                NE=sum(v['error'] for v in values)/n if n else None,
                corrected_A0=sum(r['methods']['A0']['error']>20 and r['methods'][name]['error']<=20 for r in rows),
                damaged_A0=sum(r['methods']['A0']['error']<=20 and r['methods'][name]['error']>20 for r in rows),
                corrected_native=sum(r['methods']['native']['error']>20 and r['methods'][name]['error']<=20 for r in rows),
                damaged_native=sum(r['methods']['native']['error']<=20 and r['methods'][name]['error']>20 for r in rows))
            instances=[r['instance'][name] for r in rows if name in r['instance'] and r['instance'][name]['eligible']]
            methods[name]['instance']=dict(eligible=len(instances),**{k:sum(v[k] for v in instances)/len(instances) if instances else None for k in ['top1','top5','top15']})
        result[label]=dict(states=n,methods=methods)
    projection=Counter()
    for row in records:projection.update(row['projection_counts'])
    times=np.array([r['seconds'] for r in records]);infer=np.array([r['reference_seconds'] for r in records])
    result['projection']=dict(projection)
    result['resource']=dict(mean_new_scoring_seconds=float(times.mean()),p50=float(np.median(times)),p95=float(np.quantile(times,.95)),
        reference_inference_seconds=float(infer.sum()),mean_reference_seconds=float(infer.mean()),
        peak_incremental_reference_activation_MiB=max(r['memory_mib'] for r in records),
        reference_encoded_calls=sum(r['reference_encoded'] for r in records),
        reference_cache_hits=sum(r['reference_cache_hits'] for r in records))
    result['projection_max_roundtrip_m']=max(r['roundtrip_m'] for r in records)
    result['human_projection_error']=None
    result['limitations']='Geometric support strata; registered official instance contours, not semantic visible masks; no human labels; current-only; conditional fixed AerialVG proposals.'
    return result


def make_figure(row,frame,bgr,landmarks,projections,boxes,record,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from graphnav.landmark_projection import world_to_frame
    fig,axes=plt.subplots(2,3,figsize=(15,9))
    axes[0,0].set_title('Public vector map + CURRENT footprint')
    for e in landmarks:
        xy=e['geometry_m'];axes[0,0].plot(*np.vstack([xy,xy[0]]).T,color='gray',lw=.7)
    footprint=frame['footprint_world'].numpy();axes[0,0].plot(*np.vstack([footprint,footprint[0]]).T,color='red')
    axes[0,0].scatter(*row['query_world'][row['valid']].numpy().T,s=8,color='blue')
    axes[0,0].set_aspect('equal');axes[0,0].set_xlabel('world east (m)');axes[0,0].set_ylabel('world north (m)')
    lo,hi=footprint.min(0),footprint.max(0);span=max(hi-lo);axes[0,0].set_xlim(lo[0]-.25*span,hi[0]+.25*span);axes[0,0].set_ylim(lo[1]-.25*span,hi[1]+.25*span)
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);axes[0,1].imshow(rgb);axes[0,1].set_title('Actual cached RGB (not global orthophoto)')
    axes[0,2].imshow(rgb);axes[0,2].set_title('Map projections + fixed15 boxes; gold=EVAL instance')
    for e,p in zip(landmarks,projections):
        if p['status'] in ['full','partial']:
            xy=np.asarray(p['contour_px']);axes[0,2].plot(*np.vstack([xy,xy[0]]).T,color='lime',lw=1)
    h,w=bgr.shape[:2]
    for i,box in enumerate(boxes):
        if not row['valid'][i]:continue
        cx,cy,bw,bh=box.numpy()*np.array([w,h,w,h]);rect=plt.Rectangle((cx-bw/2,cy-bh/2),bw,bh,fill=False,color='dodgerblue',lw=.7)
        axes[0,2].add_patch(rect);axes[0,2].text(cx,cy,str(i),fontsize=6,color='yellow')
    if record['target_eval_bbox']:
        x0,y0,x1,y1=record['target_eval_bbox'];axes[0,2].add_patch(plt.Rectangle((x0,y0),x1-x0,y1-y0,fill=False,color='gold',lw=2))
    for ax in [axes[0,1],axes[0,2]]:ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis('off')
    visible=[i for i,p in enumerate(projections) if p['valid_roi'] and i in record['mentioned_indices']]
    for ax,index in zip(axes[1,:2],visible[:2]):
        ax.imshow(cv2.cvtColor(crop_reference(bgr,projections[index]),cv2.COLOR_BGR2RGB));ax.set_title(landmarks[index]['name']+'\n'+projections[index]['status']);ax.axis('off')
    for ax in axes[1,:2][len(visible[:2]):]:ax.text(.1,.5,'No supported named reference crop');ax.axis('off')
    axes[1,2].axis('off');axes[1,2].text(0,1,'\n'.join([record['visibility']]+[f"{k}: NE={v['error']:.1f}m rank={v.get('rank')}" for k,v in record['methods'].items()]),va='top')
    fig.suptitle(f"TRAIN ep={row['episode']} step={row['step']}  {row['instruction'][:180]}",fontsize=10)
    fig.tight_layout();fig.savefig(path,dpi=115);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['cache','native-checkpoint','catalog','train-annotations','output']:parser.add_argument('--'+name,required=True)
    parser.add_argument('--partition',choices=['train','dev'],required=True)
    parser.add_argument('--cohort',choices=['fit','calibration'],default='fit')
    parser.add_argument('--limit',type=int,default=0);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--freeze-protocol');parser.add_argument('--figures',type=int,default=12)
    a=parser.parse_args();out=Path(a.output)
    meta=json.loads((Path(a.cache)/'manifest.json').read_text())
    if meta['partition']!=a.partition or not meta['complete']:raise ValueError('Explicit TRAIN/internal Dev only')
    if 'citynav_train_seen.json'!=Path(a.train_annotations).name:raise ValueError('No formal annotation split permitted')
    if a.partition=='dev' and not a.freeze_protocol:raise ValueError('Freeze TRAIN design first')
    if out.exists() and not a.resume:raise ValueError('Use a new output')
    out.mkdir(parents=True,exist_ok=True);(out/'figures').mkdir(exist_ok=True)
    native_sha=file_sha256(a.native_checkpoint)
    aerialvg_path=WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth'
    aerialvg_sha=file_sha256(aerialvg_path)
    protocol=dict(kind='map-guided-frozen-current-v1',partition=a.partition,native_sha=native_sha,
        aerialvg_sha=aerialvg_sha,cohort=a.cohort,cache_manifest_sha=file_sha256(Path(a.cache)/'manifest.json'),catalog_sha=file_sha256(a.catalog),
        source_sha={p:file_sha256(p) for p in [__file__,'graphnav/map_guided_grounding.py']},
        no_training=True,constants=dict(reference_budget=2,appearance_weight=.5,rule_scale=10.,kernel_sigma=8.,bound=2.,null=-10.),
        shuffle='reverse joint current target-query/reference-crop feature vectors; boxes/map/pose/language fixed',
        instance_eval='official object_id+contour registered bbox IoU>=0.5; >=64pixels/80%inview/60%depth support',
        public_depth='depth_valid only; metric depth/K/R/t/physical occlusion unavailable',limit=a.limit)
    if a.freeze_protocol:
        freeze=json.loads(Path(a.freeze_protocol).read_text())
        if freeze['source_sha']!=protocol['source_sha'] or freeze['constants']!=protocol['constants']:raise ValueError('Design changed after TRAIN freeze')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Resume protocol mismatch')
    atomic_json(protocol,out/'protocol.json')
    _,rows=load(a.cache);lookup=raw_lookup(meta,a.partition,aerialvg_sha)
    if a.partition=='train':
        fit,cal=split_train_by_group(rows)
        rows=sorted(fit if a.cohort=='fit' else cal,key=lambda r:hashlib.sha256(f"{r['group']}:{r['step']}".encode()).digest())
        if a.limit:rows=rows[:a.limit]
    elif len(rows)!=635:raise ValueError('Expected the same635 internal Dev states')
    # Public named objects are sanitized before inference. Private target annotations
    # are not used until every hypothesis has been computed below.
    catalog=json.loads(Path(a.catalog).read_text());named={m:public_landmarks(catalog,m) for m in {r['map_name'] for r in rows}}
    annotations=json.loads(Path(a.train_annotations).read_text())
    tokenizer=AutoTokenizer.from_pretrained(TOKENIZER,local_files_only=True,use_fast=True)
    extractor=ReferenceExtractor(out/'reference_features',tokenizer)
    device=torch.device('cuda');state=torch.load(a.native_checkpoint,map_location='cpu',weights_only=True)
    native=ObjectQueryRanker().to(device);native.load_state_dict(state['model']);native.eval().requires_grad_(False)
    records=[]
    if (out/'records.jsonl').exists():records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines() if line.strip()]
    done={(r['episode'],r['step']) for r in records};figure_counts=Counter()
    for r in records:figure_counts[r['visibility']]+=int(bool(r.get('figure')))
    for number,row in enumerate(rows):
        if (row['episode'],row['step']) in done:continue
        rawpath=lookup[(row['episode'],row['step'])];frame=torch.load(rawpath,weights_only=True)
        if (frame['map_name'],frame['instruction'],frame['step'])!=(row['map_name'],row['instruction'],row['step']):raise ValueError('Raw frame identity mismatch')
        bgr=cv2.imread(str(rawpath.with_suffix('.png')))
        if bgr is None or bgr.shape[:2]!=tuple(frame['depth_valid'].shape):raise ValueError('Missing/misaligned actual current RGB')
        boxes=aligned_boxes(frame,row,device)
        landmarks=named[row['map_name']];start=time.perf_counter();projections=projections_for(frame,landmarks)
        entities,relations,mentioned,reference_ids=reference_candidates(landmarks,projections,row['instruction'])
        pcounts=Counter(all_landmarks=len(projections),projection_success=sum(p['status']!='unknown' for p in projections),
            valid_rois=sum(p['valid_roi'] for p in projections),full=sum(p['status']=='full' for p in projections),
            partial=sum(p['status']=='partial' for p in projections),outside=sum(p['status']=='outside' for p in projections),
            mentioned=len(mentioned),mentioned_valid=sum(projections[i]['valid_roi'] for i in mentioned))
        ref_features=[];ref_language=[];ref_seconds=memory=0.;encoded_count=cache_hits=0
        for index in reference_ids:
            result,duration,peak,hit=extractor.extract(bgr,landmarks[index],projections[index],frame['footprint_world'].numpy(),frame['depth_valid'].numpy())
            ref_features.append(result['feature']);ref_language.append(result['language'])
            ref_seconds+=duration;memory=max(memory,peak);encoded_count+=not hit;cache_hits+=hit
        reference_features=torch.stack(ref_features) if ref_features else torch.empty(0,256)
        reference_language=torch.stack(ref_language) if ref_language else torch.empty(0,256)
        appearance_mask=target_word_mask(row['instruction'],tokenizer,len(row['text']),row['text_mask'])
        language=row['text'].float()[appearance_mask].mean(0) if appearance_mask.any() else None
        features=row['features'].float();valid=row['valid'].bool();centers=row['query_world'].float()
        prior_values=row['base_logits'][torch.cdist(row['grid_world'].float(),centers).argmin(0)]
        prior,fields=geometry_scores(centers.numpy(),prior_values,entities,relations)
        mapq,rgbq,confidence=score_candidates(prior,fields,relations,features,language,reference_features,reference_language,reference_ids,valid)
        shuffled,shuffled_refs,permutation=permute_visual_correspondence(features,reference_features,valid)
        _,shufq,_=score_candidates(prior,fields,relations,shuffled,language,shuffled_refs,reference_language,reference_ids,valid)
        with torch.inference_mode():
            f=row['features'][None].to(device).float();text=row['text'][None].to(device).float();tm=row['text_mask'][None].to(device)
            geometry=row['geometry'][None].to(device).float();v=valid[None].to(device)
            with torch.autocast('cuda',dtype=torch.bfloat16):q0,null=native(f,text,tm,geometry,v)
            grid=row['grid_world'][None].to(device).float();base=row['base_logits'][None].to(device).float();observed=row['observed'][None].to(device)
            kernel=torch.exp(-torch.cdist(grid,centers[None].to(device)).square()/(2*8.**2))*observed[:,:,None]*v[:,None]
            outputs={'A0':base[0].cpu()};ranks={'native':q0[0].float().cpu(),'map_only':mapq,'map_rgb':rgbq,'shuffled_rgb':shufq}
            for name,q in ranks.items():
                current_null=null if name=='native' else torch.full_like(null,-10.)
                outputs[name]=native.update_belief(base,q[None].to(device),current_null,kernel,observed,v,bound=2.)[0][0].cpu()
        seconds=time.perf_counter()-start
        # -------- Private evaluation labels FIRST enter here, after inference. --------
        source=annotations[row['episode']]
        if source['descriptions'][0]!=row['instruction']:raise ValueError('Official TRAIN instance identity mismatch')
        target_object=catalog[row['map_name']][str(source['object_ids'][0])]
        target_contour=target_object['contour']
        target_projection=project_polygon(target_contour,frame['footprint_world'].numpy(),bgr.shape,frame['depth_valid'].numpy())
        target_supported=target_projection['pixels']>=16 and target_projection['depth_support']>=.6 and target_projection['coverage']>=.1
        reference_supported=any(projections[i]['valid_roi'] for i in mentioned)
        visibility='unknown' if target_projection['status']=='unknown' else 'target_projected' if target_supported else 'reference_only_projected' if reference_supported else 'neither_projected'
        distances=(row['grid_world'].float()-row['labels']['target_world'][:2].float()).norm(dim=-1)
        instance={name:registered_instance_evaluation(target_contour,frame['footprint_world'].numpy(),bgr.shape,frame['depth_valid'].numpy(),boxes,q) for name,q in ranks.items()}
        # Keep eligible IoU vectors and official identity; do not persist pseudo-instance labels.
        methods={name:dict(error=float(distances[value.argmax()]),rank=int(ranks[name].argmax()) if name in ranks else None) for name,value in outputs.items()}
        record=dict(episode=row['episode'],step=row['step'],group=row['group'],map=row['map_name'],instruction=row['instruction'],
            visibility=visibility,visibility_is_geometric=True,reference_ids=[landmarks[i]['key'] for i in reference_ids],mentioned_indices=mentioned,
            official_instance_id=str(source['object_ids'][0]),target_eval_bbox=target_projection['bbox'],methods=methods,instance=instance,
            projection_counts=dict(pcounts),roundtrip_m=max([p['roundtrip_error_m'] or 0 for p in projections]+[0.]),
            footprint_center_error_m=float(np.linalg.norm(frame['footprint_world'].numpy().mean(0)-frame['pose'].numpy()[:2])),
            seconds=seconds,reference_seconds=ref_seconds,memory_mib=memory,reference_encoded=encoded_count,reference_cache_hits=cache_hits,
            raw_frame=str(rawpath),image_sha=file_sha256(rawpath.with_suffix('.png')),feature_permutation=permutation.tolist(),appearance_tokens=int(appearance_mask.sum()),figure=None)
        # TRAIN only review figures; stratify outcomes within each visibility cohort.
        success=methods['map_rgb']['error']<=20;figure_key=visibility+('_success' if success else '_failure')
        if a.partition=='train' and sum(figure_counts.values())<a.figures and figure_counts[figure_key]<2:
            name=f"{row['episode']:05d}_{row['step']:03d}.png";make_figure(row,frame,bgr,landmarks,projections,boxes,record,out/'figures'/name)
            record['figure']='figures/'+name;figure_counts[figure_key]+=1
        records.append(record)
        with (out/'records.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
        atomic_json(dict(processed=len(records),expected=len(rows),last_episode=row['episode'],last_step=row['step']),out/'progress.json')
        if len(records)%10==0:print(json.dumps(dict(processed=len(records),references=extractor.calls)),flush=True)
    if file_sha256(a.native_checkpoint)!=native_sha or file_sha256(aerialvg_path)!=aerialvg_sha:raise ValueError('Frozen Native/AerialVG changed')
    report=summarize(records);atomic_json(report,out/'summary.json')
    atomic_json(dict(complete=True,records=len(records),no_training=True,native_sha_verified=True,aerialvg_sha_verified=True),out/'complete.json')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
