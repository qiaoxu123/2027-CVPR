"""Read-only Train/internal-Dev query selection, oracle and frozen fusion diagnostics."""
import argparse,collections,csv,hashlib,json
from pathlib import Path
import cv2,numpy as np,torch
from torchvision.ops import nms,box_iou
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.train_object_query_belief import load


def selection_stages(boxes,scores,quality,geometry,guided=False,spatial=False):
    """Public inputs only; identical deployed validity/NMS/score constants."""
    valid=torch.isfinite(scores)&torch.isfinite(boxes).all(-1)&(quality>=.6)&(boxes[:,2:]>0).all(-1)
    ids=valid.nonzero().flatten();rank=torch.logit(scores.float().clamp(1e-5,1-1e-5))
    if guided and len(ids):rank=rank+.5*(geometry.float()-geometry[valid].amax()).clamp(-8,0)
    corners=torch.cat((boxes[:,:2]-boxes[:,2:]/2,boxes[:,:2]+boxes[:,2:]/2),-1)
    kept=ids[nms(corners[ids].float(),rank[ids].float(),.5)] if len(ids) else ids
    ordered=kept
    if spatial and len(kept):
        remaining=rank[kept].clone();chosen=[]
        for _ in range(min(60,len(kept))):
            j=remaining.argmax();chosen.append(kept[j]);distance=(boxes[kept,:2]-boxes[kept[j],:2]).norm(dim=-1)
            remaining-=1.5*torch.exp(-distance.square()/(2*.12**2));remaining[j]=-float('inf')
        ordered=torch.stack(chosen)
    return dict(valid=valid,ranking=rank,corners=corners,valid900=ids,nms=kept,**{f'top{k}':ordered[:k] for k in (15,30,60)})


def oracle_flags(distance,ids,a0_correct):
    best=float(distance[ids].min()) if len(ids) else None
    hit=best is not None and best<=20
    return dict(count=len(ids),oracle20=hit,oracle_NE=best,correctable=(not a0_correct and hit),keep_a0_oracle=(a0_correct or hit))


def fusion_input(raw,row,ids,device,sigma):
    f=raw['features'][ids].half().float()[None];g=raw['geometry'][ids][None];world=raw['centers'][ids][None];valid=torch.ones(1,len(ids),device=device,dtype=torch.bool)
    grid=row['grid_world'].to(device).float()[None];observed=row['observed'].to(device)[None]
    kernel=torch.exp(-torch.cdist(grid,world).square()/(2*sigma**2))*observed[...,None]
    return f,raw['text'].half().float()[None],raw['mask'][None],g,world,valid,kernel,observed,grid


def target_pixel_visible(d,target):
    depth=d['depth_valid'].numpy();height,width=depth.shape
    h=cv2.getPerspectiveTransform(d['footprint_world'].numpy().astype(np.float32),np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],np.float32))
    pixel=cv2.perspectiveTransform(np.asarray(target,np.float32)[None,None,:2],h)[0,0]
    if not np.isfinite(pixel).all() or not ((pixel>=0)&(pixel<=[width-1,height-1])).all():return False
    x,y=np.rint(pixel).astype(int);return bool(depth[y,x])


def summarize(records):
    output={}
    for cohort,predicate in [('all',lambda r:True),('a0_correct',lambda r:r['a0_correct']),('a0_wrong',lambda r:not r['a0_correct']),('target_pixel_valid',lambda r:r['visible']),('visible_a0_wrong',lambda r:r['visible'] and not r['a0_correct'])]:
        subset=[r for r in records if predicate(r)];n=len(subset);stages={};fusion={}
        for name in records[0]['stages']:
            values=[r['stages'][name] for r in subset];nonempty=[v['oracle_NE'] for v in values if v['oracle_NE'] is not None]
            stages[name]=dict(states=n,covered=sum(v['oracle20'] for v in values),coverage=sum(v['oracle20'] for v in values)/n if n else None,mean_queries=sum(v['count'] for v in values)/n if n else None,oracle_NE_nonempty=float(np.mean(nonempty)) if nonempty else None,empty_states=n-len(nonempty),correctable=sum(v['correctable'] for v in values),keep_a0_oracle_R1=sum(v['keep_a0_oracle'] for v in values)/n if n else None)
        for name in records[0]['fusion']:
            values=[r['fusion'][name] for r in subset];legal=[v for v in values if v['available']];counts={key:sum(bool(v.get(key,False)) for v in values) for key in ['covered','rank_correct','covered_rank_wrong','null_wins_covered','rank_correct_belief_wrong','a0_wrong_rank_correct_belief_wrong','updated_correct','corrected','damaged','bounded_impossible','rank_correct_fail_no_legal_support','rank_correct_fail_bound','rank_correct_fail_other']}
            fusion[name]=dict(states=n,**counts,ranking_error_given_coverage=counts['covered_rank_wrong']/counts['covered'] if counts['covered'] else None,fusion_failure_given_correct_rank=counts['rank_correct_belief_wrong']/counts['rank_correct'] if counts['rank_correct'] else None,mean_null_probability=float(np.mean([v['null_probability'] for v in legal])) if legal else None,mean_base_gap_failed_correct_rank=float(np.mean([v['base_gap'] for v in values if v.get('rank_correct_belief_wrong',False)])) if counts['rank_correct_belief_wrong'] else None)
        losses={name:dict(collections.Counter(r['first_loss'][name] for r in subset)) for name in records[0]['first_loss']}
        output[cohort]=dict(states=n,a0_correct=sum(r['a0_correct'] for r in subset),stages=stages,fusion=fusion,first_loss=losses)
    return output


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-root',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda');a=p.parse_args();root=Path(a.experiment_root);out=Path(a.output)
    if out.exists():raise ValueError('Refuse diagnostic output overwrite')
    checkpoint=root/'aerialvg_native_plain_sigma8_s0_20261101/best.pt';before=file_sha256(checkpoint);state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state['config']['relation']:raise ValueError('Use unchanged plain native checkpoint; no relation budget extrapolation')
    model=ObjectQueryRanker().to(a.device);model.load_state_dict(state['model']);model.eval().requires_grad_(False);config=state['config'];splits={};inputs=[];groups={};records_by_split={};examples=[];matches=collections.Counter();filter_queries=collections.Counter()
    out.mkdir(parents=True)
    for partition,directory in [('train','aerialvg_native_training_train256_20261101'),('dev','aerialvg_native_training_dev128_20261101')]:
        meta,rows=load(root/directory)
        if meta['partition']!=partition or meta['base_sha']!=config['base_sha']:raise ValueError('Invalid Train/internal-Dev source provenance')
        inputs.append(dict(path=str(root/directory/'manifest.json'),sha256=file_sha256(root/directory/'manifest.json')));groups[partition]={r['group'] for r in rows}
        lookup={}
        for source in meta['sources']:
            source_root=Path(source['path']);manifest_path=source_root/'manifest.json';m=json.loads(manifest_path.read_text())
            if m['partition']!=partition or m['status']!='complete' or file_sha256(manifest_path)!=source['manifest_sha']:raise ValueError('Raw source split/hash mismatch')
            inputs.append(dict(path=str(manifest_path),sha256=file_sha256(manifest_path)))
            for entry in m['frames']:lookup[(entry['episode'],entry['step'])]=source_root/(entry['key']+'.pt')
        records=[]
        with torch.inference_mode():
            for number,row in enumerate(rows):
                rawpath=lookup[(row['episode'],row['step'])];d=torch.load(rawpath,map_location='cpu',weights_only=True)
                if 'all_query_features' not in d or len(d['all_query_features'])!=900:raise ValueError('Actual900-query cache required')
                device=a.device;features=d['all_query_features'].to(device).float();features[d['query_ids'].to(device)]=d['query_features'].to(device).float();text=d['encoded_text'][0].to(device).float();mask=d['text_token_mask'][0].to(device).bool()
                scores=(features@text.T)[:,mask].sigmoid().amax(-1);boxes=d['all_boxes_cxcywh'].to(device).float();centers=d['all_world_centers'].to(device).float();quality=d['all_valid_pixel_fraction'].to(device).float();grid=row['grid_world'].to(device).float();base=row['base_logits'].to(device).float()
                nearest=torch.cat([torch.cdist(grid,block).argmin(0) for block in centers.split(64)]);prior=base[nearest]
                geometry=torch.stack(((prior-base.amax()).clamp(-20,0)/5,torch.logit(scores.clamp(1e-5,1-1e-5))/5,quality,boxes[:,2],boxes[:,3],boxes[:,0],boxes[:,1],boxes[:,2]*boxes[:,3]),-1)
                selected={name:selection_stages(boxes,scores,quality,prior,guided,spatial) for name,guided,spatial in [('native',False,False),('a0_guided',True,False),('native_spatial',False,True),('a0_guided_spatial',True,True)]}
                deployed=selected['a0_guided_spatial']['top15'];count=int(row['valid'].sum());matches['states']+=1
                aligned=len(deployed)==count and torch.allclose(centers[deployed].cpu(),row['query_world'][:count],atol=1e-4,rtol=0)
                matches['deployed_selection_matches_prepared']+=int(aligned)
                if not aligned and len(examples)<25:examples.append(dict(kind='cached_selector_mismatch',partition=partition,episode=row['episode'],step=row['step']))
                # Ground truth first enters here, only after every candidate selection above.
                target=row['labels']['target_world'][:2].to(device);distance=(centers-target).norm(dim=-1);grid_distance=(grid-target).norm(dim=-1);a0correct=bool(grid_distance[base.argmax()]<=20);good=distance<=20;visible=target_pixel_visible(d,target.cpu().numpy())
                r=dict(partition=partition,episode=row['episode'],step=row['step'],map=row['map_name'],a0_correct=a0correct,a0_NE=float(grid_distance[base.argmax()]),visible=visible,stages={},fusion={},first_loss={})
                finite_ids=torch.isfinite(distance).nonzero().flatten();r['stages']['raw900']=oracle_flags(distance,finite_ids,a0correct)
                corners=selected['native']['corners'];retention=((corners[:,2:].clamp(0,1)-corners[:,:2].clamp(0,1)).clamp_min(0).prod(-1)/(boxes[:,2:].prod(-1).clamp_min(1e-12)))
                for j in good.nonzero().flatten().tolist():
                    if not bool(torch.isfinite(scores[j])&torch.isfinite(boxes[j]).all()):reason='nonfinite_score_or_box'
                    elif not bool((boxes[j,2:]>0).all()):reason='nonpositive_box'
                    elif float(retention[j])<.6:reason='boundary_retained_area_lt_0.6'
                    elif float(quality[j])<.6:reason='depth_validity_times_retained_area_lt_0.6'
                    else:reason='passes_validity'
                    filter_queries[partition+'/'+reason]+=1
                raw=dict(features=features,text=text,mask=mask,centers=centers,geometry=geometry)
                for strategy,stages in selected.items():
                    for stage in ['valid900','nms','top15','top30','top60']:
                        r['stages'][strategy+'/'+stage]=oracle_flags(distance,stages[stage],a0correct)
                    for k in (15,30,60):
                        name=strategy+f'/top{k}';ids=stages[f'top{k}']
                        loss='retained' if bool(good[ids].any()) else ('absent_raw900' if not bool(good.any()) else 'validity_filter' if not bool(good[stages['valid900']].any()) else 'nms_suppression' if not bool(good[stages['nms']].any()) else 'budget_truncation')
                        r['first_loss'][name]=loss
                        if loss=='nms_suppression' and len(examples)<25:
                            positive=stages['valid900'][good[stages['valid900']]];overlap=box_iou(corners[positive],corners[stages['nms']]);pair=(overlap>.5).nonzero()
                            if len(pair):
                                first=pair[0];suppressed=int(positive[first[0]]);suppressor=int(stages['nms'][first[1]])
                                examples.append(dict(kind=loss,partition=partition,episode=row['episode'],step=row['step'],strategy=strategy,query=suppressed,suppressor=suppressor,IoU=float(overlap[first[0],first[1]]),suppressed_NE=float(distance[suppressed]),suppressor_NE=float(distance[suppressor]),ranking_margin=float(stages['ranking'][suppressor]-stages['ranking'][suppressed])))
                        if 'spatial' in strategy and k!=15:continue
                        if not len(ids):
                            r['fusion'][name]=dict(available=False,covered=False,rank_correct=False,updated_correct=a0correct,corrected=False,damaged=False);continue
                        f,t,tm,g,qworld,valid,kernel,observed,gworld=fusion_input(raw,row,ids,device,config['kernel_sigma'])
                        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=device.startswith('cuda')):q,null=model(f,t,tm,g,valid)
                        updated,delta=model.update_belief(base[None],q,null,kernel,observed,valid,bound=config['bound'])
                        ranked=int(ids[q[0].argmax()]);rankcorrect=bool(distance[ranked]<=20);updatedcorrect=bool(grid_distance[updated[0].argmax()]<=20);covered=bool(good[ids].any());probability=torch.cat((q.float(),null.float()[:,None]),-1).softmax(-1)[0];legal=grid_distance<=20
                        mass=1-probability[-1];cap=float(model.fusion_gain.clamp_min(0).tanh()*config['bound']*mass);support=observed[0]
                        best_legal=(base[legal]+cap*support[legal]).max() if legal.any() else torch.tensor(-float('inf'),device=device);best_wrong=(base[~legal]-cap*support[~legal]).max() if (~legal).any() else torch.tensor(-float('inf'),device=device);impossible=bool(best_legal<best_wrong)
                        failed=rankcorrect and not updatedcorrect;no_support=not bool((legal&support).any());gap=float(base.max()-base[legal].max()) if legal.any() else None
                        r['fusion'][name]=dict(available=True,covered=covered,rank_correct=rankcorrect,covered_rank_wrong=covered and not rankcorrect,null_wins_covered=covered and bool(null[0]>=q[0].max()),rank_correct_belief_wrong=failed,a0_wrong_rank_correct_belief_wrong=(not a0correct and failed),updated_correct=updatedcorrect,corrected=not a0correct and updatedcorrect,damaged=a0correct and not updatedcorrect,bounded_impossible=(not a0correct and impossible),rank_correct_fail_no_legal_support=failed and no_support,rank_correct_fail_bound=failed and not no_support and impossible,rank_correct_fail_other=failed and not no_support and not impossible,null_probability=float(probability[-1]),base_gap=gap,delta_max=float(delta.abs().max()))
                records.append(r)
                if (number+1)%100==0:print(json.dumps(dict(partition=partition,completed=number+1,total=len(rows))),flush=True)
        records_by_split[partition]=records;splits[partition]=summarize(records)
        atomic_json(records,out/(partition+'_states.json'))
    if groups['train']&groups['dev']:raise ValueError('Train/internalDev group overlap')
    if file_sha256(checkpoint)!=before:raise ValueError('Checkpoint changed')
    result=dict(base_commit='fcfde0bcd9e7f6cc66d0a1f23302ecd4ec9289f6',checkpoint=str(checkpoint),checkpoint_sha256=before,checkpoint_unchanged=True,base_sha=config['base_sha'],inputs=inputs,selection_reproduction=dict(matches),filtered_positive_queries=dict(filter_queries),examples=examples,splits=splits,protocol=dict(radius_m=20,quality_min=.6,nms_iou=.5,guidance_weight=.5,guidance_clip=8,spatial_suppression=1.5,spatial_sigma=.12,kernel_sigma=config['kernel_sigma'],fusion_bound=config['bound'],fusion_gain_tanh=float(model.fusion_gain.tanh()),A0='cached base_logits argmax on frozen A3 geometry grid; not an extra model',oracle='20m position-only; no target-instance identity available',NE='minimum candidate-center distance, conditional on nonempty sets; empty counts reported',budgets='15/30/60 reuse same frozen plain head;30/60 are offline out-of-training-budget counterfactuals',no_training=True,no_formal_rollout=True,no_test_or_new_val_annotations=True))
    atomic_json(result,out/'candidate_summary.json');print(json.dumps(dict(done=True,selection_reproduction=dict(matches),states={k:v['all']['states'] for k,v in splits.items()})),flush=True)


if __name__=='__main__':main()
