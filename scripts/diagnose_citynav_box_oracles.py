"""TRAIN-only frozen candidate waterfall + strictly labeled box Oracle sensitivity."""
import argparse,json,time
from collections import Counter
from pathlib import Path
import cv2,numpy as np,torch

from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.citynav_box_diagnostics import (OracleMode,authorize_box,public_selection_stages,registered_iou,bounded_fusion_limit,box_world_centers,bbox_cxcywh)
from graphnav.citynav_oracle_runtime import (prepare_public_evidence,pack_public,predict_packed,OracleRegionEncoder,append_oracle_region,force_selected,ideal_position)
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.workspace_paths import WORKSPACE
from scripts.train_object_query_belief import load
from scripts.prepare_citynav_box_annotations import train_sources


NORMAL_NAMES=['A0','Native15','Native30','Native60']
ORACLE_NAMES=['A_box_exists','B_existing15_selected','B_existing900_selected','B_added_box_selected',
              'C_box_bounded','C_target_bounded','C_box_unbounded','C_target_unbounded']


def outcome(error,old,native):
    return dict(error_m=float(error),correct=bool(error<=20),corrected_A0=bool(old>20 and error<=20),
                damaged_A0=bool(old<=20 and error>20),corrected_Native=bool(native>20 and error<=20),damaged_Native=bool(native<=20 and error>20))


def summarize(records,mode):
    cohorts={'all':records,'auto_high_support':[r for r in records if r['auto_high_support']],
             'Oracle_hypothesis_eligible':[r for r in records if r['oracle_authorized']],
             'A0_wrong':[r for r in records if not r['methods']['A0']['correct']],
             'A0_correct':[r for r in records if r['methods']['A0']['correct']]}
    for c in sorted({r['category'] for r in records}):cohorts['category_'+c]=[r for r in records if r['category']==c]
    for s in ['full','partial','outside','unknown']:cohorts['geometry_'+s]=[r for r in records if r['geometry_hint']==s]
    for c in sorted({r['category'] for r in records}):cohorts['eligible_'+c]=[r for r in records if r['category']==c and r['oracle_authorized']]
    result={}
    for name,rows in cohorts.items():
        methods={};n=len(rows)
        for method in NORMAL_NAMES+ORACLE_NAMES:
            values=[r['methods'][method] for r in rows if method in r['methods']
                    and not (mode=='verified' and method in ORACLE_NAMES and not r['actual_visibility_verified'])]
            if not values:methods[method]=dict(states=0,R1=None,NE=None);continue
            methods[method]=dict(states=len(values),R1=sum(v['correct'] for v in values)/len(values),NE=sum(v['error_m'] for v in values)/len(values),
                corrected_A0=sum(v['corrected_A0'] for v in values),damaged_A0=sum(v['damaged_A0'] for v in values),
                corrected_Native=sum(v['corrected_Native'] for v in values),damaged_Native=sum(v['damaged_Native'] for v in values))
        labeled=[r for r in rows if r['stages'] is not None]
        stages={}
        for stage in ['raw900','valid900','nms','top15','top30','top60']:
            values=[r['stages'][stage] for r in labeled]
            stages[stage]=dict(states=len(values),covered=sum(v['covered'] for v in values),
                              recall=sum(v['covered'] for v in values)/len(values) if values else None,
                              vl_top1=sum(v['vl_top1'] for v in values)/len(values) if values else None,
                              vl_top5=sum(v['vl_top5'] for v in values)/len(values) if values else None,
                              Native_top1=sum(v.get('Native_top1',False) for v in values)/len(values) if stage.startswith('top') and values else None,
                              Native_top5=sum(v.get('Native_top5',False) for v in values)/len(values) if stage.startswith('top') and values else None)
        wrong_labeled=[r for r in labeled if not r['methods']['A0']['correct']]
        waterfall=dict(Counter(r['first_failure'] for r in wrong_labeled))
        passed={k:sum(r['stages'][k]['covered'] for r in wrong_labeled) for k in ['raw900','valid900','nms','top15','top30','top60']}
        matched=sum(r['stages']['top15']['Native_top1'] for r in wrong_labeled)
        baseline_corrected=sum(r['methods']['Native15']['correct'] for r in wrong_labeled)
        matched_corrected=sum(r['stages']['top15']['Native_top1'] and r['methods']['Native15']['correct'] for r in wrong_labeled)
        result[name]=dict(states=n,episodes=len({r['episode'] for r in rows}),instructions=len({r['instruction_group'] for r in rows}),instances=len({r['instance_group'] for r in rows}),
            methods=methods,stages=stages,labeled_A0_wrong=len(wrong_labeled),first_failure_A0_wrong=waterfall,
            A0_wrong_pass_counts=dict(**passed,Native_top1_match=matched,Native_corrected=baseline_corrected,match_and_corrected=matched_corrected),
            bounded_impossible_A0_wrong=sum(r['fusion_limit']['bounded_impossible'] for r in rows if not r['methods']['A0']['correct']),
            annotation_or_actual_visibility_unknown=sum(not r['actual_visibility_verified'] for r in rows))
    primary=[r for r in records if r['actual_visibility_verified']]
    result['provenance']=dict(mode=mode,partition='train',no_training=True,human_verified_states=len(primary),
        human_verified_Oracle_results='NA until human reviews' if not primary else 'see verified run',
        projection_hypothesis_states=sum(r['oracle_authorized'] and not r['actual_visibility_verified'] for r in records),
        semantic_visibility_known=False,warning='Projection-sensitivity Oracles are NOT human pixel GT and do NOT prove actual visibility or detector correctness')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['cache','annotations','native-checkpoint','output']:p.add_argument('--'+n,required=True)
    p.add_argument('--reviews');p.add_argument('--oracle-mode',choices=[m.value for m in OracleMode],default='normal')
    p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int,default=0)
    a=p.parse_args();out=Path(a.output)
    if out.exists() and not a.resume:raise ValueError('No overwrite; explicit resume only')
    meta,lookup=train_sources(a.cache);_,rows=load(a.cache)
    annotations=[json.loads(line) for line in Path(a.annotations).read_text().splitlines()];annotation_by_id={r['sample_id']:r for r in annotations}
    if any(r['partition']!='train' for r in annotations):raise ValueError('Oracles prohibited outside TRAIN')
    reviews={}
    if a.reviews:
        reviews={r['sample_id']:r for r in json.loads(Path(a.reviews).read_text())['reviews']}
    native_sha=file_sha256(a.native_checkpoint);aerialvg_path=WORKSPACE/'third_party/aerialvg_assets/aerialvg.pth';aerialvg_sha=file_sha256(aerialvg_path)
    for source in meta['sources']:
        sm=json.loads((Path(source['path'])/'manifest.json').read_text())
        if sm['official_checkpoint_sha']!=aerialvg_sha:raise ValueError('Frozen AerialVG identity mismatch')
    protocol=dict(base_commit='9d8a615cc26a7feadaaad38271e7a3201fb7a046',partition='train',mode=a.oracle_mode,
        native_sha=native_sha,aerialvg_sha=aerialvg_sha,cache_manifest_sha=file_sha256(Path(a.cache)/'manifest.json'),
        annotation_sha=file_sha256(a.annotations),review_sha=file_sha256(a.reviews) if a.reviews else None,
        source_sha={n:file_sha256(n) for n in [__file__,'graphnav/citynav_box_diagnostics.py','graphnav/citynav_oracle_runtime.py']},
        constants=dict(IoU=.5,sigma=8.,bound=2.,budget=[15,30,60]),limit=a.limit,no_training=True,
        mode_warning='UNVERIFIED projection hypothesis, NOT pixel GT' if a.oracle_mode=='projection-hypothesis' else 'strict verified-only' if a.oracle_mode=='verified' else 'no Oracle/private box injection')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Resume source/protocol changed')
    atomic_json(protocol,out/'protocol.json')
    torch.set_num_threads(4);device=torch.device('cuda');checkpoint=torch.load(a.native_checkpoint,map_location='cpu',weights_only=True)
    native=ObjectQueryRanker().to(device);native.load_state_dict(checkpoint['model']);native.eval().requires_grad_(False)
    crop_encoder=OracleRegionEncoder(out/'oracle_only_crop_features',aerialvg_sha)
    records=[]
    if (out/'records.jsonl').exists():records=[json.loads(line) for line in (out/'records.jsonl').read_text().splitlines()]
    done={(r['episode'],r['step']) for r in records}
    if a.limit:rows=rows[:a.limit]
    with torch.inference_mode():
        for number,row in enumerate(rows):
            if (row['episode'],row['step']) in done:continue
            identity=f"train:{row['episode']}:{row['step']}";ann=annotation_by_id[identity]
            rawpath=lookup[(row['episode'],row['step'])];raw=torch.load(rawpath,weights_only=True)
            f=raw['all_query_features'].to(device).float();f[raw['query_ids'].to(device)]=raw['query_features'].to(device).float()
            e=prepare_public_evidence(f,raw['encoded_text'][0].to(device).float(),raw['text_token_mask'][0].to(device).bool(),
                raw['all_boxes_cxcywh'].to(device).float(),raw['all_world_centers'].to(device).float(),raw['all_valid_pixel_fraction'].to(device).float(),
                row['base_logits'].to(device).float(),row['grid_world'].to(device).float(),row['observed'].to(device))
            stages=public_selection_stages(e.boxes,e.native_scores,e.quality,e.prior)
            base_pack=pack_public(e,stages['top15'],15)
            n=int(row['valid'].sum())
            if len(stages['top15'])!=n or not torch.allclose(base_pack['world'][0,:n].cpu(),row['query_world'][:n],atol=1e-4,rtol=0):raise ValueError('Original candidate alignment failed')
            if not torch.equal(base_pack['features'][0].cpu(),row['features'].float()):raise ValueError('Original feature alignment failed')
            predictions={15:predict_packed(native,e,base_pack)}
            for budget in [30,60]:predictions[budget]=predict_packed(native,e,pack_public(e,stages['top'+str(budget)],budget))
            # Normal outputs all computed before private box/target evaluation.
            output={'A0':e.base,'Native15':predictions[15]['belief'],'Native30':predictions[30]['belief'],'Native60':predictions[60]['belief']}
            target=row['labels']['target_world'].to(device).float();distance=(e.grid-target[:2]).norm(dim=-1)
            errors={name:float(distance[value.argmax()]) for name,value in output.items()}
            diagnostic_stages=None;first_failure='annotation_or_visibility_unverified';iou=None;oracle_authorized=False;actual_verified=False;details={}
            review=reviews.get(identity)
            authorization=None
            if a.oracle_mode!='normal':authorization=authorize_box(a.oracle_mode,ann,review,partition='train')
            eval_box=authorization.bbox if authorization and authorization.actual_visibility_verified else ann['auto_bbox_xyxy'] if ann['quality']['high_geometric_support'] else None
            if eval_box is not None:
                iou=registered_iou(e.boxes,eval_box,ann['image_shape']);diagnostic_stages={}
                for name,ids in stages.items():
                    vl=ids[e.native_scores[ids].argsort(descending=True)]
                    diagnostic_stages[name]=dict(count=len(ids),covered=bool((iou[ids]>=.5).any()),vl_top1=bool((iou[vl[:1]]>=.5).any()),vl_top5=bool((iou[vl[:5]]>=.5).any()))
                    if name in ['top15','top30','top60']:
                        budget=int(name[3:]);score=predictions[budget]['q'][0,:len(ids)];ordered=ids[score.argsort(descending=True)]
                        diagnostic_stages[name].update(Native_top1=bool((iou[ordered[:1]]>=.5).any()),Native_top5=bool((iou[ordered[:5]]>=.5).any()))
                first_failure=next((name+'_no_match' for name in ['raw900','valid900','nms','top15'] if not diagnostic_stages[name]['covered']),
                    'Native_ranking_error' if not diagnostic_stages['top15']['Native_top1'] else 'Belief_or_mapping_failure' if errors['Native15']>20 else 'success')
            limit=bounded_fusion_limit(e.base,e.grid,e.observed,target,native.fusion_gain)
            if authorization is not None:
                oracle_authorized=True;actual_verified=authorization.actual_visibility_verified
                bgr=cv2.imread(ann['image_path'])
                if bgr is None or file_sha256(ann['image_path'])!=ann['image_sha']:raise ValueError('Current Oracle RGB changed')
                details['box_provenance']=authorization.provenance
                # Existing-candidate B: force correct label, retain actual score/null/kernel.
                for budget,name in [(15,'B_existing15_selected')]:
                    ids=stages['top'+str(budget)];correct=(iou[ids]>=.5).nonzero().flatten()
                    if len(correct):
                        chosen=int(correct[predictions[budget]['q'][0,correct].argmax()]);belief,mass=force_selected(native,e,predictions[budget],chosen,authorization)
                        output[name]=belief;details[name]=dict(attempted=True,evidence_mass=mass,query_world_error_m=float((predictions[budget]['pack']['world'][0,chosen]-target[:2]).norm()))
                    else:output[name]=output['Native15'];details[name]=dict(attempted=False,reason='no_correct_existing15_candidate')
                correct900=stages['valid900'][iou[stages['valid900']]>=.5]
                if len(correct900):
                    p900=predict_packed(native,e,pack_public(e,correct900,max(1,len(correct900))))
                    # Null from original15, not a GT-dependent re-estimated null.
                    p900['null']=predictions[15]['null'];chosen=int(p900['q'][0,:len(correct900)].argmax())
                    belief,mass=force_selected(native,e,p900,chosen,authorization);output['B_existing900_selected']=belief
                    details['B_existing900_selected']=dict(attempted=True,evidence_mass=mass,query_world_error_m=float((p900['pack']['world'][0,chosen]-target[:2]).norm()))
                else:output['B_existing900_selected']=output['Native15'];details['B_existing900_selected']=dict(attempted=False,reason='no_correct_valid900_candidate')
                encoded=crop_encoder.encode(authorization,bgr,raw['depth_valid'].numpy(),raw['footprint_world'].numpy(),row['instruction'],ann['image_sha'])
                if encoded['available']:
                    aug=append_oracle_region(e,base_pack,authorization,encoded['feature'],raw['footprint_world'].numpy(),ann['image_shape'],raw['depth_valid'].numpy())
                    pred=predict_packed(native,e,aug);output['A_box_exists']=pred['belief'];selected=int(pred['q'][0].argmax())
                    iou_aug=torch.cat([iou[stages['top15']],torch.zeros(15-n,device=device),torch.ones(1,device=device)])
                    order=pred['q'][0].argsort(descending=True)
                    details['A_box_exists']=dict(attempted=True,selected_injected_box=selected==15,top1_match=bool(iou_aug[order[0]]>=.5),
                        top5_match=bool((iou_aug[order[:5]]>=.5).any()),crop_provenance=encoded['provenance'],crop_query_box=encoded['selected_crop_query_box'],
                        crop_pixel_sha=encoded['crop_pixel_sha'],source_image_sha=encoded['source_image_sha'],frozen_model_sha=encoded['model_sha'])
                    belief,mass=force_selected(native,e,pred,15,authorization);output['B_added_box_selected']=belief
                    details['B_added_box_selected']=dict(attempted=True,evidence_mass=mass,query_world_error_m=float((aug['world'][0,15]-target[:2]).norm()))
                else:
                    output['A_box_exists']=output['Native15'];output['B_added_box_selected']=output['Native15'];details['A_box_exists']=dict(attempted=False,reason=encoded['reason'])
                # C separates bbox-centre mapping, evidence mass and bounded fusion.
                box_position=box_world_centers(bbox_cxcywh(authorization.bbox,ann['image_shape'])[None],raw['footprint_world'].numpy(),ann['image_shape']).to(device)[0]
                output['C_box_bounded'],output['C_box_unbounded']=ideal_position(native,e,box_position,authorization)
                output['C_target_bounded'],output['C_target_unbounded']=ideal_position(native,e,target[:2],authorization)
                for name in ORACLE_NAMES:output.setdefault(name,output['Native15'])
            elif a.oracle_mode!='normal':
                for name in ORACLE_NAMES:output[name]=output['Native15']
            final_errors={name:float(distance[value.argmax()]) for name,value in output.items()}
            record=dict(sample_id=identity,episode=row['episode'],step=row['step'],instruction_group=row['group'],instance_group=ann['instance_group'],category=ann['category'],geometry_hint=ann['geometric_visibility_hint'],
                actual_visibility_verified=actual_verified,auto_high_support=ann['quality']['high_geometric_support'],oracle_authorized=oracle_authorized,
                annotation_status=ann['annotation_status'],mode=a.oracle_mode,semantic_visibility='unverified' if not actual_verified else review['visibility'],
                methods={name:outcome(error,errors['A0'],errors['Native15']) for name,error in final_errors.items()},
                stages=diagnostic_stages,first_failure=first_failure,fusion_limit=limit,oracle_details=details)
            records.append(record)
            with (out/'records.jsonl').open('a') as handle:handle.write(json.dumps(record)+'\n')
            atomic_json(dict(processed=len(records),expected=len(rows),oracle_attempted=sum(r['oracle_authorized'] for r in records)),out/'progress.json')
            if len(records)%25==0:print(json.dumps(dict(processed=len(records),mode=a.oracle_mode)),flush=True)
    report=summarize(records,a.oracle_mode);atomic_json(report,out/'summary.json')
    if file_sha256(a.native_checkpoint)!=native_sha or file_sha256(aerialvg_path)!=aerialvg_sha:raise ValueError('Frozen weights changed')
    atomic_json(dict(complete=True,records=len(records),native_unchanged=True,aerialvg_unchanged=True,no_training=True,only_TRAIN=True,
                     human_verified_states=sum(r['actual_visibility_verified'] for r in records),hypothesis_only=a.oracle_mode=='projection-hypothesis'),out/'complete.json')
    print(json.dumps(report['all']),flush=True)


if __name__=='__main__':main()
