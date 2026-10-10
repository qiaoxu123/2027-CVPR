"""Dev-only calibration and frozen Seen evaluation of causal visual residuals."""
import argparse
import json
from pathlib import Path
import time
import torch
from .evaluate_features import load,metrics
from .train_head import collate
from .head import VisualEvidenceNetwork,Evidence,update_belief
from experiments.bidirectional_core.cache import CachedCoreDataset
from experiments.bidirectional_core.stats import summarize_rows,paired_difference
from graphnav.checkpointing import file_sha256


def load_head(checkpoint,device):
    if checkpoint=='native':return None
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=VisualEvidenceNetwork(dimension=state['dimension'],token_attention=state['token_attention']).to(device).eval()
    model.load_state_dict(state['model'],strict=True)
    return model


@torch.inference_mode()
def encode_scores(rows,model,controls,kind,device):
    records={}
    for original in rows:
        row=dict(original);row['features']=dict(row['features'])
        if kind!='real':
            row['features']['polygon_masked']=controls[row['metadata']['sample_id']][kind+'_rgb']
            row['patch_tokens']=controls[row['metadata']['sample_id']][kind+'_tokens']
        public,_,_=collate([row],'polygon_masked','phrase');public={k:v.to(device) for k,v in public.items()}
        if model is None:
            score=(public['images'].float()*public['text'][:,None].float()).sum(-1)*20
            evidence=Evidence(score,public['valid'].float(),public['valid'])
        else:evidence=model(public)
        records[row['metadata']['sample_id']]=dict(score=evidence.score.cpu(),confidence=evidence.confidence.cpu(),observed=evidence.observed.cpu(),row=original)
    return records


def predict(dataset,records,gain,threshold,history):
    outputs=[];memories={}
    order=sorted(range(len(dataset)),key=lambda i:(dataset[i]['metadata']['episode_id'],dataset[i]['metadata']['cutoff_step']))
    for index in order:
        row=dataset[index];p=row['public'];m=row['metadata'];memory=memories.setdefault(m['episode_id'],{})
        current=records.get(m['sample_id']);candidates=[]
        if current:
            source=current['row'];quality=source['public']['roi_quality'];regions=source['metadata']['roi_objects']
            for key in ['sample_id','episode_id','map_name','cutoff_step','actual_camera_pose']:
                if source['metadata'][key]!=m[key]:raise ValueError('feature/source current observation differs: '+key)
            if [r['object_id'] for r in regions]!=[r['object_id'] for r in m['roi_objects']]:raise ValueError('candidate identity/order changed')
            if not torch.equal(source['public']['roi_grid_ids'],p['roi_grid_ids']) or not torch.equal(source['public']['roi_valid'],p['roi_valid']):raise ValueError('public ROI geometry/availability changed')
            for j,region in enumerate(regions):
                if not current['observed'][0,j]:continue
                item=dict(score=float(current['score'][0,j]),confidence=float(current['confidence'][0,j]),grid=int(source['public']['roi_grid_ids'][j]),step=source['metadata']['cutoff_step'],quality=float(quality[j].prod()),object_id=region['object_id'])
                candidates.append(item)
                previous=memory.get(region['object_id'])
                if previous is None or item['quality']>=previous['quality']:memory[region['object_id']]=item
        if history:candidates=list(memory.values())
        if any(r['step']>m['cutoff_step'] for r in candidates):raise ValueError('future RGB history')
        logits=p['static_logits'][None]
        if candidates:
            evidence=Evidence(torch.tensor([[r['score'] for r in candidates]]),torch.tensor([[r['confidence'] for r in candidates]]),torch.ones(1,len(candidates),dtype=torch.bool))
            association=torch.zeros(1,len(candidates),logits.shape[-1],dtype=torch.bool)
            for j,item in enumerate(candidates):association[0,j,item['grid']]=True
            updated,known=update_belief(logits,evidence,association,gain=gain,threshold=threshold)
            if not torch.equal(updated[~known],logits[~known]):raise ValueError('unknown region penalized')
        else:updated=logits.clone();known=torch.zeros_like(logits,dtype=torch.bool)
        grid=p['bounds'][:2]+p['grid_xy']*(p['bounds'][2:]-p['bounds'][:2]);distance=torch.linalg.vector_norm(grid-row['labels']['target_world'][:2],dim=-1)
        rank=updated[0].argsort(descending=True);static=logits[0].argmax();correct=bool(distance[rank[0]]<=20);before=bool(distance[static]<=20)
        value=dict(r1=float(correct),r4=float((distance[rank[:4]]<=20).any()),r8=float((distance[rank[:8]]<=20).any()),ne=float(distance[rank[0]]),recovery=float(correct) if not before else None,damage=float(not correct) if before else None,net_recovery=float(correct)-float(before),evidence_coverage=float(bool(known.any())),forward_r1=float(correct) if m['direction']=='forward' else None,reverse_r1=float(correct) if m['direction']=='reverse' else None)
        # Only evaluation code accesses weak labels, after public predictions.
        weak=metrics(current['score'][0],current['row']) if current and bool((current['row']['labels']['positive_roi_mask']).any() and (current['row']['labels']['semantic_valid_mask'] & ~current['row']['labels']['positive_roi_mask']).any()) else None
        value.update(weak_r1=weak['r1'] if weak else None,weak_forward_r1=weak['forward_r1'] if weak else None,building_hard_r1=weak['building_hard_r1'] if weak else None)
        outputs.append(dict(sample_id=m['sample_id'],family_id=m['family_id'],map_name=m['map_name'],direction=m['direction'],metrics=value,static_top1=int(static),visual_top1=int(rank[0]),observed_cells=int(known.sum()),evidence=[{k:v for k,v in r.items() if k!='quality'} for r in candidates]))
    return outputs


def summary(rows):return {key:summarize_rows(rows,key,repetitions=2000) for key in rows[0]['metrics']}


def run(args):
    output=Path(args.output)
    if output.exists():raise ValueError('refusing experiment overwrite')
    output.mkdir(parents=True);started=time.time();dataset=CachedCoreDataset(args.cache,'E1');features=load(args.features)
    control_rows=load(args.controls);controls={r['sample_id']:r for r in control_rows}
    if set(controls)!={r['metadata']['sample_id'] for r in features}:raise ValueError('control population differs')
    if args.mode=='calibrate':
        configs=[]
        for name,checkpoint in json.loads(Path(args.heads).read_text()).items():
            model=load_head(checkpoint,args.device);records=encode_scores(features,model,controls,'real',args.device)
            for history in [False,True]:
                for gain in [0,.025,.05,.1,.2]:
                    for threshold in ([0] if checkpoint=='native' else [0,.1,.25,.5,.7,.9]):
                        rows=predict(dataset,records,gain,threshold,history)
                        forward=[r['metrics']['r1'] for r in rows if r['direction']=='forward']
                        net=sum(r['metrics']['net_recovery'] for r in rows);damage=sum(r['metrics']['damage'] or 0 for r in rows)
                        configs.append(dict(name=name,checkpoint=checkpoint,checkpoint_sha256=file_sha256(checkpoint) if checkpoint!='native' else None,gain=gain,threshold=threshold,history=history,dev_forward_r1=sum(forward)/len(forward),dev_net_recovery=net,dev_damage=damage))
        # Outcome selection happens only on internal Dev, deterministic tie rules.
        best=dict(max(configs,key=lambda c:(c['dev_forward_r1'],c['dev_net_recovery'],-c['dev_damage'],-c['gain'],-int(c['history']))))
        best.update(frozen=True,feature_representation='polygon_masked',language='phrase',calibration_cache_sha256=file_sha256(Path(args.cache)/'manifest.json'),selection='Dev forward Goal R1, net recovery, least damage, smallest gain, current-only tie preference',grid_trials=len(configs),configuration_candidates=configs)
        (output/'frozen_config.json').write_text(json.dumps(best,indent=2));print(json.dumps({k:v for k,v in best.items() if k!='configuration_candidates'},indent=2));return
    config=json.loads(Path(args.config).read_text())
    if not config['frozen']:raise ValueError('unfrozen test config')
    if config['checkpoint']!='native' and file_sha256(config['checkpoint'])!=config['checkpoint_sha256']:raise ValueError('selected checkpoint changed')
    model=load_head(config['checkpoint'],args.device);predictions={}
    predictions['geometry_only']=predict(dataset,{},0,0,False)
    for kind in ['real','zero','shuffled']:
        records=encode_scores(features,model,controls,kind,args.device)
        predictions[kind+'_rgb']=predict(dataset,records,config['gain'],config['threshold'],config['history'])
    # Same selected model and calibration, only history availability changes.
    records=encode_scores(features,model,controls,'real',args.device)
    predictions['real_current_only']=predict(dataset,records,config['gain'],config['threshold'],False)
    predictions['real_with_history']=predict(dataset,records,config['gain'],config['threshold'],True)
    for name,rows in predictions.items():
        with (output/(name+'_predictions.jsonl')).open('w') as handle:
            for row in rows:handle.write(json.dumps(row)+'\n')
    paired={name:{key:paired_difference(predictions['geometry_only'],rows,key,repetitions=2000) for key in ['r1','r4','r8','ne','forward_r1','reverse_r1']} for name,rows in predictions.items()}
    real_vs_shuffled={key:paired_difference(predictions['shuffled_rgb'],predictions['real_rgb'],key,repetitions=2000) for key in ['r1','forward_r1','weak_r1','weak_forward_r1']}
    real_vs_zero={key:paired_difference(predictions['zero_rgb'],predictions['real_rgb'],key,repetitions=2000) for key in ['r1','forward_r1','weak_r1','weak_forward_r1']}
    delta=paired['real_rgb']['r1'];gap=real_vs_shuffled['r1'];net=sum(r['metrics']['net_recovery'] for r in predictions['real_rgb'])
    gate=bool(delta['map_cluster_95ci'][0]>0 and net>0 and gap['mean']>0)
    result=dict(models={n:summary(r) for n,r in predictions.items()},paired_vs_geometry=paired,real_minus_shuffled=real_vs_shuffled,real_minus_zero=real_vs_zero,closedloop_and_unseen_gate_passed=gate,gate_definition='paired map-cluster lower95CI(real-geometry Goal R1)>0, positive Net Recovery, real>shuffled Goal R1',actual_states=len(dataset),actual_feature_states=len(features),config=config,cache_manifest_sha256=file_sha256(Path(args.cache)/'manifest.json'),features_manifest_sha256=file_sha256(Path(args.features)/'manifest.json'),controls_manifest_sha256=file_sha256(Path(args.controls)/'manifest.json'),seconds=time.time()-started,peak_gpu_bytes=torch.cuda.max_memory_allocated(),labels='CityNav attributes weak; target-world only evaluation, not VEN/fusion input')
    (output/'metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(gate_passed=gate,real_minus_geometry=delta,real_minus_shuffled=gap),indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['calibrate','evaluate'],required=True);p.add_argument('--cache',required=True);p.add_argument('--features',required=True);p.add_argument('--controls',required=True);p.add_argument('--output',required=True);p.add_argument('--heads');p.add_argument('--config');p.add_argument('--device',default='cuda');run(p.parse_args())


if __name__=='__main__':main()
