"""CPU arrival probe: grouped Train-fit/calibration, one Dev verification; no deployment."""
import argparse,json,sys
from pathlib import Path
import torch
from graphnav.native_arrival import arrival_features,FEATURE_NAMES
from graphnav.object_query_belief import ObjectQueryRanker
from graphnav.checkpointing import atomic_save,atomic_json,file_sha256
from graphnav.workspace_paths import PROJECT,WORKSPACE
from scripts.train_object_query_belief import load


def public_pose(row,source):
    if 'pose' in row:return row['pose']
    return source[row['episode']].trajectory[row['step']]


def main():
    p=argparse.ArgumentParser();p.add_argument('--train',required=True);p.add_argument('--dev',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output)
    if out.exists():raise ValueError('Refusing overwrite')
    train_meta,train=load(a.train);dev_meta,dev=load(a.dev)
    if train_meta['partition']!='train' or dev_meta['partition']!='dev' or {r['group'] for r in train}&{r['group'] for r in dev}:raise ValueError('Invalid group partitions')
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if state['config']['relation'] or train_meta['base_sha']!=state['config']['base_sha'] or dev_meta['base_sha']!=train_meta['base_sha']:raise ValueError('Plain checkpoint/base required')
    ranker=ObjectQueryRanker();ranker.load_state_dict(state['model']);ranker.eval().requires_grad_(False)
    sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
    from sbfnav.dataset import CityNavDataset
    source=CityNavDataset(WORKSPACE/'DATA/sbfnav-data','train_seen',version='processed_citynav')
    def prepare(rows):
        features=[];labels=[];groups=[]
        with torch.inference_mode():
            for row in rows:
                x={k:v[None].float() if v.is_floating_point() else v[None] for k,v in row.items() if torch.is_tensor(v)}
                pose=torch.as_tensor(public_pose(row,source),dtype=torch.float32)[None]
                with torch.autocast('cpu',dtype=torch.bfloat16):q,null=ranker(x['features'],x['text'],x['text_mask'],x['geometry'],x['valid'])
                kernel=torch.exp(-torch.cdist(x['grid_world'],x['query_world']).square()/(2*state['config']['kernel_sigma']**2))*x['observed'][...,None]*x['valid'][:,None]
                belief,_=ranker.update_belief(x['base_logits'],q,null,kernel,x['observed'],x['valid'],bound=state['config']['bound'])
                features.append(arrival_features(q,null,x['query_world'],x['valid'],x['geometry'],x['base_logits'],belief,x['grid_world'],pose)[0])
                labels.append(float((pose[0,:2]-row['labels']['target_world'][:2]).norm()<=20));groups.append(row['group'])
        return torch.stack(features).clone(),torch.tensor(labels),groups
    x,y,groups=prepare(train)
    calibration=torch.tensor([int(g,16)%5==0 for g in groups]);fit=~calibration
    if not fit.any() or not calibration.any():raise ValueError('Empty grouped fit/calibration')
    mean=x[fit].mean(0);std=x[fit].std(0).clamp_min(.05);z=(x-mean)/std
    torch.manual_seed(0);model=torch.nn.Linear(len(FEATURE_NAMES),1);optimizer=torch.optim.AdamW(model.parameters(),lr=.02,weight_decay=.1)
    for _ in range(500):
        loss=torch.nn.functional.binary_cross_entropy_with_logits(model(z[fit])[:,0],y[fit]);optimizer.zero_grad();loss.backward();optimizer.step()
    def score(prob,target,group,threshold):
        selected=prob>=threshold;tp=int((selected&(target>0)).sum());fp=int((selected&(target==0)).sum());n=int(selected.sum())
        return dict(threshold=threshold,confirmed=n,tp=tp,fp=fp,precision=tp/n if n else None,recall=tp/int(target.sum()) if target.sum() else None,confirmation_groups=len({g for g,s in zip(group,selected) if s}))
    with torch.no_grad():prob=model(z)[:,0].sigmoid()
    calgroups=[g for g,flag in zip(groups,calibration) if flag]
    choices=[score(prob[calibration],y[calibration],calgroups,t/100) for t in range(50,100)]
    eligible=[v for v in choices if v['confirmed']>=20 and v['confirmation_groups']>=5 and v['precision']>=.95]
    chosen=max(eligible,key=lambda v:(v['recall'],v['precision'],v['threshold'])) if eligible else None
    result=dict(kind='native-arrival-linear-probe-v1',fit_states=int(fit.sum()),calibration_states=int(calibration.sum()),fit_groups=len({g for g,f in zip(groups,fit) if f}),calibration_groups=len(set(calgroups)),calibration_choices=choices,chosen=chosen,native_sha=file_sha256(a.checkpoint),train_manifest_sha=file_sha256(Path(a.train)/'manifest.json'),protocol='CPU BF16 frozen native ranker; group hash modulo5 calibration, remaining groups fit. Fixed500 optimizer steps; threshold selected on calibration precision>=95%,>=20 states and>=5 groups. No Stop deployment or navigation gain claim.')
    # Dev is evaluated only once after model and threshold are fixed.
    if chosen:
        xd,yd,gd=prepare(dev)
        with torch.no_grad():pd=model((xd-mean)/std)[:,0].sigmoid()
        result['dev']=score(pd,yd,gd,chosen['threshold'])
    out.mkdir(parents=True);atomic_save(dict(model=model.state_dict(),mean=mean,std=std,feature_names=FEATURE_NAMES,chosen=chosen,native_sha=result['native_sha']),out/'probe.pt');atomic_json(result,out/'report.json');print(json.dumps({k:v for k,v in result.items() if k!='calibration_choices'},indent=2))


if __name__=='__main__':main()
