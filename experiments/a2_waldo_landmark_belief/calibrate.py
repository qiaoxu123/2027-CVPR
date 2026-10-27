"""Fit a provisional gate on independent TRAIN calibration target groups only.

No benchmark split is opened. Fit/verify groups are disjoint. After the one
chosen configuration is verified, the only permitted fallback is gate OFF;
no runner-up is tried on verification results. An internal closed-loop pilot
and an immutable freeze bundle are still required before benchmark evaluation.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from .paths import ROOT,sha,save
from .artifact_protocol import verify_training_artifacts
from .model import LandmarkBeliefNetwork
from .train import CachedViews,atomic_checkpoint
from .calibration_protocol import CALIBRATION_CONFIG,calibration_fold,group_summary,verify_gate
from .calibration_fusion import calibrated_logits


def calibration_known_mask(row,reliable):
    """No verify-fold object may become a BCE negative in fit-fold data."""
    map_name,object_id=row['target_instance'];fold=calibration_fold(map_name,object_id)
    same_fold=torch.zeros_like(row['labels']['rank_label_mask'],dtype=torch.bool)
    for i,detail in enumerate(row['private']['association']):
        oid=detail.get('object_id')
        same_fold[i]=oid is not None and calibration_fold(map_name,oid)==fold
    return row['labels']['rank_label_mask'] & reliable & same_fold


@torch.no_grad()
def predict_calibration(model,dataset,output,signature):
    directory=output/'predictions';directory.mkdir(exist_ok=True)
    signature_file=directory/'signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=signature:
        raise ValueError('Calibration prediction cache provenance changed')
    save(signature,signature_file);digest=sha(signature_file)
    packets=[]
    for i,entry in enumerate(dataset.entries):
        path=directory/entry['filename']
        if path.exists():
            packet=torch.load(path,map_location='cpu',weights_only=False)
            if packet['prediction_signature_sha256']!=digest or packet['source_row_sha256']!=entry['sha256']:
                raise ValueError('Calibration prediction row provenance mismatch')
        else:
            row=dataset[i]
            values={k:v.cuda()[None] for k,v in row['public']['network'].items()}
            result=model(**values)
            fields=('scores','reliable','candidate_match_logits','match_logit','offset')
            packet=dict(group=row['target_instance'],episode=row['episode'],step=row['step'],
                        output={k:result[k][0].detach().cpu().clone() for k in fields},
                        spatial=row['public']['spatial'],target_xy=row['labels']['target_xy'],
                        known=calibration_known_mask(row,result['reliable'][0].cpu()),
                        trusted_label=row['labels']['offset_valid'],source_row_sha256=entry['sha256'],
                        prediction_signature_sha256=digest)
            atomic_checkpoint(packet,path)
        packets.append(packet)
        if (i+1)%200==0:print(f'Calibration predictions {i+1}/{len(dataset)}',flush=True)
    return packets,digest


def fit_temperature(packets):
    groups=defaultdict(list);positive_groups=set();negative_groups=set();positives=negatives=0
    for packet in packets:
        if calibration_fold(*packet['group'])!='fit':raise ValueError('Temperature fitting cannot inspect verification groups')
        known=packet['known']
        if not known.any():continue
        key=tuple(packet['group']);labels=packet['trusted_label'][known].float()
        groups[key].append((packet['output']['candidate_match_logits'][known].float(),labels))
        p=int(labels.sum());n=len(labels)-p;positives+=p;negatives+=n
        if p:positive_groups.add(key)
        if n:negative_groups.add(key)
    c=CALIBRATION_CONFIG
    sufficient=(positives>=c['min_temperature_positive_labels'] and negatives>=c['min_temperature_negative_labels']
                and len(positive_groups)>=c['min_temperature_positive_groups'] and len(negative_groups)>=c['min_temperature_negative_groups'])
    losses=[]
    for temperature in c['temperatures']:
        group_losses=[]
        for values in groups.values():
            logits=torch.cat([v[0] for v in values]);labels=torch.cat([v[1] for v in values])
            group_losses.append(float(F.binary_cross_entropy_with_logits(logits/temperature,labels)))
        losses.append(dict(temperature=temperature,nll=sum(group_losses)/len(group_losses) if group_losses else None))
    chosen=min(losses,key=lambda row:row['nll'])['temperature'] if sufficient else 1.
    return chosen,dict(sufficient=sufficient,positive_labels=positives,negative_labels=negatives,
                       positive_groups=len(positive_groups),negative_groups=len(negative_groups),fit_losses=losses,
                       scope='Temperature fitted on verified candidate labels in fit groups; unknown candidates have no BCE labels')


@torch.no_grad()
def evaluate_choices(packets,choices,temperature,*,device='cuda'):
    results=[[] for _ in choices]
    for number,packet in enumerate(packets,1):
        p=dict(output={k:v.to(device) for k,v in packet['output'].items()},
               spatial={k:v.to(device) for k,v in packet['spatial'].items()})
        grid=p['spatial']['grid_world'];base=p['spatial']['base_logits'];target=packet['target_xy'].to(device)
        before=float((grid[base.argmax()]-target).norm())
        for index,choice in enumerate(choices):
            params=dict(choice,temperature=temperature,enabled=True)
            mixed=calibrated_logits(p,params)
            after=float((grid[mixed.argmax()]-target).norm())
            results[index].append(dict(group=packet['group'],a0_correct=before<=20.,a2_correct=after<=20.,
                                       a0_ne=before,a2_ne=after,changed=not torch.equal(base,mixed)))
        if number%200==0:print(f'Gate analysis {number}/{len(packets)} choices={len(choices)}',flush=True)
    return [group_summary(rows) for rows in results]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--training',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/training')
    parser.add_argument('--cache',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/cache')
    parser.add_argument('--output',type=Path,default=ROOT/'runs/a2_waldo_landmark_20261027/calibration')
    args=parser.parse_args()
    torch.set_num_threads(2);started=time.perf_counter()
    state,verification=verify_training_artifacts(args.training,args.cache,hash_cache_rows=True)
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'candidate_calibration.json').exists():
        raise ValueError('Calibration candidate exists; inspect it instead of refitting after verification')
    protocol_file=args.output/'predeclared_protocol.json'
    if protocol_file.exists() and json.loads(protocol_file.read_text())!=CALIBRATION_CONFIG:
        raise ValueError('Calibration protocol changed after collection began')
    save(CALIBRATION_CONFIG,args.output/'predeclared_protocol.json')
    analysis_signature=dict(checkpoint_sha256=sha(args.training/'best.pt'),
                            config=CALIBRATION_CONFIG,
                            code_sha256={name:sha(ROOT/'experiments/a2_waldo_landmark_belief'/name)
                                         for name in ('calibrate.py','calibration_protocol.py','calibration_fusion.py','artifact_protocol.py')})
    signature_file=args.output/'analysis_signature.json'
    if signature_file.exists() and json.loads(signature_file.read_text())!=analysis_signature:
        raise ValueError('Calibration implementation changed after collection began')
    save(analysis_signature,signature_file)
    model=LandmarkBeliefNetwork(**state['model_config']).cuda().eval().requires_grad_(False)
    model.load_state_dict(state['model'],strict=True)
    from .collected_snapshot import cache_source
    source=cache_source(args.cache)
    manifest=json.loads((args.cache/'manifest.json').read_text())
    dataset=CachedViews(args.cache,manifest,source,'calibration')
    signature=dict(checkpoint_sha256=sha(args.training/'best.pt'),cache_signature_sha256=manifest['signature_sha256'],
                   runtime_code_sha256=state['runtime_code_sha256'],prediction_implementation_sha256=sha(Path(__file__)))
    packets,prediction_sha=predict_calibration(model,dataset,args.output,signature)
    fit=[p for p in packets if calibration_fold(*p['group'])=='fit']
    verify=[p for p in packets if calibration_fold(*p['group'])=='verify']
    assert not ({tuple(p['group']) for p in fit} & {tuple(p['group']) for p in verify})
    temperature,temperature_report=fit_temperature(fit)
    choices=[dict(alpha=a,threshold=t) for a in CALIBRATION_CONFIG['alphas'] for t in CALIBRATION_CONFIG['thresholds']]
    summaries=evaluate_choices(fit,choices,temperature)
    candidates=[]
    for index,(choice,stats) in enumerate(zip(choices,summaries)):
        viable=(stats['mean_r1_difference']>0 and stats['mean_ne_difference_m']<=0
                and stats['supported_groups']>=CALIBRATION_CONFIG['min_fit_supported_groups']
                and stats['broken_group_rate'] is not None
                and stats['broken_group_rate']<=CALIBRATION_CONFIG['max_fit_broken_group_rate'])
        if viable:
            candidates.append((stats['mean_r1_difference'],-stats['mean_ne_difference_m'],
                               -stats['broken_group_rate'],-choice['alpha'],choice['threshold'],index))
    chosen_index=max(candidates)[-1] if candidates else None
    selected=choices[chosen_index] if chosen_index is not None else dict(alpha=0.,threshold=1.)
    # Persist the unique fit-only decision BEFORE inspecting verification metrics.
    save(dict(temperature=temperature_report,choices=[dict(parameters=c,summary=s) for c,s in zip(choices,summaries)],
              selected_index=chosen_index,selected_parameters=selected,selection_population='TRAIN-calibration FIT targets only'),
         args.output/'fit_search.json')
    verification_summary=evaluate_choices(verify,[selected],temperature)[0]
    gate=verify_gate(verification_summary)
    enabled=bool(chosen_index is not None and temperature_report['sufficient'] and verification['checkpoint_eligible'] and gate['passed'])
    parameters=dict(enabled=enabled,alpha=selected['alpha'] if enabled else 0.,threshold=selected['threshold'] if enabled else 1.,
                    temperature=temperature,checkpoint_sha256=signature['checkpoint_sha256'],
                    status='candidate_requires_internal_closed_loop_pilot',gate_final=False,
                    selected_fit_parameters=selected,temperature_calibration=temperature_report,
                    verify_summary=verification_summary,verify_gate=gate,
                    checkpoint_eligible=verification['checkpoint_eligible'],prediction_signature_sha256=prediction_sha,
                    protocol_sha256=sha(args.output/'predeclared_protocol.json'),
                    analysis_signature_sha256=sha(signature_file),
                    training_verification_sha256=sha(args.training/'artifact_verification.json'),
                    fit_groups=len({tuple(p['group']) for p in fit}),verify_groups=len({tuple(p['group']) for p in verify}),
                    no_verify_reselection=True,benchmark_data_used=False,seconds=time.perf_counter()-started)
    save(parameters,args.output/'candidate_calibration.json')
    print(json.dumps(parameters),flush=True)


if __name__=='__main__':
    main()
