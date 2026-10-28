"""TRAIN-only diagnostic classifiers; never a replacement for formal A2 training."""
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from experiments.a2_waldo_landmark_belief.paths import ROOT,save,sha
from experiments.a2_waldo_landmark_belief.collected_snapshot import BASE
from experiments.a2_waldo_landmark_belief.risk_probe import FEATURE_NAMES,WEIGHTS
from experiments.a2_waldo_landmark_belief.risk_probe import public_risk_features,outcome_labels
from experiments.a2_waldo_landmark_belief.revision_calibrate import load_packet
import torch


def opened_fit_transfer():
    """Already-opened FIT diagnostic, never re-evaluate/reselect v1 VERIFY."""
    root=BASE/'revision_v1';directory=root/'calibration/packets'
    manifest=json.loads((directory/'manifest.json').read_text())
    raw=json.loads((root/'confirmation/manifest.json').read_text());source={e['filename']:e for e in raw['entries']}
    signature=sha(directory/'signature.json');features=[];labels=[];before=[];after=[];identities=[]
    torch.set_num_threads(2)
    for entry in manifest['entries']:
        origin=source[entry['filename']]
        if origin['fold']!='fit':continue
        path=directory/entry['filename']
        if sha(path)!=entry['sha256'] or sha(path.with_suffix('.sha.json'))!=entry['sidecar_sha256']:raise ValueError('Opened FIT packet changed')
        p=load_packet(path,signature,origin['sha256'])
        if p['fold']!='fit':raise ValueError('No VERIFY in transfer diagnostic')
        base,q,grid,target=(p[k][None] for k in ('base','q','grid','target'))
        x=public_risk_features(base,q,grid,p['benefit_logit'].reshape(1).sigmoid(),torch.tensor([p['available']]))
        y=outcome_labels(base,q,grid,target)
        features.append(x.numpy());labels.append(y['outcomes'].numpy());before.append(y['before_ne'].numpy());after.append(y['after_ne'].numpy())
        identities.append(dict(map_name=p['group'][0],object_id=p['group'][1]))
    if len(identities)!=1004:raise ValueError('Opened FIT population changed')
    return dict(features=np.concatenate(features),outcomes=np.concatenate(labels),before_ne=np.concatenate(before),after_ne=np.concatenate(after)),identities


def summary(identities,before,after,allow):
    change=np.where(allow,after,before);gains=(before>20)&(change<=20);harms=(before<=20)&(change>20)
    groups={}
    for i,r in enumerate(identities):groups.setdefault((r['map_name'],r['object_id']),[]).append(i)
    baseline_groups=sum(bool((before[g]<=20).any()) for g in groups.values())
    broken=sum(bool(harms[g].any()) for g in groups.values())
    return dict(corrected=int(gains.sum()),broken=int(harms.sum()),net_corrected=int(gains.sum()-harms.sum()),
        supported_groups=sum(bool(allow[g].any()) for g in groups.values()),
        gain_groups=sum(bool(gains[g].sum()>harms[g].sum()) for g in groups.values()),
        broken_groups=broken,broken_group_rate=broken/baseline_groups if baseline_groups else None,
        mean_r1_difference=float(np.mean([np.mean(gains[g].astype(float)-harms[g].astype(float)) for g in groups.values()])),
        mean_ne_difference=float(np.mean([np.mean(change[g]-before[g]) for g in groups.values()])))


def main():
    root=BASE/'revision_v2_probe';marker=json.loads((root/'complete.json').read_text())
    if marker['status']!='complete_original_TRAIN_dev_risk_probe_extraction' or marker['signature_sha256']!=sha(root/'signature.json'):raise ValueError('Complete pinned inference required')
    signature=json.loads((root/'signature.json').read_text())
    if signature['features']!=list(FEATURE_NAMES) or signature['weights']!=list(WEIGHTS):raise ValueError('Probe feature/outcome schema changed')
    data={};ids={}
    for partition,expected in (('train',31415),('dev',4618)):
        files=marker['partitions'][partition]
        if files['views']!=expected or files['npz_sha256']!=sha(root/f'{partition}.npz') or files['identities_sha256']!=sha(root/f'{partition}_identities.json'):raise ValueError('Partial/corrupt probe source')
        with np.load(root/f'{partition}.npz') as source:data[partition]={k:source[k] for k in source.files}
        ids[partition]=json.loads((root/f'{partition}_identities.json').read_text())
        if len(data[partition]['features'])!=expected or len(ids[partition])!=expected:raise ValueError('Probe source count changed')
        d=data[partition]
        if d['features'].shape!=(expected,len(FEATURE_NAMES)) or d['outcomes'].shape!=(expected,len(WEIGHTS)) or d['before_ne'].shape!=(expected,) or d['after_ne'].shape!=(expected,len(WEIGHTS)):
            raise ValueError('Invalid probe tensor shape')
        if not all(np.isfinite(a).all() for a in d.values()):raise ValueError('Nonfinite probe arrays')
        gains=(d['before_ne'][:,None]>20)&(d['after_ne']<=20);harms=(d['before_ne'][:,None]<=20)&(d['after_ne']>20)
        if not np.array_equal(gains.astype(np.int8)-harms.astype(np.int8),d['outcomes']):raise ValueError('Saved outcome labels changed')
    groups=[{(r['map_name'],r['object_id']) for r in ids[p]} for p in ('train','dev')]
    if groups[0]&groups[1]:raise ValueError('Target leakage in diagnostic classifier')
    train=data['train'];dev=data['dev'];y=train['outcomes'][:,-1];truth=dev['outcomes'][:,-1];active=truth!=0
    if set(np.unique(y))!={-1,0,1}:raise ValueError('Insufficient diagnostic outcome classes')
    models=dict(logistic=make_pipeline(StandardScaler(),LogisticRegression(C=1.,class_weight='balanced',max_iter=1000,random_state=0)),
                forest=RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=30,class_weight='balanced_subsample',random_state=0,n_jobs=2))
    reports={};available=dev['features'][:,-1]>0;transfer,transfer_ids=opened_fit_transfer();transfer_truth=transfer['outcomes'][:,-1];transfer_active=transfer_truth!=0
    for name,model in models.items():
        model.fit(train['features'],y);probability=model.predict_proba(dev['features']);classes=list(model.classes_)
        gain=probability[:,classes.index(1)];harm=probability[:,classes.index(-1)];score=gain/(gain+harm+1e-12)
        choices=[]
        for threshold in (.1,.2,.35,.5,.65,.8):
            for risk in (.01,.02,.05,.1,.2):
                allow=available&(gain>=threshold)&(harm<=risk)
                choices.append(dict(gain_threshold=threshold,harm_threshold=risk,**summary(ids['dev'],dev['before_ne'],dev['after_ne'][:,-1],allow)))
        viable=[r for r in choices if r['supported_groups']>=30 and r['broken_group_rate']<=.02 and r['mean_ne_difference']<=0]
        best=max(viable,key=lambda r:(r['mean_r1_difference'],-r['broken'], -r['mean_ne_difference'])) if viable else None
        report=dict(dev_gain_vs_harm_auc=float(roc_auc_score(truth[active]==1,score[active])) if len(np.unique(truth[active]))==2 else None,
                    choices=choices,best_dev_diagnostic=best)
        predicted=model.predict_proba(transfer['features']);pg=predicted[:,classes.index(1)];ph=predicted[:,classes.index(-1)];ps=pg/(pg+ph+1e-12)
        allow=(transfer['features'][:,-1]>0)&(pg>=best['gain_threshold'])&(ph<=best['harm_threshold']) if best else np.zeros(len(pg),dtype=bool)
        report['opened_FIT_transfer']=dict(gain_vs_harm_auc=float(roc_auc_score(transfer_truth[transfer_active]==1,ps[transfer_active])) if len(np.unique(transfer_truth[transfer_active]))==2 else None,
            parameters='Only originaldev-selected diagnostic thresholds; no FIT search; no v1VERIFY access',
            **summary(transfer_ids,transfer['before_ne'],transfer['after_ne'][:,-1],allow))
        if name=='forest':report['feature_importance']=dict(zip(FEATURE_NAMES,map(float,model.feature_importances_)))
        reports[name]=report
    old_score=dev['features'][:,FEATURE_NAMES.index('old_benefit_probability')]
    result=dict(scope='Frozen v1 proposal; fit ORIGINALTRAIN, select originaldev diagnostics, transfer only to already-opened v1FIT; NOT full20+5 A2 training, calibration, navigation or deployment',
        source_complete_sha256=sha(root/'complete.json'),targets_disjoint=True,
        opened_FIT_packet_manifest_sha256=sha(BASE/'revision_v1/calibration/packets/manifest.json'),
        opened_FIT_raw_manifest_sha256=sha(BASE/'revision_v1/confirmation/manifest.json'),
        checkpoint_sha256=signature['checkpoint_sha256'],analysis_code_sha256=sha(ROOT/'scripts/a2_risk_probe_fit.py'),
        outcome='Actual argmax20m outcome at fixed fusionweight.25; labels separately derived from offlineTRAIN target',
        train_counts={str(int(k)):int((y==k).sum()) for k in np.unique(y)},dev_counts={str(int(k)):int((truth==k).sum()) for k in np.unique(truth)},
        old_benefit_dev_gain_vs_harm_auc=float(roc_auc_score(truth[active]==1,old_score[active])) if len(np.unique(truth[active]))==2 else None,
        old_benefit_opened_FIT_maxweight_auc=float(roc_auc_score(transfer_truth[transfer_active]==1,transfer['features'][transfer_active,13])) if len(np.unique(transfer_truth[transfer_active]))==2 else None,
        opened_FIT_fixed_weight_capacity=summary(transfer_ids,transfer['before_ne'],transfer['after_ne'][:,-1],transfer['features'][:,-1]>0),
        models=reports,remaining='Need public safety head integration, full20+5 training, new independent confirmation, real pilot and frozen fullSeen/Unseen; no v1VERIFY configuration reselection')
    result['caveats']=['Balanced classifier scores are not calibrated harm probabilities.',
        'Best of30threshold pairs on dev is optimistic development selection, not independent risk evidence.',
        'Opened FIT transfer uses only originaldev-selected thresholds; no FIT search, no v1VERIFY access.',
        'Teacher and real closed-loop observations differ; fixed.25 outcome AUC cannot be equated to learned-gate outcome AUC.']
    save(result,ROOT/'reports/a2_execution_20261027/REVISION_RISK_PROBE.json')
    print(json.dumps(dict(train_counts=result['train_counts'],dev_counts=result['dev_counts'],old_auc=result['old_benefit_dev_gain_vs_harm_auc'],
        models={n:{k:v for k,v in r.items() if k not in ('choices','feature_importance')} for n,r in reports.items()})),flush=True)


if __name__=='__main__':main()
