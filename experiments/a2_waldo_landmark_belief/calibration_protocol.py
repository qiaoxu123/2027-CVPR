"""Predeclared TRAIN-calibration-only group partition and conservative risk gates."""
from collections import defaultdict
import hashlib
import math
import numpy as np

CALIBRATION_CONFIG=dict(seed=0,temperatures=[.5,1.,2.,4.],alphas=[.05,.1,.25,.5],
    thresholds=[.05,.1,.2,.3,.5,.7,.8,.9,.95,.99],bootstrap_replicates=2000,
    min_temperature_positive_labels=25,min_temperature_negative_labels=25,
    min_temperature_positive_groups=5,min_temperature_negative_groups=10,
    min_fit_supported_groups=10,max_fit_broken_group_rate=.02,
    min_verify_supported_groups=30,min_verify_a0_correct_groups=60,min_verify_gain_groups=5,
    max_verify_broken_group_wilson_upper=.05,max_verify_mean_ne_increase_m=0.,
    pilot_unique_groups=64,pilot_max_sr_losses_minus_gains=0,pilot_max_mean_ne_increase_m=0.)


def calibration_fold(map_name,object_id):
    digest=hashlib.sha256(f'a2-calibration-fit-verify:0:{map_name}:{object_id}'.encode()).digest()
    return 'fit' if int.from_bytes(digest[:8],'big')<2**63 else 'verify'


def wilson_upper(successes,trials):
    if trials==0:return 1.
    z=1.6448536269514722 # one-sided95% normal quantile
    p=successes/trials
    return (p+z*z/(2*trials)+z*math.sqrt(p*(1-p)/trials+z*z/(4*trials*trials)))/(1+z*z/trials)


def group_summary(rows):
    groups=defaultdict(list)
    for row in rows:groups[tuple(row['group'])].append(row)
    differences=[];ne_differences=[];supported=baseline=broken=gains=0
    for group in sorted(groups):
        values=groups[group]
        delta=float(np.mean([float(v['a2_correct'])-float(v['a0_correct']) for v in values]))
        differences.append(delta)
        ne_differences.append(float(np.mean([v['a2_ne']-v['a0_ne'] for v in values])))
        supported+=int(any(v['changed'] for v in values))
        baseline+=int(any(v['a0_correct'] for v in values))
        broken+=int(any(v['a0_correct'] and not v['a2_correct'] for v in values))
        gains+=int(delta>0)
    if differences:
        rng=np.random.default_rng(0);a=np.asarray(differences)
        samples=rng.choice(a,size=(CALIBRATION_CONFIG['bootstrap_replicates'],len(a)),replace=True).mean(1)
        ci=np.quantile(samples,[.025,.975]).tolist()
    else:ci=[0.,0.]
    return dict(groups=len(groups),views=len(rows),supported_groups=supported,a0_correct_groups=baseline,
                broken_groups=broken,gain_groups=gains,
                mean_r1_difference=float(np.mean(differences)) if differences else 0.,gain_ci95=ci,
                mean_ne_difference_m=float(np.mean(ne_differences)) if ne_differences else 0.,
                broken_group_rate=broken/baseline if baseline else None,
                broken_group_wilson_upper95=wilson_upper(broken,baseline),
                metric='dense-grid argmax R1 within20m on cached A0-onpolicy observations, equal target-instance weights; not navigation SR',
                risk_metric='any original-correct observation broken in a target group / groups with any original-correct observation')


def verify_gate(summary):
    c=CALIBRATION_CONFIG
    checks=dict(enough_supported_groups=summary['supported_groups']>=c['min_verify_supported_groups'],
                enough_a0_correct_groups=summary['a0_correct_groups']>=c['min_verify_a0_correct_groups'],
                enough_gain_groups=summary['gain_groups']>=c['min_verify_gain_groups'],
                positive_group_gain_lower_ci=summary['gain_ci95'][0]>0,
                bounded_group_break_risk=summary['broken_group_wilson_upper95']<=c['max_verify_broken_group_wilson_upper'],
                no_mean_ne_increase=summary['mean_ne_difference_m']<=c['max_verify_mean_ne_increase_m'])
    return dict(passed=all(checks.values()),checks=checks)
