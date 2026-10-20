"""Recompute archived tensor statistics and join existing paired rollout results.

No training, neural inference, new navigation evaluation, or revised metrics.
"""
from collections import Counter,defaultdict
import subprocess
from common import *


def main():
    protected=FINAL/'PROTECTED_INPUTS.json'
    if not protected.exists():dump(protected,protected_snapshot())
    else:
        for name,value in read(protected).items():assert sha(ROOT/name)==value,name
    case_list=cases();lookup={r['case_id']:r for r in case_list}
    archived=list(csv.DictReader((ASSETS/'belief_change_statistics.csv').open()))
    statindex={(r['case_id'],r['control'],int(r['step']),r['comparison']):r for r in archived}
    paired=list(csv.DictReader((ROOT/'visual_residual_unseen_validation/paired_results.csv').open()))
    aggregate_checks=[]
    for split in ['val_seen','val_unseen']:
        short=split.removeprefix('val_')
        for control in ['zero','shuffle','no-belief']:
            subset=[r for r in paired if r['split']==split and r['control']==control]
            assert len({r['episode_id'] for r in subset})==len(subset)
            for branch,prefix in [('real','real'),(control,'control')]:
                path=ROOT/f'visual_evaluations/calibrated_residual_full_{short}_{branch}/metrics.json'
                saved=read(path)
                assert saved['checkpoint_sha256']==BASE_SHA and saved['calibrated_residual_sha256']==HEAD_SHA
                assert saved['metrics']['episodes']==len(subset)
                for key,column in [('SR','SR'),('SPL','SPL'),('NE','NE_m')]:
                    value=sum(float(r[f'{prefix}_{column}']) for r in subset)/len(subset)
                    assert abs(value-saved['metrics'][key])<1e-9,(path,key,value)
                aggregate_checks.append({'split':split,'comparison':control,'branch':branch,'metrics_sha256':sha(path)})
    pairidx={(r['split'],tuple(json.loads(r['episode_id'])),r['control']):r for r in paired}
    frames=read(FINAL/'RGB_FRAME_INDEX.json')['frames'];rgb={(r['case_id'],r['step']):r for r in frames}
    rows=[];initial=[];verified=0;maxadd=0;changedepisodes=[];all_mask_error=0
    for c in case_list:
        cid=c['case_id'];a=fields(cid);meta=read(ASSETS/'fields'/f'{cid}.json')
        assert sha(ASSETS/meta['arrays'])==meta['arrays_sha256']
        world=a['world_xy'];target=np.asarray(c['target_xyz'][:2]);distance=np.linalg.norm(world-target,axis=1)
        controls=[k[:-7] for k in a if k.endswith('_belief')]
        initial_error=float(distance[a['real_static'][0].argmax()]);any_changed=False
        for control in controls:
            previous=None
            for i,frame in enumerate(meta['frames']):
                step=frame['step'];s=a[control+'_static'][i];b=a[control+'_belief'][i];delta=a[control+'_residual'][i];observed=a[control+'_observed'][i].astype(bool)
                ps,pb=probability(s),probability(b);pr=probability(a['real_belief'][i])
                err=float(distance[pb.argmax()]);old=float(distance[ps.argmax()]);change=bool(pb.argmax()!=ps.argmax())
                if control=='real':any_changed|=change
                addition=float(np.abs(b.astype(float)-s.astype(float)-delta.astype(float)).max());maxadd=max(maxadd,addition)
                # Tensor BF16/FP32 capture rounding is explicit; use recorded max.
                assert addition<1e-5,(cid,control,step,addition)
                maskerr=float(np.abs(delta[~observed]).max()) if (~observed).any() else 0.
                all_mask_error=max(all_mask_error,maskerr);assert maskerr==0
                for comparison,ref in [('static_to_observed',ps)]+([('previous_to_current',previous)] if previous is not None else []):
                    expected=statindex[(cid,control,step,comparison)]
                    actual={'belief_L1':float(np.abs(ref-pb).sum()),
                        'KL_p_to_q_nats':float(np.sum(ref*np.log(np.maximum(ref,1e-300)/np.maximum(pb,1e-300)))),
                        'top1_shift_m':float(np.linalg.norm(world[ref.argmax()]-world[pb.argmax()])),
                        'top1_cell_changed':int(ref.argmax()!=pb.argmax()),
                        'reference_top1_error_m':float(distance[ref.argmax()]),
                        'observed_top1_error_m':err,'residual_max_abs':float(np.abs(delta).max())}
                    for key,value in actual.items():assert abs(float(expected[key])-value)<1e-8,(cid,control,step,key)
                    verified+=1
                pair=pairidx.get((c['split'],tuple(c['episode_id']),control)) if c['model_identity']=='calibrated_residual' and control!='real' else None
                if pair:assert abs(float(pair['real_NE_m'])-c['trace']['metrics']['NE'])<1e-8
                image=rgb.get((cid,step));max_observed=float(np.abs(delta[observed]).max()) if observed.any() else 0.
                rows.append({'case_id':cid,'episode_id':':'.join(map(str,c['episode_id'])),'split':c['split'],'model_identity':c['model_identity'],
                    'base_checkpoint_sha256':c['checkpoint_sha256'],'residual_checkpoint_sha256':c['residual_checkpoint_sha256'] or '',
                    'control':control,'step':step,'tensor_index':i,
                    'initial_static_goal_error_m':initial_error,'initial_real_goal_error_m':float(distance[a['real_belief'][0].argmax()]),
                    'current_static_goal_error_m':old,'current_goal_error_m':err,
                    'final_control_goal_error_on_real_path_m':float(distance[a[control+'_belief'][-1].argmax()]),
                    'top1_cell_changed_vs_static':int(change),'top1_shift_vs_static_m':float(np.linalg.norm(world[ps.argmax()]-world[pb.argmax()])),
                    'top1_cell_changed_vs_previous':int(previous.argmax()!=pb.argmax()) if previous is not None else '',
                    'belief_L1_static_to_control':float(np.abs(ps-pb).sum()),'belief_L1_real_to_control':float(np.abs(pr-pb).sum()),
                    'KL_static_to_control_nats':float(np.sum(ps*np.log(np.maximum(ps,1e-300)/np.maximum(pb,1e-300)))),
                    'static_entropy_nats':entropy(ps),'control_entropy_nats':entropy(pb),'entropy_change_nats':entropy(pb)-entropy(ps),
                    'residual_max_abs_logits':float(np.abs(delta).max()),'residual_mean_abs_logits':float(np.abs(delta).mean()),
                    'recorded_observed_cells':int(observed.sum()),'residual_outside_recorded_observed_max':maskerr,
                    'real_navigation_SR':c['trace']['metrics']['SR'],'real_navigation_NE_m':c['trace']['metrics']['NE'],
                    'control_navigation_SR':float(pair['control_SR']) if pair else '',
                    'control_navigation_NE_m':float(pair['control_NE_m']) if pair else '',
                    'belief_scope':'matched Real executed pose prefix; dense logit numpy.argmax, not controller tie-break',
                    'navigation_scope':'independent archived rollout summary' if pair else 'only Real archived rollout available for this model/case',
                    'recorded_observed_scope':'Stage1Output.observed; CVR meaningful-feature gate is additional and not cached',
                    'rgb_exact_archive_match':image['exact_archive_crop_match'] if image else '',
                    'rgb_local_path':image['local_png'] if image else '',
                    'rgb_observation_npz_sha256':image['npz_sha256'] if image else '',
                    'fields_sha256':meta['arrays_sha256'],'source_predictions_sha256':c['source_predictions_sha256']})
                previous=pb
        if any_changed:changedepisodes.append({'case_id':cid,'episode_id':c['episode_id'],'model_identity':c['model_identity']})
        initial.append({'case_id':cid,'model_identity':c['model_identity'],'episode_id':':'.join(map(str,c['episode_id'])),
            'initial_static_error_m':initial_error,'final_real_goal_error_m':float(distance[a['real_belief'][-1].argmax()]),
            'goal_error_improvement_m':initial_error-float(distance[a['real_belief'][-1].argmax()]),
            'navigation_SR':c['trace']['metrics']['SR'],'scope':'outcome-selected illustration; no population correlation inference'})
    assert verified==len(archived)==310
    csvwrite(FINAL/'VISUAL_CAUSAL_CASES.csv',rows);csvwrite(FINAL/'INITIAL_ERROR_ANALYSIS.csv',initial)
    # Closed-loop success is read only from true paired rollout rows, never a
    # matched-path candidate hit. Preserve model identity and data scope.
    by=defaultdict(dict)
    for r in paired:by[(r['split'],r['episode_id'])][r['control']]=r
    outcomes=[];examples=[]
    for split in ['val_seen','val_unseen']:
        for control in ['zero','shuffle','no-belief']:
            rs=[r for r in paired if r['split']==split and r['control']==control]
            outcomes.append({'split':split,'control':control,'episodes':len(rs),
                'real_success_control_fail':sum(float(r['real_SR'])==1 and float(r['control_SR'])==0 for r in rs),
                'real_fail_control_success':sum(float(r['real_SR'])==0 and float(r['control_SR'])==1 for r in rs),
                'same_success':sum(float(r['real_SR'])==1 and float(r['control_SR'])==1 for r in rs),
                'same_failure':sum(float(r['real_SR'])==0 and float(r['control_SR'])==0 for r in rs)})
        strong=[]
        for (s,eid),d in by.items():
            if s==split and float(d['zero']['real_SR'])==1 and float(d['zero']['control_SR'])==0 and float(d['shuffle']['control_SR'])==0:
                strong.append({'split':split,'episode_id':eid,'Real_NE_m':float(d['zero']['real_NE_m']),
                    'Zero_NE_m':float(d['zero']['control_NE_m']),'Shuffle_NE_m':float(d['shuffle']['control_NE_m']),
                    'model_identity':'calibrated_residual','base_checkpoint_sha256':BASE_SHA,'residual_checkpoint_sha256':HEAD_SHA,
                    'per_step_tensors_available_in_selected_bundle':False})
        examples+=strong
    csvwrite(FINAL/'PAIRED_NAVIGATION_COUNTS.csv',outcomes);csvwrite(FINAL/'REAL_WINS_ZERO_AND_SHUFFLE.csv',examples)
    inference=read(ROOT/'visual_evaluations/calibrated_residual_full_unseen_real/metrics.json')
    assert inference['checkpoint_sha256']==BASE_SHA and inference['calibrated_residual_sha256']==HEAD_SHA
    codechecks={name:{'expected_sha256':h,'current_sha256':sha(ROOT/name),'matches_recorded_inference':sha(ROOT/name)==h}
        for name,h in inference['implementation_sha256'].items() if (ROOT/name).exists()}
    assert all(r['matches_recorded_inference'] for r in codechecks.values())
    dump(FINAL/'CODE_PROVENANCE.json',{'inference_metrics':str(Path('visual_evaluations/calibrated_residual_full_unseen_real/metrics.json')),
        'inference_metrics_sha256':sha(ROOT/'visual_evaluations/calibrated_residual_full_unseen_real/metrics.json'),
        'code_checks':codechecks,'formal_protocol':inference['stage1_protocol'],
        'calibrated_config':{k:v for k,v in read(ROOT/'calibrated_residual/config.json').items() if k!='dev_episodes'},
        'local_model_weights_present':False,'weight_identity_scope':'archived per-case SHA, matching inference manifest and existing replay parity; no new full-model replay on this host'})
    dump(FINAL/'CAUSAL_SUMMARY.json',{'archived_comparisons_verified':verified,'per_frame_control_rows':len(rows),
        'cases':len(case_list),'unique_episodes':len({tuple(c['episode_id']) for c in case_list}),
        'Real_changes_dense_Top1_cases':changedepisodes,'matched_path_max_s0_plus_delta_discrepancy_logits':maxadd,
        'max_residual_outside_recorded_observed_logits':all_mask_error,
        'paired_navigation_rows_read':len(paired),'paired_navigation_source_sha256':sha(ROOT/'visual_residual_unseen_validation/paired_results.csv'),
        'real_wins_zero_and_shuffle_count':len(examples),'paired_rows_match_archived_aggregate_metrics':aggregate_checks,
        'semantic_disambiguation_verified':False,
        'limits':['case selection is outcome-conditioned','True optical visibility is unknown','CVR uses geometry, pose, masks and quality as well as RGB x text',
                  'Zero changes the CVR meaningful-feature gate; it is not a support-preserving content-only test',
                  'Shuffle tests candidate-image association, not exclusively semantic object identity',
                  'Matched-path belief changes and independent rollout SR are distinct evidence']})
    print('Verified',verified,'comparisons;',len(rows),'frame/control rows; paired wins',len(examples),flush=True)


if __name__=='__main__':main()
