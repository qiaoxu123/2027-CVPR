"""Separate existing trained visual controls from new frozen input masks."""
import json
from pathlib import Path
from graphnav.checkpointing import atomic_json,file_sha256
from scripts.geombelief_map_experiments import OUT,HERE
from scripts.geombelief_results import rows,csvwrite
from scripts.polygon_ablation_statistics import paired_rows,paired_ci,table


def summarize():
    old=json.loads((HERE/'visual_residual_unseen_validation/metrics.json').read_text());result=[];data={};stats={}
    for split in ('val_seen','val_unseen'):
        stem='calibrated_residual_full_'+('seen' if split=='val_seen' else 'unseen')
        for arm in ('real','zero','shuffle','no-belief','geometry_pose','rgb_text','current_rgb'):
            reused=arm in ('real','zero','shuffle','no-belief')
            folder=HERE/'visual_evaluations'/f'{stem}_{arm}' if reused else OUT/'visual'/arm/split
            complete=(folder/'metrics.json').exists()
            row=dict(arm=arm,split=split,status='Done' if complete else 'Pending',kind='archived_full_validation' if reused else 'frozen_input_intervention_not_fair_retraining',checkpoint_sha256=old['configuration']['calibrated_residual_sha256'],base_checkpoint_sha256=old['configuration']['checkpoint_sha256'],result_path=str(folder),training=False)
            if complete:
                metric=old['metrics'][split][arm]['episode_weighted'] if reused else json.loads((folder/'metrics.json').read_text())['metrics']
                row.update({k:metric[k] for k in ('episodes','SR','SPL','OSR','NE','False_Stop_Rate')})
                trace=rows(folder/'predictions.jsonl');assert len(trace)==(2470 if split=='val_seen' else 2697)
                data[split,arm]=trace
            result.append(row)
        for arm in ('geometry_pose','rgb_text','current_rgb'):
            if (split,arm) in data:
                d,m,ids=paired_rows(data[split,'real'],data[split,arm],'navigation',['SR','SPL'])
                stats[f'visual/{split}/real-minus-{arm}']=paired_ci(d,m,['SR','SPL'])
                stats[f'visual/{split}/real-minus-{arm}']['reference_runtime_limitation']='Real is reused from the original complete experiment as requested; new masks run now. Historical native C exact trajectory parity failed. No new same-runtime full Real was run, so very small differences cannot be uniquely assigned to masking rather than numerical runtime variation.'
        for alias in ('RGB + Geometry + Pose','Current RGB + History'):
            row=dict(next(r for r in result if r['split']==split and r['arm']=='real'))
            row.update(arm=alias,kind='exact_alias_of_archived_real_not_new_experiment');result.append(row)
    for arm in ('geometry_pose','rgb_text','current_rgb'):
        result.append(dict(arm=arm,split='Train-Seen/Train-Dev then Val-Seen/Val-Unseen',status='Pending',kind='fair_retraining',training=True,
            reason='P1 lower priority; original matched 3-epoch residual training cost 9565.5 s per arm. Three arms need about 8 GPU-hours plus full evaluation; not started during P0 pipeline. No frozen-mask result is labeled fair retraining.'))
    csvwrite('visual_input_attribution.csv',result);atomic_json(stats,OUT/'visual_input_paired_statistics.json')
    manifest_path=OUT/'experiment_manifest.json'
    if manifest_path.exists():
        manifest=json.loads(manifest_path.read_text());manifest['visual_input_experiments']=result
        manifest['visual_input_plan']=json.loads((OUT/'frozen_visual_input_plan.json').read_text())
        manifest['case_provenance']='qualitative_manifest.json; belief_figure_manifest.json; static_figure_manifest.json'
        manifest['baseline_protocol_audit']='BASELINE_PROTOCOL_UPDATE.md; baseline_source_manifest.json'
        atomic_json(manifest,manifest_path)
    report='''# GeomBelief visual input attribution

Original complete Real/Zero/Shuffle/No-belief Seen/Unseen experiments are reused by path and checkpoint SHA. RGB+Geometry+Pose and Current+History are aliases of that original Real input configuration, not extra evaluations.

New Geometry+Pose, RGB+Text and Current-RGB controls are **frozen-weight inference interventions**, not fair retraining. All retain the same frozen geometric baseline goal belief; labels refer to inputs to the residual correction module. RGB+Text Only therefore does not mean a full map-free navigation model.

Geometry+Pose masks both projected RGB/text interactions and current/history appearance scores. RGB+Text masks candidate geometric-language context, static relative logits/probability and pose distance. Both preserve common geometric ROI quality/support indicators and the residual visibility gate (current orthophoto-crop depth-validity proxy and known pose/footprint, not perspective occlusion sensing); thus this isolates RGB semantic descriptors beyond geometry/pose, not the ability to infer physical visibility from actual observation support. Current RGB disables history features, validity, quality and fusion consistently. All parameters, candidate grid, Stop0.3 and controller remain unchanged.

Zero versus No-belief equality is enforced by the existing residual's meaningful-RGB gate. It cannot be reused as a fair trained Geometry+Pose model: zeroing RGB disables the residual outright. A favorable inference mask is also not evidence a model trained from scratch on that subset would perform best.

The Real reference is historical and is not rerun, as requested. The formal C control audit observed small historical/current intermediate-pose numerical differences. No new same-runtime complete Real residual run is available; small differences in the new mask-versus-Real comparisons can therefore include runtime variation as well as masking. These comparisons are diagnostic, not an uncontaminated causal estimate of modality value. Original four-arm paired conclusions retain their original shared protocol.

Fair retraining of these three residual input schemes remains Pending, with estimated additional 8 GPU-hours from the actual archived 3-epoch runtime (9565.5s per scheme), plus navigation. No Unseen/Test tuning or weight selection is permitted. Existing calibrated checkpoint selected on Train-Dev is used throughout the new diagnostic inference.

Rates below are fractions, NE is meters. Pointwise paired bootstrap uses 10,000 draws and map clusters; only four Unseen map blocks. Existing primary Real/No-belief and Real/Shuffle intervals are preserved in ../../../../visual_residual_unseen_validation/bootstrap_results.json and not rerun.

'''
    report+=table(result,['arm','split','status','kind','episodes','SR','SPL','OSR','NE','False_Stop_Rate'])
    report+='\nThe original full validation did not meet its predeclared Unseen positive-SR/map-CI criterion: '+str(old['acceptance'])+'\n'
    (OUT/'VISUAL_INPUT_ATTRIBUTION_REPORT.md').write_text(report)
if __name__=='__main__':summarize()
