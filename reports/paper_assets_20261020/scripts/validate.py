"""Evidence checks: source preservation, probability changes, joins and figures."""
from common import *
import numpy as np
import math

def p(x):
    x=np.asarray(x,dtype=np.float64);v=np.exp(x-x.max());return v/v.sum()

def main():
    checks={}
    protected=read(OUT/'protected_sources.json')
    for r in protected:assert sha(r['path'])==r['sha256'],r['path']
    checks['formal_weights_and_reports_unchanged']=len(protected)
    cases=list(jsonl(OUT/'cases.jsonl'));assert len(cases)==9
    stats=list(csv.DictReader((OUT/'belief_change_statistics.csv').open()));checks['belief_statistic_rows']=len(stats)
    source_hashes={}
    for case in cases:
        name=case['case_id'];meta=read(OUT/'fields'/f'{name}.json');a=np.load(OUT/meta['arrays'])
        assert sha(OUT/meta['arrays'])==meta['arrays_sha256']
        assert meta['replay_matches_saved_diagnostics']
        for source,expected in meta['source_code'].items():assert sha(ARCHIVE/source)==expected,(name,source)
        for key in ('checkpoint','residual_checkpoint','source_predictions','dataset_path'):
            if not case.get(key):continue
            path=case[key];actual=source_hashes.setdefault(path,sha(path))
            skey='dataset_sha256' if key=='dataset_path' else key+'_sha256'
            assert actual==case[skey],(name,key)
        xy=a['world_xy'];assert len(xy)==784
        own=[r for r in stats if r['case_id']==name]
        controls=['real'] if case['model_identity']=='formal' else ['real','zero','shuffle','no-belief']
        for control in controls:
            previous=None
            for i,(s,b) in enumerate(zip(a[f'{control}_static'],a[f'{control}_belief'])):
                target=p(b);refs=[('static_to_observed',p(s))]
                if previous is not None:refs.append(('previous_to_current',previous))
                for comparison,ref in refs:
                    row=next(r for r in own if r['control']==control and int(r['step'])==meta['frames'][i]['step'] and r['comparison']==comparison)
                    l1=np.abs(ref-target).sum();kl=(ref*np.log(ref/target)).sum();shift=np.linalg.norm(xy[ref.argmax()]-xy[target.argmax()])
                    for key,val in [('belief_L1',l1),('KL_p_to_q_nats',kl),('top1_shift_m',shift)]:assert math.isfinite(float(row[key])) and abs(float(row[key])-val)<1e-6,(name,key)
                    assert 0<=l1<=2+1e-8 and kl>=-1e-10
                previous=target
    checks['case_fields_replayed_and_verified']=len(cases)
    checks['distinct_episode_ids']=len({tuple(c['episode_id']) for c in cases})
    checks['observed_frame_instances']=sum(len(read(OUT/'fields'/f"{c['case_id']}.json")['frames']) for c in cases)
    frames=read(OUT/'observation_index.json')['frames'];checks['distinct_observed_poses']=len({(f['map_name'],tuple(f['pose'])) for f in frames})
    geometry=read(OUT/'geometry_case.json');assert sha(OUT/geometry['arrays'])==geometry['arrays_sha256']
    for r in geometry['sources']:
        assert sha(r['checkpoint'])==r['checkpoint_sha256']
        assert sha(r['source'])==r['source_sha256']
    audit=read(OUT/'baseline_table_checks.json');assert len(audit)==5 and all(r['all_match_at_paper_precision'] for r in audit);checks['manuscript_baseline_numeric_cells']=sum(r['checked_metrics'] for r in audit)
    baseline=list(csv.DictReader((OUT/'baseline_results.csv').open()))
    assert all(not any(r[k] for k in ['NE','SR','OSR','SPL']) for r in baseline if r['evidence_layer']=='official_released_asset')
    checks['baseline_rows']=len(baseline)
    metrics=read(OUT/'offline_metric_verification.json');assert sum(r['episodes'] for r in metrics)==10448 and all(r['aggregate_matches_formal'] for r in metrics);checks['offline_episodes_verified']=10448
    groups=list(csv.DictReader((OUT/'subgroup_statistics.csv').open()))
    for split,n in [('val_seen',2470),('val_unseen',2697),('test_unseen',5281)]:
        for group in ('start_distance_bin','initial_static_error_bin'):
            assert sum(int(r['episodes']) for r in groups if r['split']==split and r['group']==group)==n
    assert sum(int(r['episodes']) for r in groups if r['group']=='visibility')==2470
    checks['subgroup_rows']=len(groups)
    for name in ('figure1_overview','figure2_geometry','figure3_visual_updates','figure4_navigation'):
        meta=read(OUT/'figures'/f'{name}.json')
        for ext in ('png','pdf'):assert sha(OUT/'figures'/f'{name}.{ext}')==meta['sha256'][ext]
        assert 'No RGB' in meta['image_use']
    checks['figures_with_pdf_png_provenance']=4
    bibliography=read(OUT/'bibliography_summary.json');assert bibliography['Verified']==48;checks['bibliography_verified']=48
    dump(OUT/'verification.json',dict(status='Passed',checks=checks));print(json.dumps(checks))

if __name__=='__main__':main()
