"""Finish queued one-off experiments, verify results, report and synchronize."""
import json,time,subprocess,sys,traceback
from graphnav.checkpointing import atomic_json
from scripts.geombelief_map_experiments import OUT,HERE


def main():
    gates=['map_complete.json','control_complete.json','visual_complete.json','controller_context_audit.json','belief_figure_manifest.json','shuffle_effectiveness.json']
    while not all((OUT/g).exists() for g in gates):time.sleep(30)
    from scripts.geombelief_results import summarize
    from scripts.geombelief_visual_results import summarize as visual_summary
    from scripts.geombelief_verify import main as verify
    summarize();visual_summary();verify()
    stats=json.loads((OUT/'paired_statistics.json').read_text());stats.update(json.loads((OUT/'visual_input_paired_statistics.json').read_text()));atomic_json(stats,OUT/'paired_statistics.json')
    def effect(key,metric):
        s=stats[key]['metrics'][metric];lo,hi=s['map_cluster_episode_weighted_ci95']
        label='positive pointwise map-cluster support' if lo>0 else 'negative pointwise map-cluster support' if hi<0 else 'CI crosses zero'
        return f'{100*s["delta"]:+.3f} pp (map-cluster 95% CI [{100*lo:+.3f}, {100*hi:+.3f}]; {label})'
    text='# GeomBelief core experiment conclusions\n\nAll core experiments complete. Three matched seeds per map geometry arm; original Correct Geometry reused. New formal-control evaluations use the same runtime after exact historical trajectory parity failed. No original checkpoint/result was overwritten. New visual-input controls are frozen interventions; fair visual-input retraining remains Pending.\n\n'
    text+='## 1. Does correct geometry beat language/name/type alone?\n\n'
    for split in ('val_seen','val_unseen'):
        for endpoint,metric in [('localization','R@1'),('navigation','SR')]:
            for right in ('name_type','shuffled'):
                key=f'{endpoint}/{split}/correct-minus-{right}/seed_mean';text+=f'- {split} {metric}, Correct − {right}: '+effect(key,metric)+'.\n'
    text+='\n## 2. Which simple spatial relations add value?\n\n'
    for pair in ('distance-minus-name_type','direction-minus-name_type','distance_direction-minus-name_type','distance_direction-minus-distance','distance_direction-minus-direction'):
        for endpoint,metric in [('localization','R@1'),('navigation','SR')]:
            key=f'{endpoint}/val_unseen/{pair}/seed_mean';text+=f'- Unseen {metric}, {pair}: '+effect(key,metric)+'.\n'
    text+='\n## 3. Localization, visual correction, trajectory and Stop\n\n'
    for pair in ('oracle_direct-minus-predicted_direct','oracle_guarded_blend-minus-predicted_guarded_blend','predicted_waypoint-minus-predicted_direct','predicted_guarded_blend-minus-predicted_waypoint','predicted_guarded_blend-minus-predicted_direct','predicted_guarded_blend-minus-rule_arrival'):
        key='controller/val_unseen/'+pair
        for metric in ('SR','SPL'):text+=f'- Unseen {metric}, {pair}: '+effect(key,metric)+'.\n'
    for arm in ('geometry_pose','rgb_text','current_rgb'):
        text+=f'- Frozen calibrated residual, Unseen SR Real − {arm}: '+effect('visual/val_unseen/real-minus-'+arm,'SR')+'.\n'
    text+='''
Oracle intervention quantifies recoverable error under a specified diagnostic controller, not a uniquely additive fraction attributable to localization. OSR−SR describes reaching but not ending in the success region, not automatically a learned-Stop failure. Geometry ablations use a transferred pretrained controller; formal controller controls use the original complete C weights; calibrated visual experiments use the previously trained residual. These different configurations cannot have their gains added together.

## Recommended paper tables and scope of claims

1. Main comparison table: GeomBelief is the paper name of the existing frozen formal GraphNav-V1 C model. Keep the archived official three-split results and exact protocol. Report current-runtime validation deltas separately; do not replace official results with a favorable new run or seed.
2. Core mechanism table: six language/map input schemes, all three seeds, Seen/Unseen R@1/R@4/R@8/NE and navigation SR/SPL/OSR/NE/False-Stop. Include map-cluster intervals for the three predeclared contrasts. The same output grid and network capacity are retained.
3. Controller table: six policies, same current runtime, with Oracle rows clearly diagnostic. Keep raw Learned Waypoint, Direct, Guarded-Blend and rule arrival irrespective of sign.
4. Visual table: archived four-arm full validation plus explicitly labeled frozen residual-input interventions. Same visibility gate and same geometric baseline are disclosed. Do not describe masked inference as retrained modality baselines.
5. Qualitative figure: actual language, reference polygons, real visited RGB and executed trajectories, including visual degradation and false stop. Correct/shuffled posterior uses predeclared Seed0/query0, not the best-looking seed.

The experiments establish the usefulness (or lack thereof) of supplied map geometry conditional on language and landmark semantics. They do not alone isolate a language–geometry interaction from an unconditional spatial prior, nor distinguish a probabilistic goal belief from every possible point-regression locator. Keep those stronger claims out of the core innovation statement without further controlled evidence. Previous Polygon-vs-BBox results remain unchanged and are not promoted to a new positive claim.

## Statistical and data limitations

Map-cluster bootstrap resamples whole maps; paired episode bootstrap is supplementary. Only four Unseen map blocks from two cities exist. Pointwise intervals are not familywise corrected. Seed means and sample standard deviations are reported; seeds are not extra independent maps. Per-map and leave-one-map-out results are retained. Shuffle effectiveness is measured explicitly because repeated public reference bundles can survive query permutation.

## Remaining experiments

- Fair retraining for Geometry+Pose / RGB+Text / Current-RGB residual inputs: Pending. Existing three-epoch training required 9565.5 seconds per scheme, so approximately 8 GPU-hours for three single-seed budget-matched runs, plus full evaluation. Three seeds would triple that budget. No unfinished row is marked Done.
- External strict official-weight/API replications retain their existing documented blocks; this round refreshed source/protocol evidence and did not train baselines. GeoNav (DeepSeek Flash) remains a substituted API experiment, never GPT-4o.
'''
    (OUT/'GEOMBELIEF_CORE_CONCLUSIONS.md').write_text(text)
    for path in (HERE/'docs/PAPER_EXPERIMENT_TODO.md',HERE/'reports/paper_experiments/PAPER_EXPERIMENT_SUMMARY.md'):
        body=path.read_text()
        body+='\nGeomBelief core acceptance: Done (15 new map trainings, 30 new map full-split evaluations, 12 matched-runtime formal control full-split evaluations, 6 new frozen visual-input full-split evaluations). Original three Correct Geometry seeds and eight visual control split runs reused. Fair visual-input retraining remains Pending. Conclusions and acceptance evidence: `reports/paper_experiments/geombelief_remaining/GEOMBELIEF_CORE_CONCLUSIONS.md`, `acceptance.json`.\n'
        path.write_text(body)
    atomic_json(dict(status='CoreDone',fair_visual_retraining='Pending',gates=gates),OUT/'progress.json')
    tests=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_geombelief_inputs.py','tests/test_geombelief_control.py','tests/test_geombelief_visual_inputs.py'],cwd=HERE,stdout=(OUT/'unit_tests.log').open('w'),stderr=subprocess.STDOUT)
    assert tests.returncode==0
    from scripts.geombelief_sync import main as sync
    sync();atomic_json(dict(status='Done',core=True,remote_verified=True,fair_visual_retraining='Pending'),OUT/'finalization_complete.json')
    # Include final completion receipt in the branch after asset verification.
    from scripts.geombelief_sync import commit_push
    commit_push('Record GeomBelief core experiment completion')

if __name__=='__main__':
    try:main()
    except Exception:
        atomic_json(dict(status='Blocked',error=traceback.format_exc()),OUT/'finalization_failure.json');raise
