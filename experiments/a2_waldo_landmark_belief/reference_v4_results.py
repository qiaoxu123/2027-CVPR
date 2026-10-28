"""Publish complete frozen V4 A0/A2 results without changing model or settings."""
import argparse
import json
from pathlib import Path
import shutil
import tempfile

from .relation_v3_reporting import compare_pairs
from .relation_v3_results import (PERCENT_METRICS,publish,metric_rows,write_tables,plot_differences,plot_examples)


def reporting_hashes(root):
    from .paths import sha
    root=Path(root);tests=sorted((root/'tests_reporting_v4').rglob('*.py'))
    if not tests:raise ValueError('V4 reporting test inventory missing')
    paths=[root/'experiments/a2_waldo_landmark_belief'/name for name in
           ('reference_v4_results.py','relation_v3_results.py','relation_v3_reporting.py')]+tests
    return {str(path.relative_to(root)):sha(path) for path in paths}


def run():
    from .paths import ROOT,DATA,sha,bootstrap
    from .collected_snapshot import BASE
    bootstrap()
    from .reference_v4_navigation import verified_arm
    from .reference_v4_navigation_freeze import validate_frozen
    from .reference_v4_navigation_sources import code_hashes
    from .revision_eval_protocol import FULL_COUNTS
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from graphnav.closedloop import PublicCatalog
    implementation=reporting_hashes(ROOT)
    root=BASE/'revision_v4';validate_frozen(root);freeze_sha=sha(root/'frozen.json')
    catalog=CityReferCatalog(DATA/'cityrefer');builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
    result=dict(status='complete_full_frozen_V4_A0_A2_comparison',freeze_sha256=freeze_sha,navigation_code_sha256=code_hashes(),splits={})
    provenance={};examples=[]
    for scope,count in FULL_COUNTS.items():
        dataset=CityNavDataset(DATA,scope);arms=[verified_arm(scope,arm,dataset,catalog,builder) for arm in ('A0','A2')]
        if any(len(arm[0])!=count for arm in arms):raise ValueError('Full paired benchmark population required')
        enriched=[]
        for rows,stored,proof in arms:
            enriched.append([dict(row,object_id=dataset[row['index']].object_id,instruction=dataset[row['index']].instruction) for row in rows])
        split=compare_pairs(*enriched);result['splits'][scope]=split
        for name,arm in zip(('A0','A2'),arms):
            provenance[scope+'/'+name]=arm[2]
            if (arm[1]['candidate_coverage'] is not None and arm[1]['candidate_coverage']!=split['candidate_coverage']):raise ValueError('Coverage report differs from verified aggregate')
            for key,value in split['navigation'].items():
                stored=arm[1]['metrics'][key];calculated=value[name]
                if ((stored is None)!=(calculated is None) or stored is not None and abs(stored-calculated)>1e-12):raise ValueError('Navigation statistic differs from original verified aggregate')
            # Cross-check final Recall against original unique-query definition.
            if arm[1]['goal_retrieval']['unique_queries']!=split['unique_queries']:raise ValueError('Unique query count differs from verified aggregate')
            for key,value in arm[1]['goal_retrieval'].items():
                if key!='unique_queries' and abs(split['recall'][key][name]-value)>1e-12:raise ValueError('Recall differs from original verified aggregate')
        selected=set()
        for before,after in zip(*enriched):
            kind='corrected' if after['metrics']['SR']>before['metrics']['SR'] else 'broken' if after['metrics']['SR']<before['metrics']['SR'] else 'fallback' if not any(d.get('a2_updated',False) for d in after['step_diagnostics']) else None
            if kind is not None and kind not in selected:
                selected.add(kind);target=dataset[before['index']].target_position
                examples.append(dict(split=scope,kind=kind,index=before['index'],target_xy=list(map(float,target[:2])),A0=before['poses'],A2=after['poses']))
    result['provenance']=provenance
    result['report_source_sha256']=implementation
    result['positive_SR_on_both_splits']=all(s['navigation']['SR']['difference']>0 for s in result['splits'].values())
    result['SR_target95_lower_positive_on_both']=all(s['navigation']['SR']['target_cluster_bootstrap95'][0]>0 for s in result['splits'].values())
    result['example_selection']='First source-index example in each corrected/broken/fallback category; descriptive only, not selected by gain magnitude.'
    output=ROOT/'reports/a2_execution_20261027/revision_v4_final';output.parent.mkdir(parents=True,exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.build-',dir=output.parent));write_tables(result,stage)
    curves=json.loads((root/'training/curve_manifest.json').read_text())
    for extension in ('png','svg'):
        source=root/f'training/training_curves.{extension}'
        if sha(source)!=curves[extension+'_sha256']:raise ValueError('Verified training curves changed')
        shutil.copy2(source,stage/source.name)
    plot_differences(result,stage);plot_examples(examples,stage)
    (stage/'examples.json').write_text(json.dumps(examples,indent=2,allow_nan=False)+'\n')
    validate_frozen(root)
    if sha(root/'frozen.json')!=freeze_sha:raise ValueError('Final freeze changed during reporting')
    if reporting_hashes(ROOT)!=implementation:raise ValueError('Reporting source/test inventory changed during reporting')
    files={str(p.relative_to(stage)):sha(p) for p in stage.iterdir() if p.is_file() and p.name!='complete.json'}
    (stage/'complete.json').write_text(json.dumps(dict(status=result['status'],freeze_sha256=freeze_sha,files_sha256=files),indent=2)+'\n')
    publish(stage,output)
    print(json.dumps(dict(status=result['status'],positive_SR_on_both_splits=result['positive_SR_on_both_splits'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.parse_args();run()
