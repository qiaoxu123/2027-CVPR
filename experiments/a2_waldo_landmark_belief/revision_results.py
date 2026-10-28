"""Verify every frozen benchmark trace before publishing paired A0/A2 results."""
import csv,json
from dataclasses import asdict
from .paths import DATA,A0_SHA,save,sha,bootstrap,ROOT
from .collected_snapshot import BASE
from .revision_eval_protocol import FULL_COUNTS,evaluation_code_hashes,read_trace,require_population
from .revision_evaluation import rescore_row,source_entries,summarize_rows
from .revision_freeze import validate_frozen
from .revision_paired_stats import paired_statistics
bootstrap()
from graphnav.closedloop import ClosedLoopProtocol,PublicCatalog


def verified_arm(scope,arm,dataset,catalog,builder):
    root=BASE/'revision_v1';directory=root/'evaluation'/scope/arm
    complete=json.loads((directory/'complete.json').read_text());signature=json.loads((directory/'signature.json').read_text())
    entries=source_entries(dataset,scope);require_population(scope,entries);digest=sha(directory/'signature.json')
    if (signature['scope']!=scope or signature['arm']!=arm or signature['a0_sha256']!=A0_SHA
        or signature['checkpoint_sha256']!=sha(root/'training/best.pt')
        or signature['freeze_sha256']!=sha(root/'frozen.json') or signature['code_sha256']!=evaluation_code_hashes()
        or signature['calibration_sha256']!=sha(root/'final_calibration.json')
        or signature['annotation_sha256']!=sha(dataset.annotation_path) or signature['protocol']!=asdict(ClosedLoopProtocol())
        or not signature['no_target_policy_access'] or signature['episode_manifest_sha256']!=sha(directory/'episodes.json')
        or json.loads((directory/'episodes.json').read_text())!=entries
        or complete['status']!='complete' or complete['episodes']!=FULL_COUNTS[scope]
        or complete['signature_sha256']!=digest or complete['predictions_sha256']!=sha(directory/'predictions.jsonl')
        or complete['metrics_sha256']!=sha(directory/'metrics.json')):
        raise ValueError('Incomplete or changed frozen benchmark provenance')
    rows=[]
    with (directory/'predictions.jsonl').open() as combined:
        for entry in entries:
            index=entry['index'];row=read_trace(directory/'traces'/f'episode_{index:06d}.json',digest,index)
            if json.loads(next(combined))!=row:raise ValueError('Combined trace differs from committed episode')
            rescore_row(row,dataset[index],ClosedLoopProtocol(),builder,catalog);rows.append(row)
        if next(combined,None) is not None:raise ValueError('Extra combined trace episodes')
    recomputed=summarize_rows(rows,dataset);stored=json.loads((directory/'metrics.json').read_text())
    for key in ('metrics','by_map','goal_retrieval'):
        if stored[key]!=recomputed[key]:raise ValueError(f'Aggregate differs from raw traces: {key}')
    if arm=='A2' and any(stored[k]!=recomputed[k] for k in ('branch','candidate_coverage')):
        raise ValueError('Branch/coverage aggregate differs from raw traces')
    for row in rows:row['object_id']=dataset[row['index']].object_id
    return rows,stored,dict(complete_sha256=sha(directory/'complete.json'),signature_sha256=digest)


def run():
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    root=BASE/'revision_v1';validate_frozen(root/'frozen.json')
    catalog=CityReferCatalog(DATA/'cityrefer');builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
    result=dict(status='complete_full_frozen_A0_A2_comparison',splits={},freeze_sha256=sha(root/'frozen.json'),
                code_sha256=evaluation_code_hashes(),statistics_code_sha256=sha(ROOT/'experiments/a2_waldo_landmark_belief/revision_paired_stats.py'))
    table=[];maps=[]
    for scope in FULL_COUNTS:
        dataset=CityNavDataset(DATA,scope);arms=[verified_arm(scope,arm,dataset,catalog,builder) for arm in ('A0','A2')]
        a0,a2=arms[0][0],arms[1][0];statistics=paired_statistics(a0,a2)
        by_map={}
        for name in sorted({r['map_name'] for r in a0}):
            by_map[name]=paired_statistics([r for r in a0 if r['map_name']==name],[r for r in a2 if r['map_name']==name],draws=2000)
            for metric,value in by_map[name]['metrics'].items():maps.append(dict(split=scope,map=name,metric=metric,**value))
        recall={k:dict(A0=arms[0][1]['goal_retrieval'][k],A2=arms[1][1]['goal_retrieval'][k],
                       difference=arms[1][1]['goal_retrieval'][k]-arms[0][1]['goal_retrieval'][k])
                for k in arms[0][1]['goal_retrieval'] if k!='unique_queries'}
        result['splits'][scope]=dict(statistics=statistics,by_map=by_map,recall=recall,A0=arms[0][1],A2=arms[1][1],
                                     provenance=dict(A0=arms[0][2],A2=arms[1][2]))
        for metric,value in statistics['metrics'].items():table.append(dict(split=scope,metric=metric,**value))
    # This is a final report, never permission to tune on Unseen or claim causal
    # multi-seed evidence. Positive means alone are separated from uncertainty.
    result['positive_SR_on_both_splits']=all(s['statistics']['metrics']['SR']['difference']>0 for s in result['splits'].values())
    result['SR_cluster95_lower_positive_on_both']=all(s['statistics']['metrics']['SR']['cluster_bootstrap95'][0]>0 for s in result['splits'].values())
    output=ROOT/'reports/a2_execution_20261027/revision_v1_final';output.mkdir(parents=True,exist_ok=True)
    save(result,output/'results.json')
    for name,rows in (('metrics.csv',table),('by_map.csv',maps)):
        with (output/name).open('w') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# A0 / A2 完整闭环对照','',
           '全部参数在完整评估前冻结；Unseen仅用于最终确认。以下比例为百分比，差值为百分点，NE为米。',
           '', '| Split | 指标 | A0 | A2 | A2−A0 | 目标实例聚类95%区间 |','|---|---|---:|---:|---:|---|']
    for scope,split in result['splits'].items():
        for metric in ('SR','OSR','SPL','NE','nDTW','sDTW'):
            v=split['statistics']['metrics'][metric];factor=1 if metric=='NE' else 100;lo,hi=v['cluster_bootstrap95']
            lines.append(f"| {scope} | {metric} | {factor*v['A0']:.3f} | {factor*v['A2']:.3f} | {factor*v['difference']:+.3f} | [{factor*lo:+.3f}, {factor*hi:+.3f}] |")
        p=split['statistics']['SR_pairs'];lines.extend(['',f"{scope}：修正{p['corrected']}个Episode，引入错误{p['broken']}个。完整逐地图、Recall、False-Stop、回退、覆盖率和延迟见results.json与CSV。",''])
    lines.extend(['单种子结果；聚类区间反映目标实例采样不确定性，不能代表跨种子稳定性。',
                  '是否超过A0须结合两完整Split的SR、配对区间及SPL/NE代价判断；不能仅凭内部定位或均值宣称可靠提升。'])
    (output/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:result[k] for k in ('positive_SR_on_both_splits','SR_cluster95_lower_positive_on_both')}),flush=True)


if __name__=='__main__':run()
