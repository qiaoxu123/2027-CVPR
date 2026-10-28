"""Publish complete frozen V3 benchmark results, never select a new candidate.

The execution path revalidates both complete populations and every saved trace
before producing tables. Imports/inference stay outside pure report helpers so
synthetic CPU tests cannot accidentally open real heldout performance.
"""
import argparse,csv,ctypes,json,os,shutil,tempfile
from pathlib import Path
from .relation_v3_reporting import compare_pairs

PERCENT_METRICS={'SR','OSR','SPL','nDTW','sDTW','policy_stop','false_stop_episode_fraction','False_Stop_Rate'}


def reporting_hashes(root):
    from .paths import sha
    tests=sorted((Path(root)/'tests_reporting_v3').rglob('*.py'))
    if not tests:raise ValueError('Reporting test inventory is missing')
    paths=[Path(root)/'experiments/a2_waldo_landmark_belief'/name for name in ('relation_v3_results.py','relation_v3_reporting.py')]+tests
    return {str(p.relative_to(root)):sha(p) for p in paths}


def publish(stage,output):
    """Atomic directory exchange on Linux keeps an old report valid on failure."""
    stage=Path(stage);output=Path(output)
    if output.is_symlink():raise ValueError('Refuse to replace a report symlink')
    if not output.exists():stage.rename(output);return
    if not output.is_dir():raise ValueError('Report destination is not a directory')
    previous=output.parent/(output.name+'_archive');previous.mkdir(exist_ok=True)
    libc=ctypes.CDLL(None,use_errno=True)
    rename=getattr(libc,'renameat2',None)
    if rename is None:raise OSError('Atomic directory exchange requires Linux renameat2; previous report preserved')
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(stage),-100,os.fsencode(output),2)!=0:
        error=ctypes.get_errno();raise OSError(error,os.strerror(error))
    # The exchange has published the complete new tree; the old tree now has
    # the staging name and remains complete even if this archival rename fails.
    stage.rename(previous/stage.name)


def metric_rows(result):
    rows=[]
    for scope,split in result['splits'].items():
        for map_name,population in [('ALL',split)]+list(split['by_map'].items()):
            for kind in ('navigation','recall'):
                for metric,value in population[kind].items():
                    percent=metric in PERCENT_METRICS or '/20m' in metric
                    scale=100 if percent else 1
                    scaled=lambda v:None if v is None else v*scale
                    interval=value['target_cluster_bootstrap95'];maps=value['map_cluster_bootstrap95']
                    rows.append(dict(split=scope,map=map_name,kind=kind,metric=metric,
                        unit='percent; difference in percentage points' if percent else 'meters',episodes=population['episodes'],
                        A0=scaled(value['A0']),A2=scaled(value['A2']),difference=scaled(value['difference']),
                        target_cluster95_low=scaled(interval[0]) if interval else None,target_cluster95_high=scaled(interval[1]) if interval else None,
                        map_cluster95_low=scaled(maps[0]) if maps else None,map_cluster95_high=scaled(maps[1]) if maps else None,
                        A0_denominator=value['A0_denominator'],A2_denominator=value['A2_denominator'],
                        target_cluster_valid_draws=value['target_cluster_valid_draws'],map_cluster_valid_draws=value['map_cluster_valid_draws']))
    return rows


def write_tables(result,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);rows=metric_rows(result)
    for name,selected in (('metrics.csv',[r for r in rows if r['map']=='ALL']),('by_map.csv',[r for r in rows if r['map']!='ALL'])):
        with (output/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(selected)
    (output/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    show=lambda v:'无分母' if v is None else f'{v:.3f}'
    lines=['# A0 与 A2 实验结果','',
        '模型及使用规则在最终验证前固定。Seen 使用全部2,470次任务，Unseen 使用全部2,697次任务。两组使用相同任务和导航规则。',
        '比例显示为百分比；差值为 A2 减 A0，单位为百分点。距离以米计，越低越好。',
        '', '| 验证集 | 指标 | A0 | A2 | 变化 | 95%差值区间 |','|---|---|---:|---:|---:|---|']
    for row in rows:
        if row['map']!='ALL' or (row['kind']=='recall' and row['metric']!='dynamic_final_R@1/20m'):continue
        low,high=row['target_cluster95_low'],row['target_cluster95_high']
        interval='无法计算' if low is None else f'[{low:+.3f}, {high:+.3f}]'
        lines.append(f"| {row['split']} | {row['metric']} | {show(row['A0'])} | {show(row['A2'])} | {show(row['difference'])} | {interval} |")
    for scope,split in result['splits'].items():
        pairs=split['SR_pairs'];mechanism=split['mechanism'];coverage=split['candidate_coverage'];latency=split['latency']
        lines.extend(['',f"{scope}：修正成功{pairs['corrected']}次（{100*pairs['corrected_fraction']:.2f}%），引入错误{pairs['broken']}次（{100*pairs['broken_fraction']:.2f}%）。",
            f"完全沿用原模型的任务占{100*mechanism['fallback_episode_fraction']:.2f}%；在全部决策中，沿用原模型的比例为{100*mechanism['fallback_step_fraction']:.2f}%。",
            f"目标出现在有效画面中的次数为{coverage['visible_target_views']}，检测框覆盖目标{coverage['point_covered_views']}次。",
            f"每次决策平均计算时间：A0 {latency['A0']['mean_ms']:.2f}毫秒，A2 {latency['A2']['mean_ms']:.2f}毫秒。此时间不包括保存图片和报告。"])
    lines.extend(['','错误停止有两种统计：False_Stop_Rate 表示主动停下的任务中有多少停错；false_stop_episode_fraction 表示全部任务中有多少停错。没有主动停止时，前一项无法计算，不记为零。',
        '目标定位只计算每条不同指令首次对应的任务，避免重复任务使结果偏高。末次观察到的位置判断可能早于最后一次移动。',
        '区间按目标分组计算；同一目标的多条任务一起抽取。逐地图结果、按地图抽取的区间、定位的全部指标及各项分母见 CSV 和 results.json。',
        '只训练了一个随机种子。区间不能说明重复训练后的稳定程度。只在 A2 实际修改过的任务中计算的结果属于补充描述，不能代替全部任务的结论。',
        '服务器另有常驻服务，缓存情况也可能不同；计算时间反映这次实验，不代表严格控制条件下的设备速度。','',
        f"两个完整验证集的成功率均有提高：{'是' if result['positive_SR_on_both_splits'] else '否'}。",
        f"两个验证集成功率差值的95%区间均完全高于零：{'是' if result['SR_target95_lower_positive_on_both'] else '否'}。",'',
        '最终结论需要同时查看成功率、路线效率和距离变化；内部小规模测试不能替代上表。',
        'A0/A1 已有完整对照保留于 [原报告](../A0_A1_PRESERVED_SUMMARY_ZH.md)。训练曲线另见同目录文件。'])
    (output/'ANALYSIS_ZH.md').write_text('\n'.join(lines)+'\n')


def plot_differences(result,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics=('SR','OSR','SPL','nDTW','sDTW')
    figure,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for axis,(scope,split) in zip(axes,result['splits'].items()):
        values=[split['navigation'][k] for k in metrics];changes=[100*v['difference'] for v in values]
        intervals=[v['target_cluster_bootstrap95'] for v in values]
        positions=list(range(len(metrics)));axis.bar(positions,changes,color=['#21865b' if v>=0 else '#bd4238' for v in changes])
        # Draw the interval directly: bootstrap intervals need not contain the point estimate.
        for x,interval in zip(positions,intervals):
            if interval is not None:axis.plot([x,x],[100*interval[0],100*interval[1]],color='black',linewidth=1.2)
        axis.axhline(0,color='gray',linewidth=.8);axis.set_xticks(positions,metrics);axis.set_ylabel('A2 minus A0 (percentage points)');axis.set_title(scope)
    figure.savefig(Path(output)/'paired_differences.png',dpi=180);figure.savefig(Path(output)/'paired_differences.svg');plt.close(figure)


def plot_examples(examples,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for example in examples:
        figure,axis=plt.subplots(figsize=(5,5),constrained_layout=True)
        for arm,color in (('A0','#536785'),('A2','#d56c35')):
            poses=example[arm];axis.plot([p[0] for p in poses],[p[1] for p in poses],'-o',markersize=2,label=arm,color=color)
        target=example['target_xy'];axis.scatter([target[0]],[target[1]],marker='*',s=120,color='#21865b',label='Target')
        axis.set_aspect('equal',adjustable='datalim');axis.set_xlabel('World X (m)');axis.set_ylabel('World Y (m)');axis.legend()
        axis.set_title(f"{example['split']} / {example['kind']} / episode {example['index']}")
        name=f"example_{example['split']}_{example['kind']}_{example['index']:06d}"
        figure.savefig(Path(output)/(name+'.png'),dpi=160);plt.close(figure)


def run():
    from .paths import ROOT,DATA,sha,bootstrap
    from .collected_snapshot import BASE
    bootstrap()
    from .relation_v3_navigation import verified_arm
    from .relation_v3_navigation_freeze import validate_frozen
    from .relation_v3_navigation_sources import code_hashes
    from .revision_eval_protocol import FULL_COUNTS
    from sbfnav.dataset import CityNavDataset,CityReferCatalog
    from sbfnav.planning_state import PlanningStateBuilder
    from graphnav.closedloop import PublicCatalog
    implementation=reporting_hashes(ROOT)
    root=BASE/'revision_v3';validate_frozen(root);freeze_sha=sha(root/'frozen.json')
    catalog=CityReferCatalog(DATA/'cityrefer');builder=PlanningStateBuilder(DATA,PublicCatalog(catalog),canvas_size=384,field_size=28)
    result=dict(status='complete_full_frozen_V3_A0_A2_comparison',freeze_sha256=freeze_sha,navigation_code_sha256=code_hashes(),splits={})
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
    output=ROOT/'reports/a2_execution_20261027/revision_v3_final';output.parent.mkdir(parents=True,exist_ok=True)
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
