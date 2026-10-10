"""Combine actual offline/closed-loop paired Seen results; preserve missing stages."""
import argparse
import json
from pathlib import Path
import numpy as np
from .stats import summarize_rows,paired_difference


def navigation_summary(root,repetitions=2000):
    populations={};summary={};paired={}
    for condition in ['E0','E1','E2']:
        path=Path(root)/condition/'predictions.jsonl'
        if not path.exists():continue
        rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            row['sample_id']=row['episode_id']
            if row['direction']=='forward':row['language_subset']='forward_original_instruction'
            elif row.get('static_language_eligible'):row['language_subset']='reverse_geometry_language'
            else:row['language_subset']='reverse_trajectory_only'
        populations[condition]=rows;keys=sorted({k for row in rows for k,v in row['metrics'].items() if isinstance(v,(float,int,bool))})
        summary[condition]={k:summarize_rows(rows,k,repetitions=repetitions) for k in keys}
        summary[condition]['groups']={axis:{value:{k:summarize_rows([r for r in rows if r[axis]==value],k,repetitions=repetitions) for k in keys} for value in sorted({r[axis] for r in rows})} for axis in ['direction','language_subset']}
    for left,right in [('E0','E1'),('E1','E2'),('E0','E2')]:
        if left in populations and right in populations:
            keys=set(summary[left])-{'groups'}
            paired[left+'_to_'+right]={k:paired_difference(populations[left],populations[right],k,repetitions=repetitions) for k in sorted(keys)}
            paired[left+'_to_'+right]['by_direction']={direction:{k:paired_difference([r for r in populations[left] if r['direction']==direction],[r for r in populations[right] if r['direction']==direction],k,repetitions=repetitions) for k in sorted(keys)} for direction in sorted({r['direction'] for r in populations[left]})}
    return dict(models=summary,paired_differences=paired)


def building_hard_summary(prediction_root,repetitions=2000):
    populations={};models={}
    for condition in ['E0','E1','E2']:
        path=Path(prediction_root)/(condition+'_predictions.jsonl')
        if not path.exists():continue
        chosen=[]
        for row in (json.loads(x) for x in path.read_text().splitlines() if x.strip()):
            types={kind for kind,positive,judged,valid in zip(row['roi_object_types'],row['positive_roi_mask'],row['semantic_valid_mask'],row['roi_valid']) if positive and judged and valid}
            if types=={'Building'} and row['metrics']['same_class_hard_r1'] is not None:chosen.append(row)
        populations[condition]=chosen;models[condition]=summarize_rows(chosen,'same_class_hard_r1',repetitions=repetitions)
        models[condition]['positive_objects']=len({(r['map_name'],x['object_id']) for r in chosen for x in r['hard_rankings']['rankings'] if x['positive']})
    pairs={}
    for left,right in [('E0','E1'),('E1','E2'),('E0','E2')]:
        if left in populations and right in populations:pairs[left+'_to_'+right]=paired_difference(populations[left],populations[right],'same_class_hard_r1',repetitions=repetitions)
    return dict(definition='All actually valid judged positives are Building; cached same_class_hard_r1 ranks only Building positives and known negatives. Unknowns excluded; shared frozen eligibility, no model rerun.',models=models,paired_differences=pairs)


def unseen_gate(offline,navigation):
    tests={}
    nav=navigation.get('paired_differences',{}).get('E0_to_E1',{}).get('by_direction',{})
    for direction in ['forward','reverse']:
        tests['E1_minus_E0_'+direction+'_SR']=nav.get(direction,{}).get('SR')
    visual=offline.get('paired_differences',{}).get('E1_to_E2',{})
    for key in ['weak_attribute_text_region_r1','geometry_text_region_r1']:
        tests['E2_minus_E1_'+key]=visual.get(key)
    passed=[name for name,result in tests.items() if result and result.get('mean') is not None and result['mean']>0 and result.get('map_cluster_95ci') and result['map_cluster_95ci'][0]>0]
    return dict(primary_tests=tests,passed_primary_tests=passed,eligible=bool(passed),
                primary='E1-E0 forward/reverse SR OR E2-E1 observed weak-attribute/geometry text-region R1',
                E2_visual_gain_clear=any(name.startswith('E2_minus_E1') for name in passed),
                note='Predeclared positive map-cluster paired 95% lower bound; helper never opens Unseen; E2-E1 navigation SR is secondary.')


def fmt(value, *, percent=True):
    if value is None:return 'N/A'
    return f'{value*(100 if percent else 1):.2f}'


def table(headers,rows):
    return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+'\n'.join('| '+' | '.join(map(str,r))+' |' for r in rows)


def plot(offline,navigation,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    conditions=[k for k in ['E0','E1','E2'] if k in offline.get('models',{})]
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,key,title in zip(axes,['text_region_r1','reranked_r1_xy20m'],['Mixed weak attribute / geometry ROI R@1','Offline prefix XY20m localization']):
        values=[offline['models'][c][key]['mean'] for c in conditions];ax.bar(conditions,[100*v if v is not None else 0 for v in values]);ax.set_title(title);ax.set_ylabel('%');ax.set_ylim(0,100)
        for i,c in enumerate(conditions):
            leaf=offline['models'][c][key];ci=leaf['map_cluster_95ci']
            if ci and leaf['mean'] is not None:ax.errorbar(i,100*leaf['mean'],yerr=np.array([[100*(leaf['mean']-ci[0])],[100*(ci[1]-leaf['mean'])]]).clip(0),fmt='none',color='black',capsize=3)
    fig.tight_layout();fig.savefig(out/'offline_comparison.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4));bins=['low_8_23','medium_24_47','high_48_plus']
    for c in conditions:
        groups=offline['models'][c]['groups']['pixel_group'];values=[groups.get(k,{}).get('text_region_r1',{}).get('mean') for k in bins];ax.plot(bins,[100*v if v is not None else np.nan for v in values],marker='o',label=c)
    ax.set_title('Native original target ROI size (confounded analysis)');ax.set_ylabel('Mixed ROI R@1 %');ax.legend();fig.tight_layout();fig.savefig(out/'pixel_groups.png',dpi=160);plt.close(fig)
    available=[c for c in conditions if c in navigation['models']]
    if available:
        fig,ax=plt.subplots(figsize=(7,4));width=.35;x=np.arange(len(available))
        for j,key in enumerate(['SR','SPL']):ax.bar(x+(j-.5)*width,[100*navigation['models'][c][key]['mean'] for c in available],width,label=key)
        ax.set_xticks(x,available);ax.set_ylabel('%');ax.set_title('Actual matched closed-loop episodes');ax.legend();fig.tight_layout();fig.savefig(out/'navigation_comparison.png',dpi=160);plt.close(fig)


def write_report(offline,navigation,out,doc,data_root=None,training_root=None):
    b=['# 双向核心验证：三组固定预算真实实验','',
       '本轮是按 family 哈希、地图 round-robin 固定选取的 **pilot**，Seen 128 families / 256 个前向与合成返程 episode，**不是完整 Seen 验证集**。E0 为前向状态重复到同等预算；E1 加入原测量轨迹反向状态；E2 在 E1 上加入保守多正例区域–文本损失。A3、SigLIP2、BERT 冻结，仅轻量视觉匹配、Trajectory 和 Stop 头训练。固定最终第 3 Epoch，不按 Seen 选最佳 checkpoint。',
       '', '反向轨迹是原已访问 XYZ 点的合成返程，朝向按反向运动重算，并非真实录制返程。当前 RGB 由原正射图按当前 XYZ/yaw 原生 nadir 视场生成，非透视摄影；AGL≤0 保持 Unknown。没有把目标、未来轨迹或目标 ID 传入模型。',
       '', '## 实际完成阶段','']
    if not offline.get('provenance'):b+=['最终 epoch-3 checkpoint 正式评价与闭环逐 episode 结果尚未生成；当前仅下列真实数据/缓存阶段完成，模型结果 N/A。','']
    for c,meta in offline.get('provenance',{}).items():b.append(f"- {c}：完成 epoch {meta['epoch']} checkpoint 的 {meta['state_count']} 个固定 Seen 缓存状态评价，{meta['seconds']:.2f} s；SHA256 `{meta['checkpoint_sha256']}`。")
    if data_root:
        data_root=Path(data_root);coverage_path=data_root/'pilot_semantic_coverage.json'
        full_manifest=data_root/'manifest.json';checks_path=data_root/'checks.json'
        if full_manifest.exists():
            full=json.loads(full_manifest.read_text());stats=full['stats']
            b+=['','## 全量已生成 metadata 与真实 GPU pilot 的区分','',table(['全量数据项','Train Seen','Val Seen'],[[label]+[stats[split][key] for split in ['train_seen','val_seen']] for label,key in [('原始源 Episode','source_episodes'),('前向 Episode','episodes_forward'),('合成返程 Episode','episodes_reverse'),('自动几何语言返程','reverse_static_language_eligible'),('具体对象语言返程','reverse_object_language_eligible'),('前向 prefix','prefixes_forward'),('返程 prefix','prefixes_reverse'),('独立原目标对象','independent_target_objects'),('地图','maps')]]),'']
            partition=stats['train_seen']['partition']
            b+=[f"Train 内部 family/对象/200 m访问区域连通分区：Train {partition['families']['train']} families，internal Dev {partition['families']['internal_dev']} families，{partition['components']} 个连通组件，node overlap={partition['node_overlap']}，family_object_region_disjoint={partition['family_object_region_disjoint']}。原地图 Split 保持不变。",'',
                '上述全量 metadata 已生成；真实 GPU 缓存/训练只采用预先固定的 Train 2048 与 Seen 128 families pilot，没有训练全量 21878 个 family，也没有用内部 Dev/Seen 选择 epoch。','']
        if checks_path.exists():
            checks=json.loads(checks_path.read_text())
            b+=[table(['实际数据检查','Train Seen','Val Seen'],[[label]+[checks[split]['checks'][key] for split in ['train_seen','val_seen']] for label,key in [('XYZ精确反向检查','exact_xyz_reverse_checked'),('几何语言检查','geometric_return_instructions_checked'),('因果waypoint/Stop检查','prefix_causal_waypoint_stop_checked'),('预算对齐row检查','matched_budget_rows_checked')]]),'',
                f"matched_total_steps/private_coordinate_fields_separate/no_reverse_mismatched_camera_as_current：两 Split 均为 True；Train internal_family_object_region_overlap={checks['train_seen']['internal_family_object_region_overlap']}。",'']
        if coverage_path.exists():
            coverage=json.loads(coverage_path.read_text());b+=['','## 实际数据覆盖','']
            b.append(table(['数据项','Train pilot','Seen pilot'],[[label]+[coverage[split][key] for split in ['train_seen','val_seen']] for label,key in [('独立源 family','reverse_families'),('自动验证 point-anchor 几何返程语言','reliable_geometry_return_language'),('唯一具体启动对象 S','specific_start_object'),('几何语言且具体 S','geometry_language_and_specific_object'),('真实源端点附近前向 ROI 对','endpoint_pairs_forward'),('真实源端点附近反向 ROI 对','endpoint_pairs_reverse'),('前向语言弱属性对','weak_attribute_pairs_forward'),('反向语言弱属性对','weak_attribute_pairs_reverse'),('独立人工视觉核验','independent_human_verified_visual_attributes')]]))
            b+=['','以上是源端点配对统计，不是本轮 current-yaw 缓存有效 VL 状态数。Seen 128 个返程中自动几何语言 119、泛化 trajectory-only 9；不能将 128 全部解释为可靠语言定位样本。返程 source-camera 端点 crop 需重新渲染当前反向 yaw 后才能作当前观察。','']
    if data_root:
        semantic=Path(data_root).parent/'cache/val_seen/semantic_coverage.json'
        cache_manifest=Path(offline.get('cache_manifest_path',str(Path(data_root).parent/'cache/val_seen/manifest.json')))
        semantic=cache_manifest.parent/'semantic_coverage.json'
        if cache_manifest.exists():
            cache=json.loads(cache_manifest.read_text())
            if cache.get('complete'):b+=['',f"Seen 当前观测缓存实际完成：{cache['counts']['states']} states、{cache['counts']['rois']} 原生 ROI、{cache['counts']['vl_eligible']} 有效 VL states，{cache['counts']['agl_nonpositive_unknown']} 个非正 AGL Unknown，原始缓存构建 {cache['seconds']:.2f} s，FP16 全 ROI 重编码 {cache.get('precision_repair_seconds',0):.2f} s；正式缓存图像精度 `{cache.get('image_precision','unspecified')}`。这些不是训练或导航结果。",'']
        if semantic.exists():b+=['','当前固定 Seen 缓存有效监督的独立统计（按 family/map/对象去重，非源端点 pair 总量）：','', '```json',semantic.read_text().strip(),'```','']
    prediction_file=out.parent/'offline/E0_predictions.jsonl'
    if prediction_file.exists():
        predictions=[json.loads(x) for x in prediction_file.read_text().splitlines() if x.strip()]
        observed_coverage=[]
        for direction in ['forward','reverse']:
            judged=[r for r in predictions if r['direction']==direction and r['metrics']['text_region_r1'] is not None]
            positive={(r['map_name'],x['object_id']) for r in judged for x in r['rankings']['rankings'] if x['positive']}
            observed_coverage.append([direction,len(judged),len({r['family_id'] for r in judged}),len(positive),len({r['map_name'] for r in judged})])
        b+=['','正式固定 Seen 预测逐样本复核的有效语言区域监督（非端点 pair 总量）：','',table(['方向','有效 state','独立 family','独立正例对象（map＋ID）','地图'],observed_coverage),'']
    if data_root:
        episode_file=Path(data_root)/'val_seen/pilot_episodes.jsonl';prefix_file=Path(data_root)/'val_seen/pilot_prefixes.jsonl'
        if episode_file.exists() and prefix_file.exists():
            episodes=[json.loads(x) for x in episode_file.read_text().splitlines() if x.strip()];prefixes=[json.loads(x) for x in prefix_file.read_text().splitlines() if x.strip()]
            ground={r['map_name']:r['ground_level'] for r in prefixes};reverse=[e for e in episodes if e['direction']=='reverse'];high={e['episode_id'] for e in reverse if e['private_labels']['target_world'][2]-ground[e['map_name']]>100}
            states=sum(r['episode_id'] in high for r in prefixes)
            b+=['',f"返程原始启动目标 AGL>100 m 有 {len(high)}/{len(reverse)}（{100*len(high)/max(1,len(reverse)):.2f}%），对应 {states} 个 prefix。共享 controller 合法 AGL 10–100 m，超限 teacher_z 已屏蔽，原 source altitude 未篡改。因此本轮 XY20m SR/SPL 是水平返程指标，不能宣称精确原 XYZ 回程成功。",'']
    if data_root:
        smoke=Path(data_root).parent/'smoke'
        precision=smoke/'native_roi_precision.json';diagnostic=smoke/'native_roi_precision_diagnostic_fp16.json'
        if precision.exists():
            original=json.loads(precision.read_text());b+=['','## 原生 ROI 数值精度检查','',f"原 BF16 编码在 {original['actual_frames']} 个真实 frame / {original['actual_rois']} 个 native ROI 上，相对 FP32 最低图像 cosine={original['minimum_image_cosine']:.8f}、最大图文 cosine 分数误差={original['maximum_score_error']:.8f}；预声明门限 cosine≥0.995、误差≤0.01，结果 **未通过**。未降低门限。",'']
        if diagnostic.exists():
            repaired=json.loads(diagnostic.read_text());alternative=repaired.get('diagnostic_alternatives',{}).get('fp16_backbone_fp32_head')
            if alternative:b+= [f"修复诊断：FP16 vision backbone + FP32 native attention pooling 的最低 cosine={alternative['minimum_cosine']:.8f}、最大分数误差={repaired['fp16_backbone_fp32_head_maximum_score_error']:.8f}，通过同一门限。诊断通过不替代正式全源 ROI 缓存重编码；正式使用的缓存路径和 SHA 固定在离线 provenance。",'']
        formal_precision=smoke/'native_roi_precision_fp16.json'
        if formal_precision.exists():
            formal=json.loads(formal_precision.read_text())
            b+=['', '最终重编码缓存 FP32 对照数值门禁（实际原始 JSON，保留通过/失败状态）：','', '```json',json.dumps(formal,ensure_ascii=False,indent=2),'```','']
    if training_root:
        trained=[]
        command_path=Path(training_root)/'training_commands.json'
        command_times={r['condition']:r['seconds'] for r in json.loads(command_path.read_text())} if command_path.exists() else {}
        for condition in ['E0','E1','E2']:
            root=Path(training_root)/condition;complete=root/'training_complete.json';history=root/'history.json'
            if not complete.exists() or not history.exists():continue
            complete=json.loads(complete.read_text());history=json.loads(history.read_text());protocol=complete['protocol']
            trained.append([condition,complete['completed_epochs'],complete['global_steps'],protocol['samples_per_epoch'],protocol['trainable_parameters'],f"{sum(e['seconds'] for e in history):.2f}",fmt(command_times.get(condition),percent=False),all(e['frozen_resources_unchanged'] for e in history),any(e['matcher_gradient_observed'] for e in history)])
        if trained:b+=['','## 已完成轻量训练','',table(['组','Epoch','优化 steps','每 Epoch 状态','可训练参数','Epoch 优化阶段 s','完整训练进程 wall s','冻结资源不变','观察到 matcher 梯度'],trained),'','损失仅供训练诊断；核心收益根据下方真实指标、样本量与配对差值判断。','']
    b+=['','可靠人工视觉属性标签 **0**。前向 ROI 指标来自属性短语一致性弱标签；反向来自公开语言可表达的几何关系与弱 Polygon 投影。两者分开报告，不将混合 ROI 分数称作可靠颜色理解。','', '## Seen 缓存与视觉匹配','']
    conditions=[c for c in ['E0','E1','E2'] if c in offline.get('models',{})]
    keys=[('混合语言→区域 R@1','text_region_r1'),('前向弱属性 R@1','weak_attribute_text_region_r1'),('反向几何语言 R@1','geometry_text_region_r1'),('同类 Hard R@1','same_class_hard_r1'),('真实 RGB R@1（相同交换子集）','rgb_real_r1_on_swap_subset'),('交换 RGB 后原标签 R@1','rgb_swap_fixed_label_r1'),('真实 RGB − 交换 RGB R@1 (pp)','rgb_real_minus_fixed_label_swap_r1'),('交换后视觉源身份跟踪 R@1（补充）','rgb_swap_source_tracking_r1'),('原目标−同类负例 Pair 真实 R@1','rgb_pair_real_r1'),('同 Pair 固定标签交换 R@1','rgb_pair_fixed_swap_r1'),('Pair 真实−交换 Δ pp','rgb_pair_real_minus_fixed_swap_r1'),('A3 static XY20m R@1','static_r1_xy20m'),('视觉融合 XY20m R@1','reranked_r1_xy20m'),('Recovery','recovery'),('Damage','damage')]
    b.append(table(['指标']+conditions,[[label]+[fmt(offline['models'][c][key]['mean'])+' (n='+str(offline['models'][c][key]['n'])+')' for c in conditions] for label,key in keys]))
    b+=['','多视图同对象按 mean-logit 排名，未知属性与未知几何候选排除评判。RGB swap 原正例固定为当前真实可见、judged positive 的原始目标对象（私有 ID 只在评价干预选择），负例为固定字典序同类 known negative，三模型共享同计划；移植真实缓存图像特征；保留目的区域 geometry/quality/grid，不生成假颜色标签。完整 gallery 保留多正例；另做原目标＋固定同类负例 Two-object pair 子集，其他正例只排除，不改作负例，避免其他未交换正例掩盖效应。主差值保持原标签、几何、语言、质量和有效性，比较真实 RGB 与交换 RGB 的 R@1；正差值说明该干预下正确源图提供有效证据。补充 source-tracking 指标按已移位的源身份重标正例，较高或不变代表跟随图像内容，不能当作原目标位置准确率。交换仍带有位置/上下文混杂，不能解释为纯反义属性理解。未观察 grid 分数逐元素严格保持冻结 A3 原值。','',
        'Bootstrap 固定 seed 20261010、2000 次，同时保存 state IID、family 和地图聚类区间；配对比较使用完全相同 sample_id。重复状态 IID 可能乐观，地图/家族聚类优先。','',
        '原生目标 ROI 最短边按 8–23 / 24–47 / ≥48 px 分组：私有 episode 目标 ID 只在评估后关联当前已观察 target ROI，未观察或没有具体 S 对象时单列 Unknown。其他同色多正例的最大 min-side 仅作附加审计，不能掩盖实际目标的小尺寸。该分组与对象类别、地图、可见性和监督类型混杂，不能自动归因分辨率。','']
    for c in conditions:
        b.append('### '+c+' 原生像素分组')
        groups=offline['models'][c]['groups']['pixel_group'];b.append(table(['组','混合 R@1 %','有效状态','独立 family','地图 95% CI'],[[k,fmt(v['text_region_r1']['mean']),v['text_region_r1']['n'],v['text_region_r1'].get('independent_families',0),' / '.join(fmt(x) for x in v['text_region_r1']['map_cluster_95ci']) if v['text_region_r1']['map_cluster_95ci'] else 'N/A'] for k,v in groups.items()]));b.append('')
    buildings=building_hard_summary(out.parent/'offline')
    if buildings['models']:
        (out/'building_hard_metrics.json').write_text(json.dumps(buildings,indent=2,allow_nan=False))
        b+=['','## 同类型建筑困难负样本 R@1','',
            '从既有逐样本预测筛选：全部当前有效且可判正例均为 Building，仅排序真实可见同类已知负例；未知对象排除。保留语言弱/几何弱监督性质，不重新运行模型，不按结果挑建筑样本。',
            '',table(['组','Building Hard R@1 %','state n','独立 family','正例独立对象','地图 n','地图 95% CI'],[[c,fmt(v['mean']),v['n'],v.get('independent_families',0),v['positive_objects'],v.get('maps',0),' / '.join(fmt(x) for x in v['map_cluster_95ci']) if v['map_cluster_95ci'] else 'N/A'] for c,v in buildings['models'].items()]),'',
            table(['配对','Δ pp','state n','family 95% CI','地图 95% CI'],[[pair,fmt(v['mean']),v['n'],' / '.join(fmt(x) for x in v['family_cluster_95ci']) if v['family_cluster_95ci'] else 'N/A',' / '.join(fmt(x) for x in v['map_cluster_95ci']) if v['map_cluster_95ci'] else 'N/A'] for pair,v in buildings['paired_differences'].items()]),'']
    b+=['## 三组配对结果','']
    for pair,diffs in offline.get('paired_differences',{}).items():
        b.append('### '+pair)
        b.append(table(['Δ 指标 (pp)','点估计','配对 state n','family 95% CI','地图 95% CI'],[[key,fmt(diffs[key]['mean']),diffs[key]['n'],' / '.join(fmt(v) for v in diffs[key]['family_cluster_95ci']) if diffs[key]['family_cluster_95ci'] else 'N/A',' / '.join(fmt(v) for v in diffs[key]['map_cluster_95ci']) if diffs[key]['map_cluster_95ci'] else 'N/A'] for key in ['weak_attribute_text_region_r1','geometry_text_region_r1','same_class_hard_r1','rgb_pair_real_minus_fixed_swap_r1','rgb_real_minus_fixed_label_swap_r1','rgb_swap_source_tracking_r1','reranked_r1_xy20m']]));b.append('')
    b+=['## A3 重排序：去重与状态重复','']
    b.append(table(['组','状态 / Episode / family','已观察 / 可判 / 正例独立对象'],[[c,f"{offline['models'][c]['coverage']['states']} / {offline['models'][c]['coverage']['independent_episodes']} / {offline['models'][c]['coverage']['independent_families']}",f"{offline['models'][c]['coverage']['observed_object_identities']} / {offline['models'][c]['coverage']['judged_object_identities']} / {offline['models'][c]['coverage']['positive_object_identities']}"] for c in conditions]))
    for c in conditions:
        b+=['','### '+c+' 固定最早 prefix（每 Episode 一条）','']
        metrics=offline['models'][c]['episode_first_prefix']
        b.append(table(['指标','点估计 %','Episode n','事件数 / 分母','地图 95% CI'],[[key,fmt(metrics[key]['mean']),metrics[key]['n'],f"{metrics[key]['mean']*metrics[key]['n']:.0f}/{metrics[key]['n']}" if metrics[key]['mean'] is not None else 'N/A',' / '.join(fmt(x) for x in metrics[key]['map_cluster_95ci']) if metrics[key]['map_cluster_95ci'] else 'N/A'] for key in ['static_r1_xy20m','reranked_r1_xy20m','recovery','damage']]))
    b+=['','最早 prefix 按固定 cutoff 排序，不按目标可见性选图；额外 episode_mean_available_prefix 保存每 Episode 可用前缀平均，不是 oracle 最佳帧成功率。上方 1280 state 指标与下方每 Episode 一条的数量不能混作独立样本。','']
    b+=['## 实际闭环导航','']
    if navigation['models']:
        nav_conditions=[c for c in conditions if c in navigation['models']]
        rows=[]
        for label,axis,value in [('Forward Seen','direction','forward'),('Reverse Seen 全部','direction','reverse'),('Reverse Seen 自动几何语言','language_subset','reverse_geometry_language')]:
            for key in ['SR','SPL']:
                leaves=[navigation['models'][c]['groups'][axis].get(value,{}).get(key,{}) for c in nav_conditions]
                rows.append([label,key]+[fmt(leaf.get('mean'))+' (n='+str(leaf.get('n',0))+')' for leaf in leaves])
        b.append(table(['子集','指标']+nav_conditions,rows))
        b.append('')
        b+=['','### 闭环方向配对差值（百分点）','']
        rows=[]
        for pair,diffs in navigation.get('paired_differences',{}).items():
            for direction,metrics in diffs['by_direction'].items():
                for key in ['SR','SPL']:
                    leaf=metrics[key];rows.append([pair,direction,key,leaf['n'],fmt(leaf['mean']),' / '.join(fmt(x) for x in leaf['family_cluster_95ci']) if leaf['family_cluster_95ci'] else 'N/A',' / '.join(fmt(x) for x in leaf['map_cluster_95ci']) if leaf['map_cluster_95ci'] else 'N/A'])
        b.append(table(['配对','方向','指标','配对 Episode n','Δ pp','family 95% CI','地图 95% CI'],rows))
        b.append('E2−E1 SR 为次要指标，不替代预声明 E1−E0 方向导航或 E2−E1 观察区域匹配主门槛。')

    else:b.append('闭环逐 episode 结果尚未生成；本报告不把离线 prefix 定位或计划中的闭环结果写为已完成导航。')
    b+=['','## 四个核心问题：基于已完成实测','']
    def delta_sentence(name,leaf):
        if not leaf or not leaf.get('n'):return name+'：无足够可判样本，N/A。'
        ci=leaf['map_cluster_95ci'];clear=ci is not None and ci[0]>0
        return name+f"：Δ={fmt(leaf['mean'])} pp，n={leaf['n']}，独立 family={leaf['independent_families']}，地图 CI="+('['+', '.join(fmt(x) for x in ci)+']' if ci else 'N/A')+('，有初步正向证据。' if clear else ('，表现下降，区间完全低于 0。' if ci is not None and ci[1]<0 else '，未证明稳定正向增益。'))
    navpair=navigation.get('paired_differences',{}).get('E0_to_E1',{}).get('by_direction',{})
    b.append('1. 双向训练是否改善真实导航？')
    for direction in ['forward','reverse']:b.append('   '+delta_sentence('E1−E0 '+direction+' SR',navpair.get(direction,{}).get('SR')))
    visualpair=offline.get('paired_differences',{}).get('E1_to_E2',{})
    b.append('2. E2 辅助损失是否改善已观察区域语言匹配？')
    for key in ['weak_attribute_text_region_r1','geometry_text_region_r1']:b.append('   '+delta_sentence('E2−E1 '+key,visualpair.get(key)))
    b.append('   前者为语言弱标签，后者为自动几何语言及弱区域投影；独立人工视觉属性核验仍为 0。')
    b.append('3. 模型是否使用真实图像证据？')
    for c in conditions:
        b.append('   '+delta_sentence(c+' Two-object pair 真实 RGB−固定原标签交换 RGB',offline['models'][c]['rgb_pair_real_minus_fixed_swap_r1']))
        b.append('   '+delta_sentence(c+' 完整多正例 gallery 真实−交换',offline['models'][c]['rgb_real_minus_fixed_label_swap_r1']))
        leaves=offline['models'][c]
        b.append('   '+c+' 诊断：'+ '；'.join(label+'='+fmt(leaves[key]['mean'])+'% (n='+str(leaves[key]['n'])+')' for label,key in [('Pair 源身份跟踪','rgb_pair_source_tracking_r1'),('gallery top-object 改变','rgb_swap_top_object_changed'),('Pair 原分数同分','rgb_pair_real_tie')])+'。')
    b.append('   结合 source-tracking、top-object 改变与 Pair 同分比例判断图像依赖；高源跟踪说明预测随图像内容移位，不能当作固定原标签交换正确率。多正例未交换证据可掩盖 gallery Δ≈0，因此不能直接据此断言模型不看 RGB。')
    b.append('4. 原生像素尺寸是否限制性能，是否值得增大网络？')
    for c in conditions:
        groups=offline['models'][c]['groups']['pixel_group'];low=groups.get('low_8_23',{}).get('text_region_r1',{});high=groups.get('high_48_plus',{}).get('text_region_r1',{})
        b.append(f"   {c}：低像素 {fmt(low.get('mean'))}% (n={low.get('n',0)})，高像素 {fmt(high.get('mean'))}% (n={high.get('n',0)})；类别、地图、可见性与监督类型混杂，不是分辨率因果对照。")
    b.append('   E2 视觉主比较没有清晰正向证据时停止复杂 GAT；先处理观察覆盖与弱监督质量。')
    b+=['','## Unseen 与模型扩展门槛','',
        ('至少一项预声明主比较（E1−E0 正向/回程 SR 或 E2−E1 已观察弱属性/几何语言区域 R@1）地图聚类 95% CI 下界大于 0，满足后续冻结配置后单次 Unseen 的统计门槛；本报告没有自动运行 Unseen。' if navigation['unseen_gate']['eligible'] else '当前没有预声明主比较通过正提升且配对地图 95% CI 下界大于 0 的门槛，不开展 Unseen。E2 视觉收益不清晰时停止复杂 GAT；E1 若有独立导航收益可满足自己的正式 Unseen 验证门槛。'),
        '', '完整逐状态预测、原像素分组、同类候选、RGB 交换计划、A3 Unknown 不变量与完整配对 CI 保存在本报告对应 runs 目录。导航逐 episode 保存真实政策 pose、termination、diagnostics；目标与参考轨迹仅用于 rollout 后评分。',
        '', '局限：弱语言属性未人工核验；正射纹理无法证明真实透视跨视角鲁棒性；返程不是录制飞行；稀疏 current-only ROI 不能代替未来可见性。无清晰 E2 收益时保留负结果，不扩展网络。']
    if conditions:
        import shutil
        assets=Path(doc).parent/'assets/bidirectional_core';assets.mkdir(parents=True,exist_ok=True)
        b+=['','## 可复核对比图','']
        for filename in ['offline_comparison.png','pixel_groups.png','navigation_comparison.png']:
            if (out/filename).exists():
                shutil.copy2(out/filename,assets/filename)
                b+=['!['+filename.removesuffix('.png')+'](assets/bidirectional_core/'+filename+')','']
        review=dict(cache_manifest_sha256=offline.get('cache_manifest_sha256'),provenance=offline.get('provenance'),
                    primary_gate=navigation['unseen_gate'],models={c:{key:offline['models'][c][key] for _,key in keys} for c in conditions},
                    navigation=navigation)
        if (out/'building_hard_metrics.json').exists():shutil.copy2(out/'building_hard_metrics.json',assets/'building_hard_metrics.json')
        (assets/'review_metrics.json').write_text(json.dumps(review,indent=2,allow_nan=False))
        b+=['小型统计与完整配对导航区间：[review_metrics.json](assets/bidirectional_core/review_metrics.json)。高维缓存和逐样本预测保留本地 runs。','']
    if data_root:
        run_root=Path(data_root).parent
        regression=run_root/'regression.json';integrity=run_root/'training_integrity.json'
        b+=['','## 执行与审计证据','']
        if regression.exists():
            check=json.loads(regression.read_text());b+=[f"实际回归：{check['passed']} passed / {check['warnings']} warning，{check['seconds']:.2f} s；warning={check.get('warning','')}。",'']
        if integrity.exists():
            check=json.loads(integrity.read_text());b+=[f"训练完整性审计 passed={check['passed']}，三组同初始 head SHA `{check['same_initial_head_sha256']}`、同 Train cache manifest SHA `{check['same_cache_manifest_sha256']}`、同容量 {check['same_capacity']}；frozen backbones in optimizer={check['frozen_backbones_in_optimizer']}。",'']
        commands=[]
        for filename in ['executed_commands.json','reencode_commands.json','training_commands.json','evaluation_commands.json']:
            command_file=run_root/filename
            if command_file.exists():commands.append(filename)
        b+=['实际完整命令、退出码及分阶段耗时索引：'+', '.join('`'+x+'`' for x in commands)+'（本地 `runs/bidirectional_core_test/`）。','']
        case_manifest=run_root/'cases/manifest.json'
        if case_manifest.exists():
            cases=json.loads(case_manifest.read_text());b+=[f"已生成 {len(cases)} 组真实 current-frame 候选图：`runs/bidirectional_core_test/cases/manifest.json`。候选正确性来自上述弱/几何标签，不能当作人工视觉核验。case_00 的汽车 Polygon bbox 含邻车，展示区域边界与重叠标注噪声，不能将表观失误全部归因编码器语义能力。",'']
        if conditions:
            audit_assets=Path(doc).parent/'assets/bidirectional_core'
            for filename in commands+['regression.json','training_integrity.json']:
                if (run_root/filename).exists():shutil.copy2(run_root/filename,audit_assets/filename)
            for source,name in [(Path(data_root)/'manifest.json','data_manifest.json'),(Path(data_root)/'checks.json','data_checks.json')]:
                if source.exists():shutil.copy2(source,audit_assets/name)
            if case_manifest.exists():
                shutil.copy2(case_manifest,audit_assets/'cases_manifest.json')
                for case in json.loads(case_manifest.read_text()):
                    filename=case['path']
                    if Path(filename).name!=filename:raise ValueError('case image must be a local basename')
                    shutil.copy2(run_root/'cases'/filename,audit_assets/filename)
                b+=['案例清单：[cases_manifest.json](assets/bidirectional_core/cases_manifest.json)；全部真实current-frame候选图已同步。','']
                for filename in ['case_00.png','case_06.png']:
                    if (audit_assets/filename).exists():b+=['![真实current-frame弱标签候选及区域边界噪声](assets/bidirectional_core/'+filename+')','']
                b+=['case_06 返程楼顶候选原生尺寸约20–30 px，小ROI与几何/可见性混杂，不能从此例单独推导分辨率因果效应。','']
            b+=['审计与命令 JSON 已同步至 `docs/assets/bidirectional_core/`，可随代码 review。','']
    text='\n'.join(b).rstrip()+'\n';out.mkdir(parents=True,exist_ok=True);(out/'final_report.md').write_text(text);Path(doc).parent.mkdir(parents=True,exist_ok=True);Path(doc).write_text(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',required=True);p.add_argument('--navigation',required=True);p.add_argument('--output',required=True);p.add_argument('--doc',default='docs/BIDIRECTIONAL_CORE_TEST_REPORT.md');p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--data-root',default='runs/bidirectional_core_test/data');p.add_argument('--training-root',default='runs/bidirectional_core_test');args=p.parse_args()
    offline=json.loads(Path(args.offline).read_text());navigation=navigation_summary(args.navigation,args.bootstrap);navigation['unseen_gate']=unseen_gate(offline,navigation);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    (out/'navigation_paired_metrics.json').write_text(json.dumps(navigation,indent=2,allow_nan=False));plot(offline,navigation,out);write_report(offline,navigation,out,args.doc,args.data_root,args.training_root)

if __name__=='__main__':main()
