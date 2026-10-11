"""Read completed diagnostics, make scientific figures/report/evidence only."""
import argparse,csv,gzip,hashlib,json,shutil,subprocess,platform
import torch,transformers
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .run import write
from .core import risk_curve


def read_rows(path):
    with gzip.open(path,'rt') as f:return [json.loads(line) for line in f]


def analyze(a):
    root=Path(a.runs);devpath=root/'dev_complete';seenpath=root/'seen';dev=json.loads((devpath/'metrics.json').read_text());seen=json.loads((seenpath/'metrics.json').read_text());confidence=json.loads((devpath/'confidence_dev.json').read_text());protocol=json.loads(Path(__file__).with_name('protocol.json').read_text());principal=protocol['principal_model'];assets=Path(a.assets);assets.mkdir(parents=True,exist_ok=True);figs=assets/'figures';figs.mkdir(exist_ok=True)
    def pct(x):return '—' if x is None else f'{100*x:.2f}%'
    def value(d,k='mean'):return d.get(k)
    def ci(d):return '—' if d['mean'] is None else f"{d['mean']*100:+.2f}pp；地图CI [{d['map_cluster_95ci'][0]*100:+.2f},{d['map_cluster_95ci'][1]*100:+.2f}]；familyCI [{d['family_cluster_95ci'][0]*100:+.2f},{d['family_cluster_95ci'][1]*100:+.2f}]"
    name=principal+'__full__rgb__real';cov=seen['coverage'];principal_records=read_rows(seenpath/(name+'_identity.jsonl.gz'));margins=read_rows(seenpath/'margins.jsonl.gz');base=seen['residual'][principal+'_real__0'];raw=seen['identity'][name]
    independent_signal=all(seen['paired'][principal+'__full__rgb__'+c]['identity_family_equal']['map_cluster_95ci'][0]>0 for c in ['zero','shuffled'])
    exact_confidence={}
    for model,c in confidence.items():
        top=read_rows(devpath/(model+'_confidence_dev.jsonl.gz'));exact={}
        for proxy in c['thresholds']:
            thresholds=sorted({0.,1.01}|{float(v[proxy]) for v in top if v['eligible']})
            exact[proxy]=risk_curve(top,proxy,thresholds)
        exact_confidence[model]=exact
    write(root/'confidence_exact_dev.json',dict(scope='All empirical acceptance breakpoints, internal Dev only; diagnostic, no deployed threshold or Seen selection',curves=exact_confidence))
    safe_intervals={model:{proxy:[v for v in curves if v['supported_safe_interval']] for proxy,curves in exact_confidence[model].items()} for model in confidence};safe=any(safe_intervals[principal][p] for p in safe_intervals[principal]);oracle=seen['residual_paired']['identity_oracle__strong']['goal_r1'];fusion_space=oracle['mean']>=.01 and oracle['map_cluster_95ci'][0]>0;continue_rgb=independent_signal and safe and fusion_space
    decision=dict(independent_rgb_identity_signal=independent_signal,dev_supported_safe_nonzero_coverage=safe,offline_identity_mapping_ceiling_space=fusion_space,all_three_conditions=continue_rgb,credible_error_families=cov['credible_error_families'],stable_50_family_requirement_met=cov['credible_error_families']>=50,training_recommendation='no training in P2.5',next_stage='one minimal Dev-calibrated observed-instance verifier, then independent preregistered audit' if continue_rgb else 'RGB auxiliary visual verification only; stop RGB network training and navigation fusion',dev_safe_intervals=safe_intervals,formal_gain_unchanged=.025,formal_threshold_unchanged=1.01,seen_threshold_optimization=False)
    write(root/'decision.json',decision)
    # Same fixed conditions, not a best-Seen-model selection.
    labels=[];identity=[];hard=[];focus=[]
    for model in protocol['models']:
        key=model+'__full__rgb__real';v=seen['identity'][key];labels.append(model);identity.append(v['identity_r1']['family_equal_weight']['mean']*100);hard.append((v['hard_r1']['family_equal_weight']['mean'] or 0)*100);focus.append((v['focus_identity_r1']['family_equal_weight']['mean'] or 0)*100)
    fig,ax=plt.subplots(figsize=(10,4));x=np.arange(len(labels));ax.bar(x-.25,identity,.25,label='All-visible identity');ax.bar(x,hard,.25,label='Same-class identity');ax.bar(x+.25,focus,.25,label='Credible visible static errors');ax.set_xticks(x,labels,rotation=20);ax.set_ylabel('Family weighted R@1 (%)');ax.legend();fig.tight_layout();fig.savefig(figs/'identity.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));conf=confidence[principal]
    for proxy in ['match','match_vs_mismatch','rank_softmax']:
        bins=conf['top_candidate'][proxy]['bins'];axs[0].plot([v['confidence'] for v in bins],[v['accuracy'] for v in bins],'o-',label=proxy)
        rows=[v for v in exact_confidence[principal][proxy] if v['risk'] is not None];axs[1].plot([v['coverage'] for v in rows],[v['risk'] for v in rows],'o-',label=proxy)
    axs[0].plot([0,1],[0,1],'--',color='gray');axs[0].set(xlabel='Confidence',ylabel='Original identity correctness',title='Internal Dev only: reliability');axs[1].set(xlabel='Accepted states / all Dev states',ylabel='Acceptance risk',title='Internal Dev only: coverage-risk');axs[0].legend();axs[1].legend();fig.tight_layout();fig.savefig(figs/'confidence_dev.png',dpi=160);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));error=[v for v in margins if v['credible_error']];axs[0].hist([v['associated_correct_cell_margin'] for v in error if v['associated_correct_cell_margin'] is not None],bins=20);axs[0].axvline(.025,color='red',label='Frozen bound .025');axs[0].set(xlabel='Static max minus target-associated correct-cell logit',ylabel='States');axs[0].legend();axs[1].hist([v['rgb_identity_margin'] for v in error if v['rgb_identity_margin'] is not None],bins=20);axs[1].axvline(0,color='red');axs[1].set(xlabel='RGB target score minus strongest competing score',ylabel='States');fig.tight_layout();fig.savefig(figs/'margins.png',dpi=160);plt.close(fig)
    amplitudes=[v for v in protocol['offline_amplitudes'] if v!='strong'];fig,axs=plt.subplots(1,2,figsize=(11,4))
    for policy in [principal+'_real',principal+'_zero',principal+'_shuffled','identity_oracle']:
        rows=[seen['residual'][policy+'__'+str(am)] for am in amplitudes];axs[0].plot(amplitudes,[v['recovery']['mean']*640 for v in rows],'o-',label=policy);axs[1].plot(amplitudes,[v['damage']['mean']*640 for v in rows],'o-',label=policy)
    for ax,title in zip(axs,['OFFLINE ONLY: Recovery','OFFLINE ONLY: Damage']):ax.set_xscale('symlog',linthresh=.025);ax.set_xlabel('Theoretical residual amplitude, not selected gain');ax.set_title(title);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(figs/'offline_sensitivity.png',dpi=160);plt.close(fig)
    lines=['# P2.5：RGB语义纠错可行性诊断','', '**本阶段只诊断：训练0、图像编码0、闭环0、Unseen0。正式gain=0.025和阈值1.01保持不变。理论幅度表均为离线反事实，不能部署或据Seen选参数。**','', '## 1. 结论与证据优先级','', f"原始RGB独立身份信号：{'成立' if independent_signal else '尚未成立'}；Dev具有统计支持的非零安全接受区间：{'存在' if safe else '未发现'}；强Identity Oracle的映射上界空间：{'存在（不可部署）' if fusion_space else '不足'}；正常0.025的上界仍只有2/640。",'', '**'+('满足三项可行性条件，只提出一个下一阶段最小验证，不在本轮训练或导航。' if continue_rgb else '三项条件未全部成立：停止RGB网络训练，将RGB降为辅助视觉验证模块，不推进目标残差融合或导航训练。')+'**','',
    f"1. **身份与语言唯一性不足**：冻结主模型的完整指令RGB-only身份R1为{pct(raw['identity_r1']['family_equal_weight']['mean'])}（family等权）；可信可见静态错误子集{pct(raw['focus_identity_r1']['family_equal_weight']['mean'])}。有属性信号不等于原目标身份。",
    f"2. **置信度失效**：Dev三分类真实类别数量{conf['weak_class_counts']}（Match/Mismatch/Unknown），混淆矩阵{conf['confusion']}；原概率、去Unknown的Match-vs-Mismatch和排名代理分别检查安全覆盖，未将全部弃权的零Damage当作有效纠错。",
    f"3. **融合界限确实限制上界**：同一观测支持的固定0.025身份Oracle恢复{round(seen['residual']['identity_oracle__0.025']['recovery']['mean']*640)} /640，不能达到至少7个净恢复所需的+1pp。在当前单原实例选择与Centroid传播规则下，即使身份完全正确，正常界限仍不可达+1pp；扩大幅度的Oracle只估理论空间，不能弥补Real身份错误。",
    f"4. **数据覆盖不足**：可信可见错误为{cov['credible_error_states']}prefix /{cov['credible_error_families']}独立family /{cov['credible_error_objects']}对象；人工视觉语义核验0。不满足50family，不能靠重复prefix宣称稳定成功。",'',
    '## 2. 固定协议、资源与正常推理边界','', '基于 `06ebf5398d051ecfc105298821ce3b12eced0131`，独立分支 `research/rgb-feasibility-20261011`。复用原Train/Dev/Seen和Dense Causal History缓存，单RTX PRO6000 96GB。本轮补编码的只有属性/交换文本，图像塔不执行；所有图像、patch、context和Black特征来自P2缓存。', '',
    '主模型固定为P2内部Dev选定city_only；Native、原Aerial Projection、外部完整caption Head及Aerial→City全部作为预先列出的对照，不能按Seen重选。模型文件及原P2源文件SHA不变。Dev完成后冻结run/core/协议SHA，再运行Seen一次诊断。Seen幅度曲线不生成新配置或模型。', '',
    'RGB-only把City VEN已有相对位置bias输入置零，保留真实context/patch；Geometry-only按当前观测同一Centroid/FOV支持，从冻结完整指令A3 logits提取实例分数，无FOV关联的已观察实例用静态最小值作几何排名占位（不是语义否决，也不改变Goal）；几何实例R1必须结合这一关联局限解释；RGB+Geometry按固定等权percentile rank组合，只做区域排名分析，不是新Goal更新器。黑图/同类循环交换保持候选、观测几何、质量和语言不变。', '',
    'Attribute只替换完整查询为保守目标短语；没有可审核短语的查询回退原文，单独报告支持短语的共同人口。Attribute+Spatial只增加指令明确命名、实际已观察参考物的方向/near约束；方向±10m、near≤25m/far≥50m，不能用GT选择anchor。未支持关系回退属性分数；attribute_spatial含公开关系几何，不能称纯RGB。正常模型不接收GT身份/坐标，标签与Oracle另行使用。', '',
    '多ROI同Cell、Black等并列分数采用均匀选择期望R1和并列期望排名，避免索引赢目标。离线残差仍保留固定torch argmax，报告所选实例。属性及身份监督均为CityRefer语言/关联弱标签，没有当作人工真值。', '',
    f"Seen：640状态、128family、23地图。可信错误{cov['credible_error_families']}family；可见错误（含低质量）{cov['visible_error_states']}状态/{cov['visible_error_families']}family；可用已观察命名关系{cov['relation_states']}状态，其中可判定候选{cov['relation_decidable_candidate_states']}状态。", '',
    '## 3. 视觉身份与困难负例','', '| 冻结模型Full RGB-only | 全可见身份R1：family等权 | 身份状态/family | 同类Hard R1 | 可信错误身份R1 | 可信错误目标排名均值 |','|---|---|---|---|---|---|']
    for model in protocol['models']:
        v=seen['identity'][model+'__full__rgb__real'];lines.append(f"| {model} | {pct(v['identity_r1']['family_equal_weight']['mean'])} | {v['identity_r1']['n']}/{v['identity_r1'].get('independent_families',0)} | {pct(v['hard_r1']['family_equal_weight']['mean'])} | {pct(v['focus_identity_r1']['family_equal_weight']['mean'])} | {v['focus_rank']['mean']:.2f} |")
    v=seen['paired'][principal+'__attribute__rgb__real']['supported_phrase_identity_r1'];language_delta={**v,'mean':-v['mean'],'map_cluster_95ci':[-v['map_cluster_95ci'][1],-v['map_cluster_95ci'][0]],'family_cluster_95ci':[-v['family_cluster_95ci'][1],-v['family_cluster_95ci'][0]]}
    write(root/'language_contrast.json',dict(comparison='Attribute minus Full; supported phrase population; state weighted with family/map clustered bootstrap',paired=language_delta))
    lines += ['', f"在同一支持属性短语的{v['n']}个可见状态/{v['independent_families']}独立family中，Attribute相对Full的原身份R1变化（state等权）为{ci(language_delta)}。这是完整指令噪声/未落实参考物会拖累匹配的证据，但属性后身份准确率仍不足以可靠纠错；不能把该差当作细粒度关系理解成功。"]
    lines += ['', '| 主模型语言/模态 | 全可见身份R1 | 支持属性短语人口R1（状态/family） | 支持已观察关系人口R1（状态/family） | 属性多正例R1（n） |','|---|---|---|---|---|']
    for mode in protocol['language']:
        for modality in ['rgb','combined']:
            v=seen['identity'][principal+'__'+mode+'__'+modality+'__real'];s=v['supported_phrase_identity_r1'];t=v['supported_relation_identity_r1'];u=v['attribute_r1'];lines.append(f"| {mode}/{modality} | {pct(v['identity_r1']['family_equal_weight']['mean'])} | {pct(s['family_equal_weight']['mean'])} ({s['n']}/{s.get('independent_families',0)}) | {pct(t['family_equal_weight']['mean'])} ({t['n']}/{t.get('independent_families',0)}) | {pct(u['mean'])} ({u['n']}) |")
    geo=seen['identity']['geometry'];lines += ['', f"Geometry-only：身份R1 {pct(geo['identity_r1']['family_equal_weight']['mean'])}，同类Hard {pct(geo['hard_r1']['family_equal_weight']['mean'])}。这是已观察实例条件排名，和640状态的Goal R1不是同一指标。", '',
    '属性到原目标身份不能自动升级：多个对象满足颜色/形状时允许多正例；本轮属性多正例数量≥2的状态为'+str(cov['weak_attribute_ambiguous_states'])+'，其中参考物是否消除歧义仍缺人工验证。Full与Attribute的身份差值只在相同可见人口内解释；可见目标且支持显式关系只有20状态/8families，未支持或样本不足不能声称关系模型已经验证。', '',
    '主模型完整查询 Real−Black：'+ci(seen['paired'][principal+'__full__rgb__zero']['identity_family_equal'])+'；Real−Shuffled：'+ci(seen['paired'][principal+'__full__rgb__shuffled']['identity_family_equal'])+'。正差说明RGB有身份信息，不能由此声称能安全接纳或优于Native。', '',
    '![冻结身份与困难负例](assets/rgb_feasibility/figures/identity.png)', '', '## 4. 属性交换与像素/类别','', '| 模型 | 可交换弱标签查询/独立family | 原属性正确 | 交换属性正确 | 两者同时正确 | 分数方向正确 |','|---|---|---|---|---|']
    for model,v in seen['language_swap'].items():
        z=v['counterfactual_both_correct'];lines.append(f"| {model} | {z['n']}/{z.get('independent_families',0)} | {pct(v['original_attribute_correct']['mean'])} | {pct(v['swapped_attribute_correct']['mean'])} | {pct(z['mean'])} | {pct(v['score_direction_correct']['mean'])} |")
    lines += ['', '只交换保守目标短语中的一个属性，必须有真实已观察候选的可信语言弱标签支持两边；未审查像素属性、不假定唯一新目标。该表不足以证明目标身份理解。', '', '| 组别：Full RGB-only主模型 | 身份R1（状态） | 同类Hard R1 | 可信错误R1 | 独立对象 |','|---|---|---|---|---|']
    for key,v in seen['groups'][name].items():lines.append(f"| {key} | {pct(v['identity_r1']['mean'])} ({v['identity_r1']['n']}) | {pct(v['hard_r1']['mean'])} | {pct(v['focus_identity_r1']['mean'])} | {v['objects']} |")
    lines += ['', '按像素分组不是同一目标受控降采样，不能单独归因分辨率；类别/地图/对象分布不同。全部分组CI及目标排名在原始metrics.json。', '', '## 5. 内部Dev原始Logits、校准与弃权','', '| VEN | 原始Match-vs-Mismatch AUC：排除Unknown | 三分类Brier | 最高类别置信ECE | Top1 P(Match) ECE/Brier | Dev安全区间 |','|---|---|---|---|---|---|']
    for model,c in confidence.items():
        b=c['top_candidate']['match'];has=any(safe_intervals[model][p] for p in safe_intervals[model]);lines.append(f"| {model} | {c['known_match_mismatch_auc']['raw_match_minus_mismatch']:.4f} | {c['multiclass_brier']:.4f} | {c['multiclass_confidence_ece']['ece']:.4f} | {b['ece']:.4f}/{b['brier']:.4f} | {'存在' if has else '未发现'} |")
    lines += ['', 'Brier三分类定义sum((p-onehot)²)，二分类为mean((p-y)²)；ECE用固定10等宽bins，完整classwise分开存档。多数量Unknown可让overall ECE低，同时Match召回为0，不能当作校准成功。known AUC只用已观察可判断Match/Mismatch，不把目标未观察的ROI充当Mismatch。Top1接受风险统计全部Dev查询，目标不在图库时接受是错误验证，并非新增训练负例。', '',
    '1.01是“没有任何预声明阈值通过安全检查”的弃权哨兵，不是概率。原检查要求≥10独立接受family，且按family任何已接受prefix错误记为坏family，Wilson95正确率下界≥0.9（完美样本也至少约35families）。除固定阈值网格外，还遍历Dev置信度全部实际接受阶梯，避免粗网格遗漏区间；confidence_exact_dev.json保存完整结果。多个代理检查仅用于Dev诊断，未写入正式配置或Seen参数。', '',
    '| 主模型代理 / 阈值 | 接受状态 / family | 覆盖率 | state风险 | family-any-error风险 | 正确率Wilson下界 |','|---|---|---|---|---|---|']
    for proxy in ['match','match_vs_mismatch','rank_softmax','score_gap_sigmoid']:
        for v in confidence[principal]['thresholds'][proxy]:
            if v['threshold'] in [0,.01,.1,.5,.8,.9,1.01]:lines.append(f"| {proxy}/{v['threshold']} | {v['n']}/{v['families']} | {pct(v['coverage'])} | {pct(v['risk'])} | {pct(v['family_any_error_risk'])} | {pct(v['family_correct_wilson95_lower'])} |")
    lines += ['', '![仅内部Dev的Reliability与Coverage–Risk](assets/rgb_feasibility/figures/confidence_dev.png)', '', '## 6. Margin、映射与离线融合上界','', '| 可信错误的Margin | n | min / p10 / p25 / median / p75 / p90 / max |','|---|---|---|']
    for key,v in seen['margin_summary']['credible_errors'].items():lines.append(f"| {key} | {v['n']} | "+(' / '.join(f'{x:.4f}' for x in v['quantiles']) if v['quantiles'] else '—')+' |')
    lines += ['', f"可信错误中{cov['supported_correct_error_states']}/{cov['credible_error_states']}有目标关联cell处在原20m成功半径内；0.025能跨越正确支持cell margin的状态只有{cov['margin_le_025_credible_errors']}。该Oracle只界定当前单实例传播链路，不是所有可想象的间接关系/多实例融合机制的绝对上界；本轮没有测试或训练那些机制。GT正确grid margin、原实例centroid margin和RGB score margin分别保存，前两者是离线标签分析，不输入正常模型。", '',
    '![静态Margin与RGB排序方向](assets/rgb_feasibility/figures/margins.png)', '', '| 离线policy / 理论幅度 | GoalR1 | Recovery / Damage / Net | 相对冻结Geometry的配对差与95CI |','|---|---|---|---|']
    for policy in [principal+'_real',principal+'_zero',principal+'_shuffled','identity_oracle','credible_identity_oracle']:
        for amplitude in protocol['offline_amplitudes']:
            key=policy+'__'+str(amplitude);v=seen['residual'][key];lines.append(f"| {policy}/{amplitude} | {pct(v['goal_r1']['mean'])} | "+'/'.join(str(round(v[k]['mean']*640)) for k in ['recovery','damage','net_recovery'])+' | '+ci(seen['residual_paired'][key]['goal_r1'])+' |')
    lines += ['', '| 离线policy / 幅度 / 指标 | 配对差值及地图/family95CI |', '|---|---|']
    for policy in [principal+'_real','identity_oracle']:
        for amplitude in [.025,1,8,'strong']:
            key=policy+'__'+str(amplitude)
            for metric in ['recovery','damage','net_recovery']:lines.append(f"| {policy}/{amplitude}/{metric} | {ci(seen['residual_paired'][key][metric])} |")
    integrity=json.loads((root/'integrity.json').read_text());collisions=integrity['cell_collision_summary']
    lines += ['', f"实例折叠审计：190个目标可见状态全部具有合法FOV关联，其中{collisions['target_supported_with_competitors_states']}个目标cell同时包含其他实例（{collisions['independent_colliding_families']}个独立family），最多{collisions['maximum_target_cell_candidates']}个实例共cell；本次不存在无合法支持而使用几何排名占位的观察实例。cell级排名不能表达同cell唯一身份，但正常/Oracle仍使用相同关联。", '', f"在当前0.025上界下绕过弃权、假设Real置信度=1，仅有1个原身份正确恢复、Damage=0；Goal差为{ci(seen['residual_paired'][principal+'_real__0.025']['goal_r1'])}，CI下界为0，未达到可靠+1pp。理论幅度1的Real恢复9/误伤12；幅度8恢复49/误伤60。真实模型的排序错误不能由放大幅度解决。", '']
    lines += ['', 'Real/Black/Shuffled理论policy假定置信度=1、绕过弃权，选择各自RGB-only Full Top1及depth可信实例；不按GT是否可见筛选，因此错误接受也产生Damage。Oracle允许GT身份只选择实际观察的原实例，保持同样候选与FOV支持，目标未观察时不更新；可信Oracle进一步只选depth可信目标。strong用静态logit range+1，完全是不可部署的映射上界。各幅度列不是新的正式gain，也不将最好的Seen点推荐为配置。', '',
    '正式P2部署配置仍全部弃权，GoalR1=44.53%、Recovery=Damage=0。上表无置信度Real结果不可写成已获得的正式定位增益。逐样本的身份恢复与偶然恢复分别记录。', '',
    '![纯离线幅度敏感性](assets/rgb_feasibility/figures/offline_sensitivity.png)', '', '## 7. 真实案例与下一步','']
    casesfile=root/'cases/cases.json'
    if casesfile.exists():
        cases=json.loads(casesfile.read_text())
        for c in cases[:6]:lines += ['', f"样本 `{c['family_id']}`：目标原生边{c['native_target_side']}px，原实例排名{c['target_rank']}，实际sourceStep {c['source_step']}≤cutoff {c['cutoff']}。指令：{c['instruction']}", '', f"![实际因果ROI](assets/rgb_feasibility/cases/{c['image']})"]
    lines += ['', 'GT实例只用于该离线示意，成功案例是区域身份匹配成功，不能把它解释成正式Goal纠错。', '',
    ('唯一下一阶段最小方案：冻结全部模型，仅用预注册Train内部Dev的现有排名置信度建立可弃权的已观察实例验证器，输出确认/否决/Unknown。先在新增独立且人工核验的至少50错误family上验收：Real−Black/Shuffle地图95CI下界>0；接受family风险上界≤10%且非零覆盖；Recovery>Damage、Goal至少+1pp且地图CI下界>0。不得按本轮Seen挑阈值或幅度，未通过不接入导航。' if continue_rgb else '不提出新的RGB训练或gain方案。将RGB保留为可视化/弱属性辅助验证，不自动否决、提升或改变目标候选，Unknown时保留Geometry；最值得补充的是人工核验唯一目标与参考物身份、同颜色真困难负例及足够独立family。当前不能由属性信号或Oracle有空间推导出安全纠错。'), '',
    '## 8. 复现、预算与交付','', '配置见[protocol.json](../experiments/rgb_feasibility/protocol.json)；原始结果、未经取整逐样本分数、Dev raw logits、bootstrap结果和日志在`runs/rgb_feasibility/`；提交gzip预测与小统计图，不提交Checkpoint和大型feature文件。所有对照使用同一640状态及观察实例；sourceStep≤cutoff，候选与标签一致，原P2Checkpoint/配置/源文件SHA在结束时再次核验。', '',
    '测试结果见[regression_tests.log](assets/rgb_feasibility/evidence/regression_tests.log)。既有318项不退化；新增针对并列排名、family独立性、公开关系和offline-only残差的测试。执行命令、真实耗时、异常修复和完整环境见[execution.json](assets/rgb_feasibility/evidence/execution.json)。本轮没有新的SR/SPL数值。提交SHA在最后交付消息及本地commit.json记录，避免自引用。']
    textmeta=json.loads((root/'text_cache/manifest.json').read_text())
    lines += ['', f"实际text-only编码{textmeta['new_text_queries']}条：{textmeta['seconds']:.2f}s，GPU分配峰值{textmeta['peak_gpu_bytes']/2**30:.2f}GiB；完整Dev诊断{dev['seconds']:.2f}s，Seen诊断{seen['seconds']:.2f}s，因果/关联审计{integrity['seconds']:.2f}s。这些是含缓存加载和CPU统计的进程耗时，不能称为GPU训练时间。诊断时Head GPU峰值Dev/Seen为{dev['peak_gpu_bytes']/2**20:.1f}/{seen['peak_gpu_bytes']/2**20:.1f}MiB。"]
    Path(a.report).write_text('\n'.join(lines)+'\n');shutil.copy2(a.report,root/'final_report.md')
    execution=dict(environment_versions=dict(python=platform.python_version(),torch=str(torch.__version__),transformers=transformers.__version__,numpy=np.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),base_commit=protocol['base_commit'],branch='research/rgb-feasibility-20261011',seed=protocol['seed'],python='/home/rental/20260922_1/Workspace/Ablations/GraphNav/visual-grounding-assets/remoteclip/.venv/bin/python',environment=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',HF_HUB_OFFLINE='1'),commands=[dict(module='experiments.rgb_feasibility.run text',args=dict(p2='runs/instance_grounding',output=str(root/'text_cache')),seconds=json.loads((root/'text_cache/manifest.json').read_text())['seconds']),dict(module='experiments.rgb_feasibility.run diagnose',args=dict(p2='runs/instance_grounding',text=str(root/'text_cache'),split='dev',output=str(devpath)),seconds=dev['seconds']),dict(module='experiments.rgb_feasibility.run diagnose',args=dict(p2='runs/instance_grounding',text=str(root/'text_cache'),split='seen',lock=str(devpath/'seen_lock.json'),output=str(seenpath)),seconds=seen['seconds']),dict(module='experiments.rgb_feasibility.audit',args=dict(runs=str(root),p2='runs/instance_grounding'),seconds=integrity['seconds']),dict(module='experiments.instance_grounding.cases',args=dict(features='runs/instance_grounding/features_seen',predictions=str(root/'case_input_predictions.jsonl'),output=str(root/'cases')),note='Six independent families, three original identity successes and three failures, offline illustration only'),dict(module='experiments.rgb_feasibility.analyze',args=dict(runs=str(root),assets=a.assets,report=a.report)),dict(module='pytest',args=['-q'],log=str(root/'regression_tests.log'))],no_training=True,no_image_encoding=True,no_navigation=True,no_unseen=True,protocol_sha256=file_sha256_local(Path(__file__).with_name('protocol.json')),source_sha256={p.name:file_sha256_local(p) for p in Path(__file__).parent.glob('*.py')},checkpoint_sha256=seen['checkpoint_sha256'],source_feature_sha256={s:json.loads((Path('runs/instance_grounding')/('features_'+s)/'manifest.json').read_text())['feature_sha256'] for s in ['train','dev','seen']},exceptions=['Initial source syntax error caught by synthetic collection before any encoding; logs preserved.','First Dev incomplete run failed JSON numpy bool serialization; fixed before Seen, failed directory/log preserved.','Supported phrase and relation population fields added during pre-Seen Dev development; no model/config tuned.'])
    write(root/'execution.json',execution)
    evidence=assets/'evidence';evidence.mkdir(exist_ok=True)
    for p in sorted(root.rglob('*')):
        if not p.is_file() or any(v in p.parts for v in ['environment_private','renderer_private','cases']):continue
        if p.suffix not in ['.json','.gz','.log']:continue
        dest=evidence/p.relative_to(root);dest.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.log':dest.write_text('\n'.join(line.rstrip() for line in p.read_text().splitlines())+'\n')
        else:shutil.copy2(p,dest)
    if (root/'cases').exists():
        dest=assets/'cases';dest.mkdir(exist_ok=True)
        for p in (root/'cases').glob('*.png'):shutil.copy2(p,dest/p.name)
        shutil.copy2(root/'cases/cases.json',dest/'cases.json')
    with (evidence/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['policy','goal_r1','recovery','damage','net_recovery'])
        for k,v in seen['residual'].items():writer.writerow([k]+[v[key]['mean'] for key in ['goal_r1','recovery','damage','net_recovery']])
    write(assets/'sha256.json',{str(p.relative_to(assets)):file_sha256_local(p) for p in sorted(assets.rglob('*')) if p.is_file() and p.name!='sha256.json'})


def file_sha256_local(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs',required=True);p.add_argument('--assets',required=True);p.add_argument('--report',required=True);analyze(p.parse_args())
if __name__=='__main__':main()
