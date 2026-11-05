"""Chinese report and explicitly non-training weak associations for the diagnostic split."""
import argparse
import json
from pathlib import Path
import shutil


def pct(value):return 'NA' if value is None else f'{value*100:.2f}%'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args();out=a.output;s=json.loads((out/'summary.json').read_text());run=json.loads((out/'runtime.json').read_text())
    rows=list(map(json.loads,(out/'results.jsonl').open()));base=s['base_rankings'];common=s['common_reference_rankings']['real_other_instance']
    names=dict(geometry='纯几何：入镜比例',pgvb='原PGVB：完整Polygon',pgvb_zero_shape_tokens='PGVB：形状Token置零',explicit='显式边缘/区域/背景评分',appearance='过去帧外观参考',explicit_reference='外观参考+显式评分')
    lines=['# Polygon—RGB完整实例对照与跨帧关联','',
        '日期：2026-11-05。基于 `eb4c53edad1445e76bf1f2e9c9049344e1e392f4`，工作分支 `experiment/unified-pgvb-landmark-acquisition-20261104`；不合并main。','',
        '**主要结论：原PGVB确实使用Polygon输入，上一轮只换名称不能据此判断实例区分失败。显式边缘匹配改善部分合成几何扰动区分，却没有改善真实同类实例检索；过去帧外观参考有弱监督代理增益。当前不能自动生成可靠像素金标准，也不应直接把显式评分接入已验证导航。**','',
        '## 1. 勘误、数据和对照定义','',
        '- 撤回上一轮从“只换name_text、保留Polygon A”的对照推导实例区分能力的结论。它只能说明名称敏感性，原始统计保留但解释更正。',
        '- 本轮复用141帧及固定64条StableTrackedPGVB轨迹。这64条是Val-Seen来源的具名地标诊断任务，**当前数据不可用于后续模型拟合、阈值选择或Checkpoint选择**；如果生成训练监督，必须用TRAIN按地图/物理地标隔离重新生成。本轮未训练、未重新导航、未使用原CityNav真正目标GT。',
        f'- 141帧中：93帧/43任务满足PGVB对比支持；32帧像素或可靠正负Patch不足；16帧为PGVB不支持类型。后两组记未知/不适用，不算实例匹配失败。真实同类困难负样本可构建{base["real_other_instance"]["n"]}帧/{base["real_other_instance"]["tasks"]}任务，严格过去帧参考共同子集{common["n"]}帧/{common["tasks"]}任务。',
        '- 正确A、真实同类其他实例B、平移±16/±32像素、世界坐标旋转±30°、面积保持的各向异性变形0.75/1.25。所有候选重新计算完整Polygon、局部形状Token、ROI Mask、弱正负标签、可用视觉候选UV及ROI外观池化。保持同一当前RGB、冻结权重与查询名称A。',
        '- 查询名称始终为请求实例A；B是候选身份，不能把查询一起改为B再称“错误匹配”。同名多Polygon合并为一个对象候选，Union内部空洞保留。',
        '- 困难B必须在同一实际视图可用，局部面积为A的0.5–2倍，整物体入镜比例差≤0.2，至少256稳定像素、2个正Patch、2个负Patch；最多3个B，按纯几何相似性确定，不按RGB分数或GT挑选。扰动也采用同样公平条件。',
        f'- 不足支持的扰动排除{s["excluded_perturbations"]}个；可见ROI完全不变的扰动{s["no_observable_ROI_change_controls"]}个。所有排除记录和全部141帧状态均保留。',
        '- **关键语义：真实Polygon B可能也正确贴合RGB。** 因而A优于B是“请求实例A的身份检索”代理，不能拿它直接评判轮廓配准质量。A与合成扰动比较也只是地图弱监督，原始Polygon本身未必是屋顶像素最优边界。','',
        '## 2. 冻结方法','',
        '- Geometry：仅计算公开Polygon与Footprint交集占完整Polygon比例；不读取RGB颜色、特征或图像边缘。所有方法使用相同的公共有效性筛选。此基线衡量投影支持先验，**不是全部地图身份先验**：如果直接按已知对象ID查表，地图身份本就确定，但那不能确认RGB中对应实例是否真实可见。',
        '- 原PGVB：保留原网络对整个当前有效Patch网格评分，按每个候选新Polygon的正/负区域计算平均分差。Shape Token由当前候选裁剪轮廓的(x,y,dx,dy)组成，经Cross-Attention产生视觉上下文；网络没有显式像素边缘对应或分割输出。',
        '- Token消融：同一当前RGB和候选ROI，单独将Polygon Token置零。这用于验证Polygon输入作用，**不是**代替完整错误实例对照。Token同时含图像位置及轮廓，不能将消融差全部归因于纯形状理解。',
        '- Explicit：固定0.5边界Sobel支持+0.3内外Lab差异+0.2边界/背景增益，边界容忍3像素，排除相机裁剪边缘。Canny100/200只用于图册展示，不用于评分。未在本集合调整权重。',
        '- Appearance：使用同对象公开ID在严格过去实际观测帧的Polygon区域SigLIP2描述子，与当前候选区域描述子做余弦匹配；参考需位移≥1m或yaw差≥5°，不读未来帧。',
        '- Explicit+Reference：固定0.8外观余弦+0.2显式评分。只作冻结离线诊断，没有接入Belief、Gate或Stop。',
        '- Top1平分采用1/平分候选数的期望得分，Rank用中位次，避免A总放第0个导致虚假成功。每张比较表的所有方法使用完全相同查询和候选集合；候选数量见JSON。','',
        '## 3. 同类真实困难负样本','',
        '|方法|无参考共同34查询Top1代理|有参考共同25查询Top1代理|25查询平均Rank|25查询平均A对最佳B分差|','|---|---:|---:|---:|---:|']
    for method,name in names.items():
        v=common['metrics'][method];b=base['real_other_instance']['metrics'].get(method)
        lines.append(f'|{name}|{pct(b["top1"]) if b else "NA"}|{pct(v["top1"])}|{v["mean_rank"]:.3f}|{v["mean_margin"]:.4f}|')
    lines += ['', '分差单位不同，只能在各自方法内解释，不能横向比较数值大小。主表84%=21/25是地图ID代理正确选择，不是真实实例准确率。Explicit与Appearance融合仍为84%，**没有相对纯外观参考的总体增益**；无参考Explicit 41.18%还低于PGVB 52.94%。','',
        f'排除可见性先验差异的完全入镜共同子集仅{s["fully_visible_common_reference"]["n"]}查询/{s["fully_visible_common_reference"]["tasks"]}任务：','',
        '|Geometry|PGVB|Explicit|Appearance|Explicit+Reference|','|---:|---:|---:|---:|---:|',
        '|'+ '|'.join(pct(s['fully_visible_common_reference']['metrics'][m]['top1']) for m in ('geometry','pgvb','explicit','appearance','explicit_reference'))+'|','',
        '按任务组聚类的配对bootstrap（探索性代理，不是独立泛化检验）：','']
    for comparator,v in s['task_group_bootstrap'].items():
        lines.append(f'- 融合参考相对{names[comparator]}：{v["task_groups"]}任务组等权平均差{v["task_balanced_top1_difference"]*100:.2f}个百分点，95%区间{[round(x*100,2) for x in v["CI95"]]}。任务组平均不等于上表逐查询平均；地图/纹理仍可能相关。')
    lines += ['', '## 4. 平移和形状扰动','',
        '|候选组|查询/任务|Geometry Top1|PGVB Top1|Explicit Top1|Explicit两两胜率|','|---|---:|---:|---:|---:|---:|']
    for family,name in [('position_shift','平移'),('shape_rotation','旋转'),('shape_scale','各向异性变形')]:
        g=base[family];pair=s['perturbation_pairwise'][family]
        lines.append(f'|{name}|{g["n"]}/{g["tasks"]}|{pct(g["metrics"]["geometry"]["top1"])}|{pct(g["metrics"]["pgvb"]["top1"])}|{pct(g["metrics"]["explicit"]["top1"])}|{pct(pair["methods"]["explicit"]["pair_win_fraction"])} ({pair["pairs"]}对)|')
    lines += ['', '两两胜率与同时胜过多个扰动的Top1不同。显式评分对旋转/变形较有帮助，但平移组A同时排名第一仅29.03%；这不足以安全确认所有公开轮廓。原Polygon可能是地面Footprint、粗Contour，屋顶/树木/背景等像素边界未必与它一致；扰动轮廓有时可能更接近图像边缘，不能据此强制制造错误像素标签。','',
        '## 5. Polygon作用及跨帧一致性','',
        f'- 共{s["shape_token_sensitivity"]["candidate_evaluations"]}次候选评分，Polygon Token置零使平均Patch分数变化MAE={s["shape_token_sensitivity"]["mean_patch_score_MAE"]:.4f}。34查询PGVB Top1从52.94%降至38.24%；共同25查询从52%降至32%。**原PGVB并非不使用Polygon。**',
        '- 但现结构主要通过位置/轮廓Query改变全局视觉上下文，不等于建立了边界点到像素的显式一一对应；去掉Token也同时丢失图像位置。',
        f'- {s["causal_references"]["n"]}个可用过去帧参考、{s["causal_references"]["tasks"]}任务组，同公开ID平均ROI描述子余弦{s["causal_references"]["mean_same_ID_cosine"]:.4f}；当前A与参考共享的可观测区域比例中位数{s["causal_references"]["median_common_observed_fraction"]:.4f}。',
        f'- 参考对平移中位数{s["causal_references"]["median_pose_translation_m"]:.3f}m、yaw差中位数{s["causal_references"]["median_yaw_change_degrees"]:.2f}°，相同RGB SHA对数{s["causal_references"]["identical_RGB_sha_pairs"]}。位移小但yaw变化≥5°也可入组，不把重复像素域裁剪当三维新视角。',
        '- 逐候选CPU复核1174次ROI SHA、Token及有效正候选UV全部一致，57个真实负候选满足公平约束。58个扰动改变像素Mask但未改变稀疏正中心：原生24×24特征后PGVB取12×12稀疏中心，存在区域量化，不能当作缓存未更新。核验见`full_swap_audit.json`。',
        '- 过去帧参考区域本身依赖已知公开对象ID及Polygon，增益不能解释为仅凭语言独立找到了地标。',
        '- 当前是同一正射栅格的不同裁剪/旋转，相同区域常复用同一底图纹理。因此外观参考收益可以支持“缓存真实观测中的对象外观进行关联”的机制，但不证明真实三维跨视角鲁棒性，也不构成像素级实例金标准。','',
        '## 6. 类别、尺度、高度、入镜比例','',
        '|分组（同组所有6方法共用查询）|查询/任务|Geometry|PGVB|Explicit|Appearance|融合参考|','|---|---:|---:|---:|---:|---:|---:|']
    labels=dict(Building='Building',Road='Road',height_0_20='AGL<20m',height_20_40='20≤AGL<40m',height_40_plus='AGL≥40m',partial_visible='入镜比例<0.75',mostly_visible='入镜比例≥0.75',small_ROI='投影面积<1024px²',medium_ROI='1024–8192px²',large_ROI='投影面积≥8192px²')
    for key,g in s['strata'].items():
        lines.append('|'+labels[key]+f'|{g["n"]}/{g["tasks"]}|'+ '|'.join(pct(g['metrics'][m]['top1']) if g['n'] else 'NA' for m in ('geometry','pgvb','explicit','appearance','explicit_reference'))+'|')
    lines += ['', '小规模、非随机化分组，不能反向选择高度或匹配权重。小ROI共同困难组为0，Parking/Car也未纳入原PGVB支持类别，不能把Building/Road的84%外推为车辆/停车场准确率。Building融合比纯参考略差、Road略好，但Road共同组仅6查询，不据此制定分类权重。','',
        '## 7. 自动视觉监督能否成立','',
        '1. **已知Polygon与Pose，能否可靠确认RGB实例？** 能投影并为部分实例获得轮廓/外观证据；完整对照显示单帧边缘不足以普遍确认指定实例，过去观测参考能改善本批地图ID关联代理，但仍有失败和地图轮廓不确定性。',
        '2. **PGVB利用轮廓了吗，显式方法有多少增益？** Polygon输入确有作用；显式评分相对PGVB在旋转组提升约16.47个百分点、变形组约11.96个百分点，但真实困难实例组没有增益，加入参考后也未优于纯参考总体84%。因此暂不将显式分数作为可靠视觉接管条件。',
        '3. **能自动构造可靠实例监督吗？** 可以生成带来源和不确定性的公开ID弱关联、困难负例和扰动样本，不能宣称已经得到可靠像素Mask或高精度实例GT。当前导出的`weak_associations.jsonl`全部`training_allowed=false`：本批源自Val-Seen诊断。后续需在隔离TRAIN重新生成，保留原始Polygon、可见性、边界不确定性、跨帧对应及多方法分歧，不将未知/扰动强行标成真实实例负标签。','',
        '**建议优先项：** 在独立TRAIN上复核“过去观测对象外观参考+公开ID候选”的实例关联，而非继续增强单帧边缘分数。边缘/Polygon分歧可作不确定性提示，先不要用来硬过滤候选或生成像素硬标签。不修改现有Stable Goal、Trajectory、Stop，也不启动闭环。','',
        '## 8. 工程与交付','',
        f'- RTX PRO6000 Blackwell，冻结BF16真实GPU循环{run["seconds"]:.2f}s（不含进程/权重加载和绘图），实际SigLIP2视觉前向{run["native_rgb_forwards"]}次，缓存命中{run["cache_hits"]}次，峰值allocated {run["peak_gpu_allocated_gib"]:.3f}GiB；没有每个Polygon重复编码整张RGB。',
        '- Checkpoint SHA256 `379cfc419ef098dc40ad85daa29419340058f46ae8a359415054625594d6c812` 前后不变；代码、参数、输入SHA和排除记录见`manifest.json`/`results.jsonl`。所有权重冻结，真实GPU无训练或优化器更新。',
        '- 53项相关单元测试通过：完整Polygon替换同步改变Token/Mask/候选标签、Union内部空洞保留、出画负例拒绝、无RGB几何分数不变、合成边缘匹配及平分公平性。',
        '- 隔离Firefox实际测试93条可用查询切换、候选切换、Token显示和真实RGB加载；临时服务已关闭。HTML下载所在目录后可打开，也可在该目录用本地HTTP服务浏览。',
        '- [交互HTML](index.html) · [共同子集排名图](ranking_summary.png) · [逐候选分数CSV](scores.csv) · [完整结果](results.jsonl) · [统计JSON](summary.json)。图册含正确A、真实B及全部扰动、RGB边缘、分数分解、Token、过去帧对应。10个静态案例覆盖两类，未按成绩只选成功。',
        f'- 原始描述子NPZ及GPU日志保留服务器 `{a.run_root}/results`；没有覆盖旧实验。`reproduce_gpu.sh`记录真实命令及断点恢复，`reproduce_analysis.sh`记录统计与可视化命令。','']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))
    associations=[]
    for row in rows:
        if not row['candidates']:continue
        ids=row['candidates'][0]['public_ids']
        for c in row['candidates']:
            associations.append(dict(key=row['key'],query_public_ids=ids,candidate_public_ids=c['public_ids'],candidate_id=c['id'],
                weak_same_ID_label=1 if c['family']=='correct' else 0 if c['family']=='real_other_instance' else None,
                source='public_map_ID_association_not_independent_visual_GT',source_split='val_seen_named_landmark_diagnostic',training_allowed=False,
                pixel_mask_verified=False,synthetic_perturbation=c['family']not in ('correct','real_other_instance'),
                appearance_score=c['appearance'],explicit_score=c['explicit'],reference=row['reference'],requires_uncertainty_handling=True))
    (out/'weak_associations.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in associations))
    shutil.copy2(a.run_root/'run.sh',out/'reproduce_gpu.sh');shutil.copy2(a.run_root/'gpu.log',out/'gpu_stdout.txt')
    (out/'reproduce_analysis.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain\n/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.analyze_polygon_rgb_matching --run '+str(a.run_root/'results')+' --nearfield reports/unified_pgvb/nearfield_scene_audit --rollout /home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/stable_landmark_20261105/full64 --output reports/unified_pgvb/polygon_rgb_matching\n')

if __name__=='__main__':main()
