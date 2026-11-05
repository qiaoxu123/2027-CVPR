"""Generate evidence-qualified Chinese nearfield conclusions from frozen measurements."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args();out=a.output;s=json.loads((out/'summary.json').read_text());runtime=json.loads((out/'runtime.json').read_text());r=s['relation_proxies'];c=s['anchor_controls'];proxy=s['neighborhood_visual_class_proxy']
    frames=list(map(json.loads,(out/'dataset.jsonl').open()));near=[f for f in frames if f['near']]
    lines=['# UnifiedPGVB-v2：近场地标、场景与关系证据审查','',
        '日期：2026-11-05。基于 `97c705581104fcdfd58ac806bc3b442fa1cce6ea`，分支 `experiment/unified-pgvb-landmark-acquisition-20261104`；不合并main。','',
        '**结论：现有模型有真实RGB区域/类别响应，但尚不能证实指定地标的实例身份识别。公开地图能检索近邻及计算几何关系，RGB只能为部分节点提供粗类别支持；证据不足以直接宣称已经具备复杂CityNav间接目标定位能力。**','',
        '## 数据与实验边界','',
        f'- 复用64条StableTrackedPGVB真实闭环、{s["maps"]}张地图，固定选取{ s["frames"]}帧：首次进入公开地标20m、最后近场帧、首次近场Tracking帧（去重）；其中{s["near_frames"]}帧来自{s["near_tasks"]}个到过近场的任务，另保留未到达的Celine Grove终帧。它是冻结轨迹派生分析集，不是新的独立泛化评测集。',
        '- 所有RGB均为原384×384真实已访问观测；恢复Depth时仅按原实际Pose重采样，并严格逐帧验证RGB SHA和Footprint与缓存一致。未读取原CityNav目标GT，没有未来观测或未访问区域RGB，没有重新跑导航。',
        '- 不训练、不新增视觉网络；复用原冻结SigLIP2及已训练PGVB Head。每个采样帧只编码一次真实RGB，所有公共对象ROI共享同一Patch图；零RGB对照共享一次黑图编码。Polygon用于提案和代理评价，不把地图类别作为视觉分类输入。',
        '- 场景类别为原SigLIP2特征对8个固定通用文本Prompt的零样本诊断读出，并非UnifiedPGVB原本具备的场景分割Head。PGVB原生仅支持Building/Road；Other保持原回退机制。',
        '- 真实地标身份、真实追踪准确率、场景真实分类准确率、真实像素IoU、直接/间接关系准确率：**全部NA**。独立人工实例/Mask真值为0，不能写成0%准确率。`summary.json`这些字段为null；人工核验模板全部初始unreviewed。',
        '- 地图坐标关系仅指World XY；不据本轮声称+Y为地理北。RGB来自正射裁剪，Depth是竖直高度差代理，不具备真实三维遮挡/透视可见性真值。','',
        '## 1. PGVB是否真正识别了指定地标','',
        '|采样近场帧证据状态（不是人工身份判定）|帧数|含义|','|---|---:|---|',
        f'|RGB支持代理|{s["status_proxy_frames"]["RGB_supported_proxy"]}|已Tracking且独立通用类别读出与地图类别一致；只支持类别/区域|',
        f'|RGB冲突代理|{s["status_proxy_frames"]["conflicting_RGB_proxy"]}|已Tracking但粗类别与地图不一致；可能是混合ROI、地图误差或视觉错误，不能直接宣布错误实例|',
        f'|仅几何证据|{s["status_proxy_frames"]["geometry_only"]}|有有效投影，但本帧未达到上述视觉支持条件|',
        f'|视觉证据不足|{s["status_proxy_frames"]["insufficient_visual_evidence"]}|可用像素/ROI支持不足；不算检测失败|','',
        f'采样中{s["tracked_near_tasks"]}个任务、{s["tracking_frames_in_sample"]}帧处于Tracking；其中48/70=68.57%同时满足粗类别一致。这不是追踪准确率。PGVB重算与日志分数最大MAE={s["logged_score_mae_max"]}，排除此次读出接口与原轨迹评分不一致。','',
        '|同一批可比较帧的PGVB正负区域AUC代理|平均AUC|样本帧|','|---|---:|---:|']
    for name,key in [('真实RGB/正确名称','real_auc_proxy'),('同图错位Patch特征','shuffled_auc_proxy'),('视觉特征置零','zero_features_auc_proxy'),('同类型错误地标名称','wrong_name_auc_proxy')]:
        lines.append(f'|{name}|{c[key]["mean"]:.4f}|{c[key]["n"]}|')
    lines += ['', f'正负区域来自公开Polygon弱标签，不是人工分割。错误名称AUC有{s["same_type_wrong_name_auc_within_005"]}/93帧与正确名称差≤0.05，{s["same_type_wrong_name_auc_not_worse"]}/93帧不差于正确名。真实RGB有贡献，但专名身份区分证据很弱。PGVB原生打分包含`name + 3×type`的文本项，类别信息占较大权重；本轮未改公式。',
        f'有分数帧的最高分Patch落在Polygon可靠正区域{ s["top_scored_patch_label_proxy"].get("1",0)}帧、可靠负区域{s["top_scored_patch_label_proxy"].get("0",0)}帧、边界未知区{s["top_scored_patch_label_proxy"].get("-1",0)}帧；这是投影重合代理，不是实例匹配准确率。',
        '结论不是“必须从屋顶读出地标名字”。真正缺少的是：在给定地图Anchor时，区分同图多个同类对象、确认对应视觉实例并跨帧保持关联的独立证据。','',
        '## 2. 地标周围100m的场景','',
        f'按完整Polygon边界距离检索全部合法公共对象，不沿用旧日志最多12邻居的显示截断。近场共{s["neighbor_object_views"]}个对象—视图对，涉及{s["unique_neighbor_objects"]}个不同(map, public_id)组，重复帧不当成独立物理实例。',
        f'- 完全在当前投影视野外：{s["out_of_view_views"]}个（{s["out_of_view_views"]/s["neighbor_object_views"]*100:.2f}%），未知，不算视觉失败。',
        f'- 有投影但有效ROI/Patch不足：{s["insufficient_patch_views"]}个；剩余{proxy["n"]}个可以进行弱ROI视觉类别对比。门槛固定为有效像素≥64且≥2个有效原生Patch，未按结果调参。',
        f'- 已投影ROI的平均RGB/Depth源支持比例{s["geometric_mask_support"]["mean_supported_fraction_on_projected_ROIs"]*100:.2f}%；只是几何与源有效性，不是Polygon—真实物体Mask的IoU。',
        f'- 可比较邻居ROI：Top1地图标签代理一致率{proxy["top1_map_proxy"]*100:.2f}%，Top3 {proxy["top3_map_proxy"]*100:.2f}%；错位特征Top1 {proxy["shuffled_map_proxy"]*100:.2f}%，零RGB {proxy["zero_rgb_map_proxy"]*100:.2f}%。真实RGB有类别信号，但没有证明精确实例识别。','',
        '|对象类别|可比较对象—视图|Top1地图代理一致率|Top3地图代理一致率|错位特征Top1|','|---|---:|---:|---:|---:|']
    for name,v in s['by_type'].items():
        lines.append(f'|{name}|{v["n"]}|{v["top1_map_proxy"]*100:.2f}%|{v["top3_map_proxy"]*100:.2f}%|{v["shuffled_map_proxy"]*100:.2f}%|')
    lines += ['', 'Other合并了墙体、街具等异质对象；Ground不等于绿地。Car、Parking等区域可能混入背景或受到粗Contour/上下文影响，不能把地图代理不一致全归为模型错误。共享PatchTransformer在整帧上有上下文，ROI池化不等于独立裁剪识别，更不等于像素分割。','',
        '### 距离与高度分层（均为观察性代理统计）','',
        '|分组|对象—视图|Top1地图代理一致率|','|---|---:|---:|']
    for prefix,key in [('对象至地标Polygon距离(m)','by_distance'),('UAV至对象Polygon距离(m)','by_uav_distance'),('观测高度AGL(m)','by_height')]:
        for name,v in s[key].items():
            label=name.replace('50-100.00001','50-100').replace('40-inf','≥40').replace('50-inf','≥50')
            lines.append(f'|{prefix} {label}|{v["n"]}|{v["top1_map_proxy"]*100:.2f}%|')
    lines += ['', '采样高度实际约10–114.58m；初始较高航段不能伪写成全部≤100m。本轮不改原控制器。上述高度、距离组并非固定同一对象的随机化对照，类别、视野和目标尺寸均混杂，不能据此选择飞行高度或宣称高空优于低空。','',
        '## 3. 直接及间接空间关系','',
        '|问题|地图计算结果|RGB证据范围|真正关系准确率|','|---|---|---|',
        f'|建筑旁是哪条路|{r["map_building_nearest_road_queries"]}个建筑—最近道路查询，保留等距候选和边界距离|只说明公开几何最近，不把最近强称为语言beside|NA|',
        '|停车场相对地标|逐对象提供World XY质心差、边界距离、相交/包含/接触|RGB类别可作部分节点支持；方向仍来自地图|NA|',
        f'|地标→停车场→车辆|地图侧{r["map_car_parking_chains"]}条链（按任务地图记录，非独立实例数）；近场{r["car_parking_chain_views"]}个链—视图对|{r["both_RGB_rois_available"]}个两端有ROI；仅{r["both_node_types_match_map"]}个两端粗类别同时符合地图|NA|',
        f'|区域被多条道路围合|道路并集有{r["exact_road_union_holes"]}个精确空洞；其中{r["qualified_named_multi_road_holes"]}个满足≥3条不同具名道路、面积≥25m²|对应{r["qualified_multi_road_hole_views"]}个视图中，完整道路ROI支持{r["all_road_ROIs_available"]}个|NA|','',
        '同名道路多Polygon片段不能算多条道路；已在离线分析按规范化道路名称合并并补充单元测试。没有空洞不代表没有围合：道路缺段或注释尺度可使拓扑无法闭合。不做buffer补缝制造封闭区域。',
        f'两端类别均符合地图的80/{r["car_parking_chain_views"]}个链—视图对，只是节点类别+公开包含关系的代理支持，不是学会了语言中的“停车场里面的那辆车”。真正的目标属性、排列和关系作用域模型本轮并未引入。','',
        '## 4. 真实图像审查与瓶颈','',
        '- 已由当前助手直接查看16案例拼图，并放大任务13、21、62；结论单独保存在`assistant_qualitative_review.json`。它是AI定性审查，不是独立人工真值，未写入人工核验标签。',
        '- 任务13：仓库屋顶、停车通道与车辆清楚，支持粗类别与局部场景，但不足以确认每辆车的身份。',
        '- 任务21：街道和停车排可辨，同名道路多个轮廓重叠；几何片段数量不能当成多道路关系。',
        '- 任务62：屋顶清楚，但公开建筑Contour部分包含树木/院落。说明数值投影自洽不等于像素级分割准确。',
        '- 任务33：低空使局部外观变大，同时建筑裁剪严重；上下文缺失仍然限制身份及关系判断。','',
        '主要瓶颈按当前证据：','',
        '1. **可观测范围与实例级视觉证据不足。** 大多数100m邻居不在当前有效RGB中；现有PGVB能区分局部建筑/道路，却不能证实具体地图实例，更没有对所有中间场景的可靠识别。',
        '2. **缺乏干净对象Mask和关系节点绑定。** 现有对象ROI由粗Polygon提出，可能混合屋顶、地面、树木及邻近车辆；原模型没有场景分割或目标实例关系读出。几何距离/包含可计算，但不能替代视觉实体关联和指令关系作用域理解。',
        '没有证据指向全局World→RGB数值变换是主要错误；然而粗地图轮廓与真实视觉边界偏差仍是监督/ROI质量问题。不能给每个“代理冲突”强行归唯一真实失败原因。','',
        '## 三个最终问题与下一步','',
        '1. **PGVB真正识别并追踪指定地标了吗？** 证明了真实RGB参与、已知ROI的区域和类别支持；尚未证明专属实例身份和真实追踪准确率。错误同类名称对照依然接近正确名，不能过度宣称。',
        '2. **能找到停车场、道路及中间场景吗？** 地图侧可以完整检索、计算几何关系；部分可观测ROI有RGB支持，但覆盖和语义纯度有限，尤其小车及多道路完整场景。',
        '3. **足以支持复杂CityNav目标定位吗？** 目前不足以直接进入复杂关系联合训练。需要先获得少量可靠ROI/实例核验、量化地图边界不确定性、建立同图同类对象的身份对应和可观测性管理，再用已验证节点构建局部关系图。应先验证这些能力，而不是继续调整Stop或叠加大型关系网络。','',
        '## 复现与交付','',
        f'- RTX PRO6000 Blackwell，冻结GPU读出{runtime["frames"]}帧，分析循环{runtime["seconds"]:.2f}s（不含权重加载和图册生成），峰值allocated {runtime["peak_gpu_allocated_gib"]:.3f}GiB。无新训练、无新闭环。权重SHA前后不变，见`manifest.json`与`runtime.json`。',
        '- Unified Checkpoint SHA256 `379cfc419ef098dc40ad85daa29419340058f46ae8a359415054625594d6c812`；Stable Goal、PGVB运行时、Stop和导航协议代码未修改。',
        '- 48项相关测试通过；隔离Firefox实际测试141帧切换、Polygon开关、核验备注保存/刷新恢复，测试标注已清空，临时服务已关闭。截图和测试记录见`browser_smoke.png`、`browser_test.json`。',
        '- [交互HTML](index.html)：下载本目录后可本地打开；或在该目录运行`python -m http.server 8767 --bind 127.0.0.1`。同时查看原始RGB、Polygon投影、PGVB评分、公开地图及对象和关系表。人工复核只存浏览器本地，导出后需独立确认，不自动当GT。',
        '- [16案例总览](contact_sheet.png) · [类别代理混淆图](scene_proxy_confusion.png) · [逐对象视图CSV](object_view_predictions.csv) · [数据集JSONL](dataset.jsonl) · [关系视图](relation_views.json)。全部141帧原始RGB、逐帧元数据及64份公共几何关系文件随报告提交。',
        '- GPU提取时使用的几何辅助源码快照在`source_at_extraction/nearfield_geometry.py`，SHA与提取manifest相符；之后新增的道路名称分组仅用于离线分析，不改变提取或策略。',
        f'- 原始NPZ/Depth及中间缓存保存在服务器 `{a.run_root}/dataset_v2`。`reproduce_gpu.sh`使用新目录复现，保留现有结果；`reproduce_analysis.sh`记录离线分析命令。阶段日志及初次工具device参数错误修复记录保留在服务器，没有影响原模型或旧实验。','']
    (out/'REPORT_zh.md').write_text('\n'.join(lines))
    command=(a.run_root/'run.sh').read_text().replace('/dataset_v2 ', '/dataset_reproduction ')
    (out/'reproduce_gpu.sh').write_text(command)
    (out/'reproduce_analysis.sh').write_text('#!/bin/bash\nset -euo pipefail\ncd /home/20261024_1/Workspace/2027-CVPR/Experiments/GeomBeliefLandmarkChain\n/home/20261024_1/Workspace/third_party/aerialvg_env/bin/python -m scripts.analyze_nearfield_scene --dataset '+str(a.run_root/'dataset_v2')+' --output reports/unified_pgvb/nearfield_scene_audit\n')
    shutil.copy2(a.run_root/'gpu.log',out/'gpu_stdout.txt')

if __name__=='__main__':main()
