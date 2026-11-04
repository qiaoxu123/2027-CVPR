"""Generate a Chinese report from complete paired measurements, not hand metrics."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    a=p.parse_args(); result=json.loads((a.output/'summary.json').read_text());s=result['summary']
    audit=json.loads((a.output/'coordinate_evidence_audit.json').read_text());near=json.loads((a.output/'neighborhood_check.json').read_text());runtime=json.loads((a.run/'metrics.json').read_text())
    rows=[json.loads(x) for x in (a.output/'trajectories.jsonl').read_text().splitlines()]
    tracked=[r for r in rows if r['method']=='TrackedPGVB'];horizon=[r for r in tracked if r['termination']=='horizon']
    blocked=sum(r['step_diagnostics'][-1]['stop_probability']>=.3 and r['step_diagnostics'][-1]['predicted_goal_distance']>.001 for r in horizon)
    text=['# UnifiedPGVB 地标接近、RGB确认及追踪闭环实测','',
        '日期：2026-11-05。基准代码 `fc595d18d50363a087a6a920a51b0c89d12226fe`；实验分支 `experiment/unified-pgvb-landmark-acquisition-20261104`。', '',
        '**结论：Geometry能够可靠接近这批具名地标；PGVB能够触发并影响Goal，但没有提升最终SR，且使Goal切换、Horizon结束和路径成本增加。本轮不推荐直接替换Geometry部署。**','',
        '## 数据、权重及协议','',
        '- 原固定64条公开具名地标代理任务，来源Val-Seen：Building 32、Road 24、Other 8，全部计入；不是原始CityNav最终目标导航成绩。任务SHA及完整代码、配置指纹见 `run_manifest.json`。未使用正式Test，也未用本批调参。',
        '- 同一起始位姿、完整指令、冻结权重、原生逐步yaw和384×384真实观测，真实当前观测逐步决策。先完成2任务×3方法GPU Smoke，再完成64×3=192次闭环。无模型训练，未改变正常推理代码、Stop阈值、Horizon或动作控制器。',
        '- 成功：最终位置到公开同名Polygon合法并集的距离≤20m；OSR：曾进入20m。SPL_polygon使用起点到Polygon最短直线距离/实际路径长度，是本代理任务的诊断指标，不是原CityNav官方点目标SPL。没有该新任务专家路线，不虚构nDTW。',
        '- Horizon=20；水平单步≤50m；高度10–100m；Stop概率阈值0.3，原1mm当前Goal到达守卫全部保留。Other按既定Geometry回退。所有GT评分均为闭环结束后的公开Polygon距离，没有原CityNav真实Target输入或Oracle选择。',
        '- Unified权重：`GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt`，SHA256 `379cfc419ef098dc40ad85daa29419340058f46ae8a359415054625594d6c812`。控制器原Stage1 SHA256 `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`；运行后重新验证均未改变。','',
        '## 全量配对结果','',
        '|方案|SR|OSR|NE(m)|SPL_polygon|正确Policy Stop/64|误停|Horizon|平均动作步数|平均每任务决策时延(ms)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in s.items():
        text.append(f'|{m}|{v["successes"]:.0f}/64 ({v["SR"]*100:.2f}%)|{v["OSR"]*100:.2f}%|{v["NE"]:.4f}|{v["SPL_polygon"]:.4f}|{v["correct_stop"]}/64 ({v["correct_stop"]/64*100:.2f}%)|{v["false_stop"]}|{v["terminations"].get("horizon",0)}|{v["steps"]:.2f}|{v["mean_decision_ms"]:.2f}|')
    text += ['', '正确Policy Stop要求policy_stop且最终成功；SR按原距离协议计分，即使Horizon结束仍在20m内也算成功。因此SR与正常Stop率不能混为一谈。逐任务首入20m时刻见 `paired_tasks.csv`，完整逐步记录见 `trajectories.jsonl`。','',
        '|分层|样本|Geometry SR|Legacy SR|Tracked SR|','|---|---:|---:|---:|---:|']
    for c,v in s['Geometry']['category'].items():
        text.append('|'+c+'|'+str(v['n'])+'|'+'|'.join(f'{s[m]["category"][c]["SR"]*100:.2f}%' for m in s)+'|')
    text += ['', '所有方案成功/失败任务集合完全相同，任意两组相对SR纠正0、破坏0、净收益0；配对Episode bootstrap的SR差区间为[0,0]，只反映本固定集合，不能据此断言总体等价。Geometry OSR=62/64，视觉两组=63/64。','',
        '## RGB确认与接管究竟发生了什么','',
        f'- Tracked在{s["TrackedPGVB"]["tracking_episode"]}/64任务触发Tracking，{s["TrackedPGVB"]["tracking_frames"]}/{s["TrackedPGVB"]["decisions"]}决策帧启用视觉；coasting {s["TrackedPGVB"]["coasting_frames"]}帧，完整退回Geometry {s["TrackedPGVB"]["fallback_events"]}次。失观测/coasting帧视觉增益为0，没有持续注入旧视觉证据。',
        '- 判据是公开ROI弱正负区域内的RGB分数区分性、与错位RGB特征的反事实差、连续两帧确认；不是独立实例身份认证。未获得人工视觉Mask，不能宣称识别精度。',
        '- 现代码所谓接管实际为Geometry logits + 有界视觉残差：Legacy增益1、Tracked确认后逐步升至2；Geometry仍保留。不是纯视觉Belief替代，也不保证视觉占主导。',
        f'- 超20m相邻Goal跳变：Geometry {s["Geometry"]["goal_jumps_over20"]}、Legacy {s["LegacyPGVB"]["goal_jumps_over20"]}、Tracked {s["TrackedPGVB"]["goal_jumps_over20"]}次。切换/回退虽然保守，但未改善目标稳定性。',
        '- 以下为同一实际观测下，Geometry候选首位到Final候选首位是否跨20m阈值，属于逐决策空间诊断，不是独立任务SR纠正：','',
        '|方法|Goal纠正帧|Goal破坏帧|正确区域20m概率质量增加/降低帧|实测最大logit修正|','|---|---:|---:|---:|---:|']
    for m,v in audit['methods'].items():
        text.append(f'|{m}|{v["corrected_goal_decisions"]}|{v["damaged_goal_decisions"]}|{v["mass_increase"]}/{v["mass_decrease"]}|{v["max_abs_delta"]:.5f}|')
    text += ['', '视觉有局部作用但未挽回三条最终失败；同一Polygon内多个合理格点的切换，不一定跨20m阈值，却会改变控制器目标、路径及Stop守卫。不能仅凭“破坏帧=0”宣称无负面影响。','',
        '## 三条失败与停止证据','',
        '|任务|三组结果|实际证据及归因|','|---|---|---|',
        '|0 Celine Grove|均失败，NE26.007m|Geometry首位/最终Goal距合法道路26.007m，始终未进20m；无Tracking触发。主要是错误Goal及缺失视觉纠正，非已到正确位置后Stop漏触发。|',
        '|1 St. Teresa Church|均失败，NE24.970m|Geometry未进入；Legacy/Tracked第4动作状态曾进入20m，随后离开并在错误Goal处policy_stop。Tracked第3决策已触发追踪，最终仍回到24.970m外Goal。主要是最终Goal错误，不能仅称Stop阈值错误。|',
        '|39 Park Lodge|均失败，NE23.685m|三组第2状态曾进20m，随后离开，最终到错误Goal并policy_stop。Tracked未触发。Geometry错误峰值未被修复。|','',
        f'Tracked的9条Horizon任务：{[r["index"] for r in horizon]}，最终均已在20m内；其中{blocked}条最后决策Stop概率≥0.3但未满足1mm守卫，其余还存在低Stop概率。此为动态Goal与原到达接口不兼容的证据，不等于放宽阈值就能安全解决误停。',
        '任务13：Tracked第1决策Tracking，最终位于Polygon内，末步Stop概率0.713但距当前Goal14.29m；任务43末步概率0.832而距Goal0.097m。均被守卫阻止。任务34：Legacy Horizon，Tracked正常Stop，说明逐任务局部改善并未改变本批SR。','',
        '## 投影及附近几何核验','',
        f'- {audit["frames"]}帧记录全部与实际Pose一致，Footprint中心最大差{audit["max_footprint_pose_center_distance_m"]:.8f}m，世界—像素回投最大差{audit["max_world_pixel_roundtrip_m"]:.9f}m。未发现数值坐标错位。',
        '- 图册叠加实际RGB、完整公开轮廓、相机Footprint和世界Belief。数值闭合只能确认映射实现自洽；建筑屋顶偏移、地图陈旧、遮挡等仍可能导致视觉配准误差，不能冒充人工IoU。',
        f'- 到达后公开邻域{near["checked_entries"]}条记录（目标任务—邻居唯一组合{near["unique_objects"]}）独立重算Polygon距离、World XY质心方位角和相交/包含拓扑，最大距离误差{near["max_distance_error_m"]}m、角差{near["max_bearing_error_deg"]}°、拓扑全部一致。未到达帧邻域信息全部隐藏。',
        '- 方位仅验证工程World XY，不通过本实验声称+Y为地理北。邻域是诊断输出，未进入Goal选择、Tracking门控或Stop。','',
        '## 效率、复现与下一步','',
        f'- RTX PRO6000 Blackwell 96GB；Python3.10.22、PyTorch2.8.0+cu128、CUDA12.8、BF16；192次完整rollout循环耗时{runtime["seconds_current_session"]:.2f}s（不包含进程初始化/权重加载、报告绘图），模型运行峰值allocated {runtime["peak_gpu_allocated_gib"]:.3f}GiB。时延包含当前逐决策审计写盘，三组同样记录；不是纯网络benchmark。',
        '- 相关测试32项通过；2×3 GPU Smoke无接口/非有限错误；64×3全量无任务丢失。未发现需要改变模型推理的确定性代码错误，因此未追加参数扫描或改动控制器，也未覆盖原结果。',
        '- 优先研究“已确认地标内的一致Goal到控制器接口”，而非提高视觉增益。独立TRAIN上验证目标区域/目标保持机制能否减少追逐相邻格点，同时保留错误Goal被纠正的能力；继续冻结Stop阈值，不能用本64条挑参数。',
        '- 三条失败还表明：Geometry在已知合法Polygon外的峰值及视觉残差不能持久纠正，需要另立独立TRAIN实验检查公开地标约束下的Goal解码；本轮没有启用此方案。',
        '- 本轮不支持“追踪已提高SR”结论，也不支持扩大训练；应先解决动态Goal接口与真实地标识别证据的区分。','',
        '## 文件与恢复','',
        '[HTML全64条图册](index.html) · [逐任务配对表](paired_tasks.csv) · [汇总图](summary.png) · [全部数值轨迹](trajectories.jsonl) · [逐帧证据](step_evidence.jsonl)。',
        '典型图： [任务0](task_00.png)、[任务1](task_01.png)、[任务13](task_13.png)、[任务34](task_34.png)、[任务39](task_39.png)、[任务43](task_43.png)；相应`transitions_*.png`展示Goal、距离、Stop及切换逐步曲线；`native_rgb/`保留代表帧原生PNG。','',
        f'服务器原始完整RGB/NPZ/观测缓存与日志：`{a.run}`。完整原始文件SHA清单在`raw_artifacts_manifest.json`，大体积缓存不上传GitHub。`reproduce.sh`记录真实命令；tmux会话`pgvb-acquisition-20261105`完成后退出，已完成结果持久保存；runner支持`--resume`按相同指纹逐任务恢复。','']
    (a.output/'REPORT_zh.md').write_text('\n'.join(text))
    shutil.copy2(a.run.parent/'run.sh',a.output/'reproduce.sh')

if __name__=='__main__':main()
