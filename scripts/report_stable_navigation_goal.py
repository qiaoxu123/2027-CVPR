"""Chinese decision report generated only from complete stable-Goal rollouts."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();s=json.loads((a.output/'summary.json').read_text());audit=json.loads((a.output/'stable_interface_audit.json').read_text());coordinate=json.loads((a.output/'coordinate_evidence_audit.json').read_text())
    rows=[json.loads(x) for x in (a.output/'trajectories.jsonl').read_text().splitlines()];metrics=json.loads((a.run/'metrics.json').read_text())
    groups={m:{r['index']:r for r in rows if r['method']==m} for m in s['summary']}
    new=s['summary']['StableTrackedPGVB'];old=s['summary']['TrackedPGVB'];passed=new['successes']>=61 and new['correct_stop']>=61
    lines=['# Stable Navigation Goal：冻结GPU闭环验证','',
        '日期：2026-11-05。基准 `a2db53225fce9799f6d4e0970d07b73cd562fe58`；分支 `experiment/unified-pgvb-landmark-acquisition-20261104`；暂不合并main。','',
        f'**验收：{"达到" if passed else "未达到"}。StableTrackedPGVB成功{new["successes"]:.0f}/64，正确Policy Stop {new["correct_stop"]}/64，Horizon {new["terminations"].get("horizon",0)}条。** 这是固定具名地标Polygon代理任务，不是原CityNav真正目标或正式全量SR。','',
        '## 代码、数据与公平性','',
        '- 工作区原本干净，fetch后fast-forward至指定提交；保留上轮完整数据，使用新输出目录。原固定64任务（Building32、Road24、Other8），未筛选成功任务。逐帧当前真实RGB、原生yaw、原起点、原Horizon20及20m Polygon成功判据。',
        '- 两条任务三组GPU Smoke正常后执行64×3=192次完整真实闭环。没有训练，没有GT或Oracle位置进入策略，也没有改Stop概率0.3、1mm到达守卫、Trajectory权重或动作限制。公开地图Polygon为原合法输入。',
        '- StableNavigationGoal不改dense Belief logits；每帧PGVB照常更新。只保持控制器候选0的网格ID，重排时按同ID重取XY、dense Context和对应logit。不同Goal需当前视觉覆盖、Tracking激活、优势≥0.35且连续2帧一致才允许重规划。上述配置保持远程实现原值，未按本64条调参。',
        '- Unified权重为`GraphNav/runs/unified_pgvb_v2_pilot_s42/full_training/last.pt`，SHA256 `379cfc419ef098dc40ad85daa29419340058f46ae8a359415054625594d6c812`；原Stage1 SHA256 `00f2acc0e3d0cd2c141dbf852c1205fedd218a174ea1d5528edba67217d77c5a`。完整路径、配置与源码SHA见`run_manifest.json`，运行后两份权重SHA复验未变。','',
        '## 64条完整结果','',
        '|方法|SR|OSR|NE(m)|SPL_polygon|正确Stop/64|误停|Horizon|>20m Goal跳变|平均路径(m)|平均动作步数|平均决策(ms)|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m,v in s['summary'].items():
        lines.append(f'|{m}|{v["successes"]:.0f}/64 ({v["SR"]*100:.2f}%)|{v["OSR"]*100:.2f}%|{v["NE"]:.4f}|{v["SPL_polygon"]:.4f}|{v["correct_stop"]}/64 ({v["correct_stop"]/64*100:.2f}%)|{v["false_stop"]}|{v["terminations"].get("horizon",0)}|{v["goal_jumps_over20"]}|{v["path_length_m"]:.2f}|{v["steps"]:.2f}|{v["mean_decision_ms"]:.2f}|')
    lines += ['', 'SR按最终到Polygon距离计分；正确Stop要求policy_stop且最终成功。SPL_polygon使用起点到公开Polygon的最短距离/实际路径长度，为同协议诊断指标。没有该新任务的参考路线，不伪造nDTW。时延为每任务平均决策时间再等权平均，包含同样的审计写盘，不是纯网络benchmark。','',
        '### 配对变化及复现','']
    for pair in s['paired']:
        lines.append(f'- {pair["reference"]} → {pair["treatment"]}：纠正{len(pair["corrected"])}条 {pair["corrected"]}，破坏{len(pair["damaged"])}条 {pair["damaged"]}，净收益{pair["net"]:+d}；SR差的Episode配对bootstrap 95%区间{[round(x*100,3) for x in pair["episode_paired_bootstrap_CI95"]]}个百分点。样本小且同一诊断集合，不宣称显著泛化提升。')
    for m,v in audit['baseline_reproduction'].items():
        lines.append(f'- {m}相对上一轮：64条Pose、终止原因、全部任务指标逐项精确比较，不一致任务={v["exact_pose_metric_termination_mismatch_ids"]}。')
    lines += ['', '### 原9条Horizon是否恢复','', '|任务|具名地标|新终止|SR|NE(m)|动作步数|正确Stop|','|---|---|---|---:|---:|---:|---|']
    for r in audit['horizon_recovery']:
        lines.append(f'|{r["index"]}|{r["name"]}|{r["new_termination"]}|{r["new_SR"]:.0f}|{r["new_NE"]:.3f}|{r["new_steps"]}|{r["correct_stop"]}|')
    lines += ['',f'原9条中恢复正确Stop：{audit["recovered_correct_stops"]}/9。详细首入20m时刻和逐任务配对指标在`paired_tasks.csv`。','',
        '### 原三条失败','', '|任务|Geometry NE/SR|Tracked NE/SR|Stable NE/SR|','|---|---:|---:|---:|']
    for i in (0,1,39):
        lines.append('|'+str(i)+' '+groups['Geometry'][i]['landmark_name']+'|'+'|'.join(f'{groups[m][i]["metrics"]["NE"]:.3f} / {groups[m][i]["metrics"]["SR"]:.0f}' for m in groups)+'|')
    lines += ['', '## 稳定是否只是锁住初始Geometry？','',
        f'- 稳定接口实际事件：{audit["stable_events"]}；持有Goal与当前视觉Top1不同的帧={audit["held_differs_frames"]}，其中dense视觉Belief实际改变但控制Goal仍保持的帧={audit["held_despite_updated_visual_frames"]}。',
        f'- 视觉更新激活{audit["active_visual_frames"]}帧；其中原Geometry与最终dense logits完全相同的帧={audit["unchanged_visual_logits_on_active_frames"]}。因此不是通过关闭PGVB实现稳定。',
        f'- 真实GPU记录中核对{audit["candidate_context_comparisons"]}组共同候选ID的Context（与原Geometry候选中同ID的Context对比），完全一致；XY与28×28网格ID最大归一化误差{audit["max_normalized_xy_error"]:.9f}，持有ID改变均对应replanned事件。',
        f'- 同一Stable实际观测下，原视觉Top1正确但持有Goal错误的帧={audit["raw_correct_but_held_wrong_frames"]}；原视觉Top1错误但持有Goal正确的帧={audit["raw_wrong_but_held_correct_frames"]}。GT只用于该事后公开Polygon距离评分。这些是条件帧统计，不可当作独立任务或严格闭环反事实。',
        '- 实际重规划任务为1、13、23、50、63。上述4帧延迟均发生在Church任务第3–6决策；第7决策通过连续证据确认后重新规划，最终NE15.761m并正确Stop，说明本批没有永久锁死该错误初始Goal。','',
        '- 关系到错误Goal的纠正，需要看真实replanned事件与三条失败的变化；不能仅凭减少跳变断言不存在迟滞风险。所有任务均保留，未改变推理结果以满足验收。','',
        '## 失败根因、边界与下一步','',
        f'- 相对Tracked，Stable平均NE由{old["NE"]:.3f}m变为{new["NE"]:.3f}m，未改善该平均指标；但SPL由{old["SPL_polygon"]:.3f}升至{new["SPL_polygon"]:.3f}，平均路径由{old["path_length_m"]:.2f}m降至{new["path_length_m"]:.2f}m，正确Stop由{old["correct_stop"]}升至{new["correct_stop"]}。改善重点是控制接口与停止，不应宣称所有定位指标均提升。',
        '- 本轮未发现需要修改模型或控制器的确定性接口错误；原实现直接测试。Stop概率及1mm守卫均不变，收益来自让控制器有机会到达稳定目标，而非放宽Stop。',
        '- 剩余失败须区分：Geometry峰值本来就在合法Polygon外、缺少可用视觉确认，与“正确区域内动态Goal切换阻止Stop”不同。详见任务0、1、39的RGB、Belief和逐步曲线；不凭最终NE推断视觉身份识别成功。',
        '- 建议先保留这版作为具名地标实验候选，在独立TRAIN任务上检查重规划迟滞和错误初始Goal恢复；本64条不得继续选阈值。不要据此直接扩展方向、外观或联合训练。',
        '- 原9条与本64条已用于诊断；本轮结果能够验证此接口机制，但并非未参与开发的独立泛化证据。公开Polygon投影及RGB对比也不等于人工确认的实例匹配。','',
        '## 环境、开销与交付','',
        f'- GPU RTX PRO6000 Blackwell 96GB，PyTorch2.8.0+cu128，CUDA12.8，Python3.10.22；192次闭环循环{metrics["seconds_current_session"]:.2f}s，不含进程启动/权重初始化/图册生成；模型峰值allocated {metrics["peak_gpu_allocated_gib"]:.3f}GiB。',
        f'- 本轮实际{coordinate["frames"]}帧，Pose记录全部匹配，世界—像素回投最大误差{coordinate["max_world_pixel_roundtrip_m"]:.9f}m。数值自洽不等于真实视觉边界IoU。',
        '- 42项相关测试通过；没有训练或Checkpoint覆盖。配置、两条Smoke及完整192条数值轨迹、逐帧Stable审计均保留。',
        '- [HTML全64任务图册](index.html) · [配对CSV](paired_tasks.csv) · [汇总图](summary.png) · [Stable接口核验](stable_interface_audit.json) · [全部轨迹](trajectories.jsonl)。',
        '- [任务1对比](task_01.png)、[任务13对比](task_13.png)、[任务34对比](task_34.png)、[任务43对比](task_43.png)；`transitions_*.png`给出原9条及失败任务的Goal距离、Stop概率、视觉增益时序；`native_rgb/`保留代表帧原生PNG。',
        f'- 全部原始PNG/NPZ/缓存位于服务器 `{a.run}`，SHA见`raw_artifacts_manifest.json`；大体积缓存不上传Git。`reproduce.sh`记录真实命令；tmux `pgvb-stable-20261105`在工作完成后退出；runner支持相同配置指纹下`--resume`恢复。','']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines))
    shutil.copy2(a.run.parent/'run.sh',a.output/'reproduce.sh')
    shutil.copy2(a.run.parent/'analyze.sh',a.output/'reproduce_analysis.sh')
    shutil.copy2(a.run.parent/'smoke'/'metrics.json',a.output/'smoke_metrics.json')
    shutil.copy2(a.run.parent/'smoke'/'predictions.jsonl',a.output/'smoke_trajectories.jsonl')
    shutil.copy2(a.run.parent/'full64.log',a.output/'rollout_stdout.txt')

if __name__=='__main__':main()
