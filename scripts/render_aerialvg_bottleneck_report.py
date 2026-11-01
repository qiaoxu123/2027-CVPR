"""Render concise Chinese report from completed read-only diagnostic artifacts."""
import argparse,json
from pathlib import Path


def percent(n,d):return f'{100*n/d:.2f}%' if d else 'NA'


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--navigation',required=True);p.add_argument('--output',required=True);a=p.parse_args();x=json.loads(Path(a.candidate).read_text());nav=json.loads(Path(a.navigation).read_text());s=x['splits']['dev'];train=x['splits']['train'];N=s['all']['states'];A=s['all']['a0_correct'];W=s['a0_wrong']['states'];deployed=s['all']['fusion']['a0_guided_spatial/top15'];wrong=s['a0_wrong']['fusion']['a0_guided_spatial/top15'];visible=s['visible_a0_wrong'];key='a0_guided_spatial/top15'
    # Explanatory annotations below bind to this frozen diagnostic snapshot.
    # Refuse different evidence rather than silently reuse numeric conclusions.
    required=(N==635 and A==437 and W==198 and train['all']['states']==1275
              and s['all']['stages']['raw900']['covered']==410
              and deployed['corrected']==22 and deployed['damaged']==15
              and wrong['covered']==104 and wrong['covered_rank_wrong']==50
              and wrong['rank_correct']==54 and wrong['rank_correct_belief_wrong']==37
              and wrong['rank_correct_fail_no_legal_support']==2
              and wrong['rank_correct_fail_bound']==13 and wrong['rank_correct_fail_other']==22
              and nav['history8_vs_current']['helped']==10 and nav['history8_vs_current']['harmed']==7)
    if not required:raise ValueError('Evidence differs from frozen snapshot; regenerate diagnostic interpretation')
    lines=['# AerialVG候选筛选与性能瓶颈诊断','',f'基于 `{x["base_commit"]}`；仅复用Train **{train["all"]["states"]}**、内部Dev **{N}** 状态及已有128条配对归档。没有训练、写权重、读取Test或运行新闭环。正式验证进程已暂停，旧输出保留。','',f'A0定义为缓存的冻结静态 `base_logits`（A3几何分支）在28×28网格上的argmax；Dev正确 **{A}/{N}={percent(A,N)}**、错误 **{W}**。20m均指候选中心/网格点到目标位置的距离，**不是目标实例识别**。所有评分先完成选择，GT仅在其后审计。状态相关且采样有限，以下为描述性计数。','', '## 1. 候选覆盖与位置Oracle','',f'主表为内部Dev，A0正确/错误列分母分别为{A}/{W}。Native与A0-Guided共享有效像素比例≥0.6、NMS IoU=0.5及相同900 Query；Guided仅增加已有的0.5×截断先验分数。空间多样性另列，不与评分效果混淆。','', '|候选集合|Oracle R@1/20m|A0正确覆盖|A0错误覆盖|Oracle NE(m)|保留正确A0的纠错上界R@1|','|---|---:|---:|---:|---:|---:|']
    keys=[('全部900','raw900'),('有效900','native/valid900'),('Native NMS全部','native/nms'),('Guided NMS全部','a0_guided/nms')]+[(f'{label} Top-{k}',f'{strategy}/top{k}') for strategy,label in [('native','Native'),('a0_guided','Guided')] for k in (15,30,60)]+[('现部署Guided+多样性15',key)]
    for label,k in keys:
        v=s['all']['stages'][k];c=s['a0_correct']['stages'][k];w=s['a0_wrong']['stages'][k];ne=f'{v["oracle_NE_nonempty"]:.2f}' if v['oracle_NE_nonempty'] is not None else 'NA'
        lines.append(f'|{label}|{v["covered"]}/{N} ({percent(v["covered"],N)})|{percent(c["covered"],A)}|{w["covered"]}/{W} ({percent(w["covered"],W)})|{ne}|{100*v["keep_a0_oracle_R1"]:.2f}%|')
    lines += ['', 'Oracle NE为非空集合中最小位置误差的平均；空集不填0。上界假设能保留全部正确A0、在A0错误且集合覆盖时由Oracle选择正确中心，**不是导航SR上界**。空集数量见JSON。','', '|Train位置覆盖对照|Native|A0-Guided|','|---|---:|---:|']
    for k in (15,30,60):
        n=train['all']['stages'][f'native/top{k}'];g=train['all']['stages'][f'a0_guided/top{k}'];lines.append(f'|Top-{k}|{n["covered"]}/1275 ({percent(n["covered"],1275)})|{g["covered"]}/1275 ({percent(g["covered"],1275)})|')
    losses=s['all']['first_loss'][key];visible_count=s['target_pixel_valid']['states'];visible_hits=s['target_pixel_valid']['stages'][key]['covered']
    lines += ['',f'**过滤损失：**Dev的900 Query原本覆盖410个状态；现部署保留391个。首次丢失于有效性过滤 **{losses.get("validity_filter",0)}**、NMS **{losses.get("nms_suppression",0)}**、Top-15预算 **{losses.get("budget_truncation",0)}**；其余225个状态在原900中就无20m中心，不能全部解释成检测器失败（许多目标尚不在当前有效视野）。正候选级无效原因主要是depth有效率×保留面积不足，而非单纯框越界，计数见JSON；NMS有“18.45m正确中心被20.36m中心抑制、IoU=0.545”的真实例子。',f'目标像素有效子集的部署覆盖 **{visible_hits}/{visible_count}={percent(visible_hits,visible_count)}**；A0错误且目标像素有效时 **{visible["stages"][key]["covered"]}/{visible["states"]}={percent(visible["stages"][key]["covered"],visible["states"])}**。A0错误全体中，900原始覆盖112/198、部署104/198，只有8个状态的额外位置候选在筛选中完全丢失。Guided评分本身在Top-15较Native新增覆盖87、损失3；不能据此认定Guided是主要召回瓶颈。','', '## 2. 排序与Belief融合','', '固定同一个plain checkpoint、σ=8m和bound=2；30/60仅为超出训练候选预算的离线反事实，不作为可部署结论。排名正确指最佳非Null候选中心≤20m；Null胜出另行统计。','', '|Dev候选集合|有候选但排序错/覆盖状态|排序对但更新错/排序对|纠正A0 / 误改A0|更新后R@1|','|---|---:|---:|---:|---:|']
    for label,k in keys[4:]:
        v=s['all']['fusion'][k];lines.append(f'|{label}|{v["covered_rank_wrong"]}/{v["covered"]} ({percent(v["covered_rank_wrong"],v["covered"])})|{v["rank_correct_belief_wrong"]}/{v["rank_correct"]} ({percent(v["rank_correct_belief_wrong"],v["rank_correct"])})|{v["corrected"]} / {v["damaged"]}|{percent(v["updated_correct"],N)}|')
    lines += ['',f'现部署在Train纠正90、误改50（净+40/1275）；Dev纠正 **{deployed["corrected"]}**、误改 **{deployed["damaged"]}**（净+7/635，68.82%→69.92%）。在真正需要纠错的A0错误子集中：覆盖104个，其中 **{wrong["covered_rank_wrong"]}/104={percent(wrong["covered_rank_wrong"],104)}** 排序错；排序对的54个中 **{wrong["rank_correct_belief_wrong"]}/54={percent(wrong["rank_correct_belief_wrong"],54)}** 仍未纠正。',f'上述37个融合失败：**2**个目标合法网格无当前观测支持；**13**个在现有证据质量/增益、支持域和幅度约束下，即使用GT理想化加减残差也不能翻转先验；其余 **22**个未被这个逐网格独立加减残差的放松界排除，但不保证当前高斯核与概率混合可实现翻转；混合、校准、栅格化及实例偏差各自贡献**未知**，不能全部归咎于bound太小。A0错误且有20m候选时Null胜出40/104，但位置邻近不能证明是真实目标实例，不能直接断言这40次Null判断全错。','', '## 3. 已有128条导航：定位、控制与Stop证据','', '只读取三组已存Val-Seen128轨迹，按本轮要求事后描述；不读取新的Val-Seen标注。用配对终点与已有NE通过三角不等式计算目标距离上下界，不猜目标坐标。**证据可重叠，不把终点偏差强行等同于唯一根因。**','', '|归档配置|OSR成功/SR失败|其中终点错误目标证据|其中主动错误Stop|其中可证实曾在20m内但未停|目标>20m跳变/转换|','|---|---:|---:|---:|---:|---:|']
    for tag,label in [('baseline','原基线'),('current','单帧Native'),('history8','8帧Native')]:
        allv=nav['summary'][tag]['all'];v=nav['summary'][tag]['OSR_success_SR_failure'];n=v['episodes'];loc=v['causes'].get('定位偏差证据',{}).get('count',0)
        lines.append(f'|{label}|{n}/128 ({percent(n,128)})|{loc}/{n} ({percent(loc,n)})|{v["false_policy_stop"]}/{n} ({percent(v["false_policy_stop"],n)})|{v["certified_near_not_stopped"]}/{n} ({percent(v["certified_near_not_stopped"],n)})|{allv["goal_jumps_gt20"]}/{allv["goal_transitions"]} ({percent(allv["goal_jumps_gt20"],allv["goal_transitions"])})|')
    lines += ['', '全部失败中，单帧64/64、8帧61/61均有终点选中目标在20m外的定位偏差证据；未发现可**独立证实**为控制不收敛或Stop阈值错误的案例，不能因此断言它们不存在。唯一根因和未能确认的中间到达时刻标为**未知**。单帧OSR+/SR−中已证实到达后未停的11条，存在44次精确到达守卫拒绝、8次低Stop概率拒绝（事件可重复）；这说明观察到区域到达不等于语义和守卫允许停止，不在本轮改阈值。',f'8帧相对单帧：改善{nav["history8_vs_current"]["helped"]}条、损害{nav["history8_vs_current"]["harmed"]}条。跳变率15.48%→7.07%；但7条受损任务均出现≥3步持续错误目标并错误停止（单帧对应任务曾持续错误目标3/7）。8帧机制是有界均值而非无界相加；可怀疑旧错误证据存续/锁定，但轨迹已经分叉，且日志缺每帧Query后验和当前/历史delta分解，**是否由错误证据累积造成仍未知**。','', '## 4. 最主要的两个瓶颈与下一步方案（本轮不实施）','', '1. **实例消歧/候选排序，而非单纯候选数量。**A0错误且目标有效可见时部署位置覆盖已93.10%，但有候选时近半数非Null排序错误；Guided Top-15→60只再覆盖10个Dev状态，冻结模型全局R@1反而69.76%→68.66%。建议在现有小候选池上区分目标/参照物角色，给关系推理显式相对几何，并以实例标注或可核验对应检查“20m邻近假阳性”；同步诊断Null校准。先在Train/内部Dev验证，不直接扩大正式预算。', '2. **视觉排名到全局Belief的转换与历史可信度门控。**A0错误但排序正确仍有68.52%未纠正，且误改15个正确A0。建议记录query概率、正/负区域质量、先验margin、当前/历史delta并做因果离线重放；再研究支持域内的置信校准和保守更新，历史按世界位置关联、时效与冲突衰减。不能仅调大bound或放宽Stop；22/37失败仍有理论余量，需要区分混合/定位与真实实例问题。','', '## 复现与完整性','',f'选择复现 **{x["selection_reproduction"]["deployed_selection_matches_prepared"]}/{x["selection_reproduction"]["states"]}**；Checkpoint SHA256 `{x["checkpoint_sha256"]}` 前后不变。数据、参数及输入manifest哈希见 `candidate_summary.json`；导航来源及逐案例未知标记见 `navigation_summary.json`。脚本和单元测试独立新增，原实验代码、权重和评测协议未修改。']
    Path(a.output).write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
