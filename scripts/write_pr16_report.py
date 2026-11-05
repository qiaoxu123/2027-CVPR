"""Summarize preserved BF16 and TRAIN-justified FP32 fixed56 diagnostics."""
import argparse
import json
from pathlib import Path
import shutil


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.root;o=a.output
    bf=json.loads((o/'summary.json').read_text());fp=json.loads((o/'fp32/summary.json').read_text())
    train=json.loads((r/'train_audit_fp32/summary.json').read_text());prec=json.loads((o/'precision_batch_audit.json').read_text());dev=train['state_metrics_by_subset']['dev']
    methods=['Geometry','Relation','RGB']
    lines=['# PR #16：直接关系证据安全与真实闭环诊断','',
      '结论：代码/绑定/数值接口修复通过CPU与GPU验证，但没有证据证明Relation或RGB优于Geometry。方向Head与目标实例识别能力仍未通过研究验收；PR保持实验状态，不合并main。','',
      '## 修改与验证范围','',
      '- 基于PR #16 Head `b4584aad…`，使用原Checkpoint `training_aligned/best.pt`，SHA256 `aa359894a172c38b024aa5ef4e89caeecc8a496d723320cea14862f0eaaf61e9`。不重新训练、不改历史权重/报告、不访问Test。',
      '- 修复新增测试夹具缺少predicate；公开Anchor按当前地图Catalog重新绑定名称、全部对象ID与轮廓，防止任务文件给出错误Polygon。处理真实Catalog整数ID与JSON字符串ID差异；不同地图不混用。Geometry保留全部同名片段，RGB在多Anchor/多ID或不支持类别时弃权，不按GT择优。',
      '- 当前帧PGVB正负区域及错位特征对照通过、连续确认后才允许目标RGB残差。它证明的是Anchor区域有暂定RGB支持，不是目标实例已识别。offscreen/追踪丢失不能添加残留证据。',
      '- 未校准方向回退时，有效Relation/RGB字段与实际导航Geometry一致；被拒绝的原始Head热图另存，避免把未应用输出当成导航Belief。Geometry本身仍编码原始方向语言，回退不是“无方向模型”。',
      '- 正式评估遇到无候选/接口错误也保留任务并计为失败，完成固定分母；TRAIN Smoke默认首错停下便于定位。',
      '- Stable Goal阈值仍为0.35和连续2帧；Stop阈值0.3、1mm到达守卫、原Trajectory/Stop权重、20步Horizon及20m成功半径不变。200帧CPU等价回放含10次重规划，候选ID/XY/Context与旧实现完全相同，PR只增加拒绝原因。','',
      '## 数值问题：基于TRAIN修复，非Val调参','',
      f'- 80条TRAIN开发查询、240状态，同一模型/输入比较batch=1与8：BF16有{prec["bf16"]["geometry_different_states"]}个Geometry、{prec["bf16"]["relation_different_states"]}个Relation Top-1不同，最大logit差{prec["bf16"]["max_abs_logit_difference"]:.6f}。',
      f'- FP32对应Top-1差异均为0，最大logit差{prec["fp32"]["max_abs_logit_difference"]:.8f}。因此小型Belief核心默认FP32；冻结SigLIP2编码和原导航控制器仍按原精度运行。没有根据GT或Val SR选精度。',
      '- BF16完整56×3已先完成并保留。数值错误由TRAIN批尺寸实验独立复现后，代码再冻结，重新执行同一56×3 FP32复测。两轮完整结果均报告，不隐去FP32中Geometry少成功1条的负面变化。','',
      '## TRAIN数据与方向/实例检查','',
      '- 573条查询，493拟合/80开发，1717缓存状态；467单公开ID，106同名多ID。当前任务文件和真实公开Catalog的绑定差异为0；修复的是可复现的错误输入接口，而不能宣称真实数据中错误Polygon已被大量纠正。',
      '- 5条目标ID与Anchor ID重叠；50个目标点在Anchor并集内。Inglis Building、17 Mill Lane、Department of History of Art等涉及命名范围/物理实例粒度歧义；不能据此一概判GT错，也不能用GT挑选多Polygon中的某个片段。完整案例见TRAIN/TRAIN_ID_conflicts.json。',
      '- TIF没有CRS，正射图的+Y数值增大不能独立证明地理北；不依据GT翻轴。当前工程轴下，north 26条只有10条在质心北侧、3条在完整Polygon北侧；south 40条只有12条在质心南侧、0条在完整Polygon南侧。粗轮廓、长道路/多Polygon、语言作用域及坐标定义共同存在不确定性，不能凭统计强行归一为坐标反转。',
      f'- 同步修改原始英文north/south、结构ID，并重算BERT、节点/规则特征：63条TRAIN探针中，Geometry有{train["synchronized_north_above_south"]["Geometry"]*100:.2f}%满足原工程预期的north平均Y高于south，新增Relation为{train["synchronized_north_above_south"]["Relation"]*100:.2f}%。这是完整文字干预的分布敏感性测试；不为改写指令伪造GT，不是地理方向金标准。',
      '- 当前不能确认方向偏移正确，因此保留回退。仅增加Epoch不能解决地理口径与实例监督不确定性，本轮没有重新训练新网络。','',
      '### 地图隔离TRAIN开发组（80查询、240状态）','',
      '|方法|位置R20|NE(m)|','|---|---:|---:|']
    for m in ('Geometry','RawRelation','RawRGB','SafeRelation','SafeRGB'):
        q=dev[m];lines.append(f'|{m}|{q["R20"]*100:.2f}%|{q["NE"]:.3f}|')
    lines += ['', '- 三帧不是三个独立实例；TRAIN稀疏起/中/末观测只作因果缓存回放，不冒充连续闭环。拟合组另表保留，不能与开发组合并宣称泛化收益。历史Geometry本身已在TRAIN预训练，此处隔离仅指本轮新增Head拟合。',
      '- 无独立人工像素Mask/实例标注。目标是否真实可见、像素IoU及实例Top-1均为未知；20m位置正确不能证明选中同一辆车或同一栋楼。','',
      '## 同一固定56条Val-Seen：三组完整配对','',
      '任务清单SHA256：`6dc11959e4a94ba14adc20984e9973ca9739face16d5388c2bc32f4e7c2b5c14`。来源2470条Val-Seen中的原冻结严格可解析子集，不能称整个CityNav SR。全部任务含无证据、回退及失败均在分母。三组共享同一联合训练后Checkpoint，不是历史未微调Geometry的闭环复现。','',
      '|精度|方法|完成|SR|OSR|SPL|NE(m)|nDTW|步数|推理ms|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for precision,data in [('BF16保留',bf),('FP32数值修复',fp)]:
        for m in methods:
            q=data['closedloop'][m];lines.append(f'|{precision}|{m}|{q["episodes"]}/56|{q["SR"]*100:.2f}%|{q["OSR"]*100:.2f}%|{q["SPL"]:.4f}|{q["NE"]:.3f}|{q["nDTW"]:.4f}|{q["steps"]:.2f}|{q["inference_ms"]:.2f}|')
    lines += ['', 'FP32配对变化：','', '|比较|纠正|破坏|净收益|','|---|---:|---:|---:|']
    for key,q in fp['paired_changes'].items():lines.append(f'|{key}|{q["corrected"]}|{q["damaged"]}|{q["net"]:+d}|')
    lines += ['', f'FP32峰值分配显存约{fp["GPU_peak_GiB"]:.3f}GiB（进程内，含冻结视觉模型）。服务器存在其他GPU任务，时延不是独占硬件基准；不包含一次性模型加载。两轮均0推理异常，三组均56次Policy Stop、0 Horizon。错误Stop是到达错误预测Goal后停止，不等于Stop阈值错误。','',
      '## RGB、Belief偏移与Stable Goal','',
      f'- FP32 RGB：{fp["RGB_decisions"]}次真实决策，{fp["RGB_admitted"]}次通过当前帧门控；重规划{fp["replan_events"]}次。门控是实际生效的，不是仅增加开关。',
      f'- RGB相对Relation有{fp["raw_RGB_top1_changes"]}次逐帧Top-1变化，其中按20m位置代理纠正{fp["raw_location_proxy_corrections"]}帧、破坏{fp["raw_location_proxy_damages"]}帧；帧数不能当作任务数或像素实例准确率。',
      f'- 有确认RGB的最大新Goal logit优势{fp["max_confirmed_visual_logit_advantage"]:.4f}，低于原0.35门槛。没有完成重规划；原因明细见fp32/blocked_RGB_changes.json。错误位置候选也出现，不能依据少数正确帧降低门槛。',
      '- FP32当前帧源状态：168次有可编码Anchor、186次无支持局部Anchor、53次因多Anchor/多ID拒绝视觉、41次方向回退。无观测支持不能直接归为“目标不可见”或检测失败。',
      '- 门控确认的是Anchor，不是目标；近邻关系不足以区分相似车辆/建筑。RGB局部证据既不能稳定识别实例，也不足以可靠推翻错误主峰。这比Stop或运动控制更符合本轮失败证据。',
      '- 所有失败轨迹最终选择的Goal均在真实目标20m外；并非Goal正确却到达不了。OSR高于SR的任务确实曾经过目标邻域，但最终仍飞向错误Goal，不应把它们直接归因于Stop故障。','',
      '## 剩余问题和下一步','',
      '1. 首先校正目标—具名地标的实体粒度、关系作用域和可独立验证的方向坐标口径。对多片段道路保留完整几何，未来仅根据当前可观测支持保留多个局部候选，不能据GT挑片段。',
      '2. 针对目标实例训练可靠匹配/空间证据，而不是拿Anchor识别当Target识别。使用TRAIN实例/Contour时保持弱监督与不确定性，增加同图同类负样本；方向变换先用合成几何验证显式算子，再用可靠真实关系校准距离与分布。',
      '3. 在独立TRAIN开发组证明方向控制与RGB反事实增益后再重训；不要继续加Attention层、加Epoch或降低Stable Goal/Stop阈值。现有安全补丁可研究使用，但没有性能提升证据，暂不合并main。','',
      '## 测试、审查与复现','',
      '- 339项已跟踪CPU测试通过，1项上游rasterio弃用警告；工作区额外3项旧未提交记忆测试也通过，但未混入本PR可复现计数。Python编译检查通过。',
      '- 2条TRAIN真实GPU Smoke、每种方法各2条均跑通；首次Smoke发现整数/字符串ID错误，失败日志保留在服务器TRAIN_smoke目录，修复后的TRAIN_smoke_v2和TRAIN_smoke_fp32独立保存。',
      '- 实际调用codex adversarial-review，但workspace routing discovery超时，未得到有效独立审查结论；原输出保存在adversarial_review_tool.txt，不能说该审查已通过。',
      f'- 原权重、全部缓存、逐步RGB/NPZ、失败Smoke及两轮完整闭环保留在 `{r}`；原实验位于相邻unified_direct_target_s42_20261105目录，未覆盖。',
      '- 代码/配置/训练审计及两轮结果按阶段Commit。仅公开地图和当前观测进推理，目标GT仅用于TRAIN监督/事后评分，Test未访问。',
      '- 已用隔离Firefox实际验证7个案例的时间轴切换、复合图加载与原生RGB链接，临时HTTP/WebDriver服务已停止。切换图片时浏览器取消旧HTTP请求产生的BrokenPipe日志保留；交互断言全部通过。',
      '- 图册：FP32见fp32/index.html；BF16见index_bf16.html；全部逐任务配对及168条轨迹分别保存。绿色叉为事后GT点，黄色/青色为公开Polygon投影，白色虚线为相机Footprint，均不代表人工像素分割金标准。']
    (o/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    (o/'fp32/REPORT_zh.md').write_text('# FP32数值修复复测\n\n参见[完整中文报告](../REPORT_zh.md)。本目录保留全部56×3结果与逐时刻RGB/热图。\n')
    if not (o/'index_bf16.html').exists():shutil.copy2(o/'index.html',o/'index_bf16.html')
    (o/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>PR16诊断总览</title><style>body{font:18px sans-serif;max-width:1400px;margin:auto}img{width:100%}iframe{width:100%;height:1000px;border:1px solid #ccc}</style><h1>PR #16：固定56条真实闭环诊断</h1><p>未训练新模型；未使用Test；未调Stop/Goal阈值。FP32为TRAIN数值不变性修复，不是按Val选成绩。</p><p><a href="REPORT_zh.md">中文报告</a> · <a href="fp32/index.html">FP32逐时刻图册</a> · <a href="index_bf16.html">完整保留BF16图册</a></p><img src="fp32/summary.png"><iframe src="fp32/index.html" title="FP32真实RGB与Belief"></iframe>''')
    (o/'STATUS_zh.md').write_text('# 已完成状态\n\nPR16代码与接口修复、339项CPU测试、TRAIN审计/同步方向探针、两条GPU Smoke、原固定56条三组BF16诊断及TRAIN数值修复后的FP32复测均完成。原权重未变，无Test访问。未证明Relation/RGB提升SR，方向与实例能力未通过研究验收，不合并main。详见REPORT_zh.md。\n')
    print('Chinese report written',o/'REPORT_zh.md')


if __name__=='__main__':main()
