"""Write the Chinese decision report from measured, sealed experiment artifacts."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'anchor':'Anchor不移动','centroid32':'质心点+旧32参数核','no_direction':'已有无方向模型','original32':'原32参数（完整Anchor）','polygon_ray':'Polygon-Ray（辅助）','polygon_cone':'Polygon-Cone（预定主方案）'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();o=a.directory;scan=json.loads((o/'scan_manifest.json').read_text());selection=json.loads((o/'selection_seal.json').read_text());split=json.loads((o/'split_seal.json').read_text());decision=json.loads((o/'development_decision.json').read_text());diag=json.loads((o/'diagnosis.json').read_text());metrics={**json.loads((o/'development_metrics.json').read_text()),**json.loads((o/'confirmation_metrics.json').read_text())};cd=json.loads((o/'confirmation_diagnostics.json').read_text())['confirm'];pred=json.loads((o/'prediction_seal.json').read_text());telemetry=json.loads((o/'region_telemetry.json').read_text());pub=json.loads((o/'public.json').read_text());cases=json.loads((o/'cases.json').read_text());idx={(r['episode'],r['word']):r['image'] for r in cases}
 fig,axes=plt.subplots(1,2,figsize=(10,3.5));axes[0].bar(['Retrieved','Text accepted','Geometry valid'],[scan['candidate_pairs'],selection['selected_pairs'],split['total_pairs']]);axes[0].set_title('All 22,002 TRAIN instructions scanned');words=['north','south','east','west'];axes[1].bar(words,[sum(r['word']==w for r in pub) for w in words]);axes[1].set_title('Valid direct pairs (not independent instances)');fig.tight_layout();fig.savefig(o/'data_summary.png',dpi=140);plt.close(fig)
 L=['# 全量TRAIN直接方向关系与完整Polygon区域基线','',
'## 结论与执行状态','',
'**本轮实现的完整Polygon方向区域没有超过旧32参数模块。** 新确认组30条：原模型R@20m=9/30（30.00%），Polygon-Cone=6/30（20.00%），纠正3、破坏6、净收益−3；NE由47.9244m升至50.0813m，GT20概率质量由0.12790降至0.06444。此处是离线定位，不是导航SR。',
'开发组的预定训练门槛三项均未通过，因此**没有训练新的轻量模型**。未训练模型不填造指标、不创建伪Checkpoint。完成本轮先行数据筛选、无训练基线和一次性独立确认；不把负结果作为继续叠加网络的理由。','',
'## 1. 全量筛选与新分组','',
'- 重新扫描全部22,002条TRAIN原文，逐条与官方TRAIN JSON核对文本和地图，22,002/22,002完全一致。重用的是原文和公开名字span缓存，不是此前154条候选列表。完整索引见[full_scan.json](full_scan.json)。',
'- 以方向谓词+相邻公开名字及依存候选的并集检索，得到634处north/south/east/west of或side of词法谓词、427条有公开地标绑定的关系候选（395条指令）。复合方向不是本轮四个独立方向；专有名词中的West/East本身不算关系。未绑定谓词、无法识别的句式仍可能漏检，不能声称召回了全部真实直接关系。',
'- 当前AI逐句核对候选，排除目标→停车场/车排/工地/另一道路→地标的链、朝向、版面参考、作用域和多目标歧义。保留427条原文、span、实体来源和逐条决定，[decisions.json](decisions.json)可复查；**没有人工语言Gold，不能报告人工解析准确率**。',
'- 文本保留274条：north142、south130、east1、west1；公开几何检查再排除9条（任一同名片段无效则整条弃权，不根据GT挑其余片段），最终265条/243个Episode：north139、south124、east1、west1。线性道路/人行道/桥的同名合法片段共同保留，跨建筑/道路身份混合则弃权。',
'- 新增west例2552的“Full Circle Events building”通过唯一公开名字前缀绑定Full Circle Events & Exhibitions Ltd；774同理绑定Nike Factory Store。仅为有记录的文本简称绑定，不宣称已人工确认实际实例。east例14382的目标本身是Parking Area，所以其与两个具名建筑的关系属于直接关系，而非“车辆→停车区域”的间接链。',
'- 按城市限定的公开地标名称、Episode、地图限定官方目标实例ID建立连接组。对象ID仅用于划分隔离，不进入推理；未按GT坐标、方向一致性或模型成败筛选。旧154条曝光组强制放入拟合池，新组按预定SHA顺序划分。新开发组和确认组均不重用此前曝光的地标/Episode/官方实例组。','',
'|分组|关系数|组数|north|south|east|west|','|---|---:|---:|---:|---:|---:|---:|',
'|可用拟合池（本轮未拟合）|215|101|114|99|1|1|',
'|新开发组|20|14|13|7|0|0|',
'|新确认组|30|13|12|18|0|0|','',
'地图可能重叠；这是新的地标/实例组确认，**不是未见地图泛化**。east/west各1条且在拟合/曝光组，不能给出这两个方向的独立泛化结论。同一Episode的多个关系不当作独立物理目标。官方ID按地图分组，未额外宣称解决所有跨Block全局实例ID问题。','',
'## 2. 新基线改变了什么','',
'需要纠正一个前提：旧32参数模块已经使用完整Polygon面积栅格Anchor，并非只输入质心；它对整个Anchor卷积一个词条件位移核。本轮额外加入“质心点+相同旧核”对照，避免把完整Polygon输入本身误称为新贡献。','','- **Polygon-Ray**：沿每个网格横向截面求完整Polygon的方向前缘，向指定工程方向延伸，保留不同距离和分离片段的候选。',
'- **Polygon-Cone（预定主方案）**：每片段64个横向采样取联合前缘，沿45°半角方向锥扩展；不压缩成质心或固定60m平移。每个候选权重为cos²(angle)/(1+distance²/scale²)，scale=max(20m,sqrt(公开并集面积)/2)，取前缘核的最大响应后归一化。允许多片段、多距离、多峰/脊状分布，不强制单峰高斯。',
'- 两者均在公开地图边界内预测，并将Polygon内部从方向候选中排除；没有合法网格支持时明确弃权，评分使用不移动Anchor回退。这里的“排除内部”是需要验证的假设，**不是道路Polygon的事实语义**。',
'- 所有方法相同28×28网格、相同公开Anchor、相同目标GT与阈值。输入仅公开Polygon/地图范围/方向ID，不输入目标外观、GT目标位置/ID、未来帧或RGB。PGVB、SigLIP和导航器没有运行。',
'- 保持north=工程+Y、south=−Y、east=+X、west=−X；地理含义仍未证实，不用GT翻转、旋转坐标或选择路段。','',
'## 3. 测量结果','',
'R@20m使用最大概率格中心到原始目标GT的欧氏距离；NE同一点误差。GT20概率质量是目标20m圆内的概率总和，不是实例匹配准确率、像素Mask IoU或闭环SR。',
'所有方法的预测先封存，之后独立进程评分。Polygon-Cone是否训练只由新开发组决定；预定门槛为GT20质量高于原32参数，且R20和NE均不变差。三项均失败后冻结“不训练”决定，才导出、评分30条确认标签；未按确认结果修改公式。','']
 for sub,title in [('dev','新开发组（20条）'),('confirm','新确认组（30条，一次性）')]:
  L+=['### '+title,'','|方法|R@20m|NE (m)|GT20概率质量|','|---|---:|---:|---:|']
  for k,name in NAMES.items():
   v=metrics[sub][k]['all'];L.append(f"|{name}|{v['correct']}/{v['n']}（{100*v['R20']:.2f}%）|{v['NE']:.4f}|{v['GT20_mass']:.5f}|")
  v=metrics[sub]['polygon_cone_vs_original32'];L+=['',f"Cone相对原32参数：纠正{v['corrected']}、破坏{v['damaged']}、净收益{v['net']:+d}。",'']
 L+=['### 新确认组按方向','','|方向|数量|原32 R20 / NE / GT20质量|Polygon-Ray R20 / NE / GT20质量|Polygon-Cone R20 / NE / GT20质量|','|---|---:|---|---|---|']
 for w in words:
  n=metrics['confirm']['original32']['by_word'][w]['n'];cells=[]
  for k in ['original32','polygon_ray','polygon_cone']:
   v=metrics['confirm'][k]['by_word'][w];cells.append('—' if not n else f"{v['correct']}/{n} / {v['NE']:.2f}m / {v['GT20_mass']:.5f}")
  L.append('|'+w+'|'+str(n)+'|'+'|'.join(cells)+'|')
 L+=['','拟合池及全部方法/分方向结果见[development_metrics.json](development_metrics.json)；新确认结果见[confirmation_metrics.json](confirmation_metrics.json)，逐样本见[samples.csv](samples.csv)。新轻量模型：**未训练，指标不适用**。',
'按13个确认组进行5,000次配对bootstrap，Cone−原32参数的R20差95%区间为[−36.0,+8.6]个百分点，NE差区间[−9.65,+14.75]m。样本小、地图重叠，不能宣称统计显著劣化或泛化提升。','',
'## 4. 为什么完整Polygon未带来可靠收益','',
'1. **候选区域覆盖不等于最终定位。** Cone在21/30条确认关系的20m目标邻域内有合法网格候选，但只有6/30峰值命中；15条“区域有支持、最终峰值未中”。其平均支持288.3/784格，约36.8%的地图网格，因此70%区域支持率不能冒充定位准确率。宽候选区域需要更可靠的局部侧向、距离或多约束信息。',
'2. **道路Polygon不宜一律视为禁止目标进入的占据区域。** 后验发现9/30目标GT处于公开道路Polygon内部（Leslie Road、Thornbury Road、Victoria Street、Manhattan Drive相关例）；若语言是“道路北侧”，完整道路Polygon可能是粗区域而非精确路面分割。排除内部会丢失目标中心的概率支持。20m圆仍可能触及外侧区域，不能把这9条全部计作不可恢复，也不能由此证明公开Polygon标错。',
'3. **方向/作用点问题仍未消失。** 仅14/30的GT相对Anchor均值符合原工程方向符号，3/30位于整个Polygon的反侧。新确认组有9/30的20m邻域连Cone候选支持也没有。不能据这些标签反向重设坐标；需要独立核实语言参考系、道路区域语义和局部Anchor作用点。',
'4. **射线版本只改善了平均距离，没有提高成功定位。** 确认NE47.92→41.99m，但R20 9→6、GT20质量0.12790→0.08287；不能因单项NE改善就在确认组改选它并继续训练。',
'5. 单个方向词不包含准确距离；面积自适应的重尾距离先验本轮也没有弥补该缺失。不是网络层数或训练轮数不足的证据。','',
'## 5. 真实地图案例与可视化','']
 for e,w,desc in [(496,'north','改善：目标在道路Polygon内部；Cone峰值命中不代表内部排除假设正确'),(6453,'south','破坏：原模型约19.5m，Cone约33.3m；有方向区域仍选不到正确距离/侧向位置'),(2895,'north','GT在整个道路Polygon反侧；坚持原工程方向的区域仍不能覆盖'),(14382,'east','罕见east例，仅拟合/曝光组说明，不作为独立验证')]:
  L.append(f'- [{e}：{desc}]({idx[(e,w)]})。')
 L+=['','[HTML图册](index.html)展示全部30条确认关系、east/west各例及4条开发例，共36张真实地图对照图；[四方向区域控制图](four_direction_regions.png)、[数据统计图](data_summary.png)、[确认结果图](confirmation_summary.png)。绿色GT仅后验叠加，地图背景不输入模型，不是目标可见性或像素标注。','',
'## 6. 验证、恢复与下一步','',
'- 30项相关测试通过：四轴、连续距离、多片段、旋转/平移一致性、空支持弃权、无效Polygon、关系链排除、GT字段拒绝、组隔离与预测SHA；见[tests.log](tests.log)。新增几何测试先红后绿，日志保留。',
'- 原模型SHA、输入/脚本/预测SHA分别记录于scan/selection/split/prediction/scoring的seal文件。Checkpoint不覆盖，普通导航入口未修改。',
'- CPU四线程265例冻结前向总计约1.64秒（含全部方向几何及旧模型对照）；每例均摊旧32参数约0.30ms、Ray约1.06ms、Cone约4.05ms，包含候选缓存，不代表在线单例完整系统时延。无GPU模型调用或GPU显存开销。',
'- 私有标签保存在服务器`GraphNav/runs/polygon_direction_region_v1/`，推理不读取。恢复顺序：scan→select→prepare→evaluate→export fit_dev→score development→（门槛通过才训练；本轮不通过）→export confirm→score confirmation→report→summarize。完整命令见[REPRODUCE.md](REPRODUCE.md)，参数见各脚本`--help`；预测和确认评分已有覆盖保护。',
'- **下一步先修几何语义而非扩大网络**：在拟合/开发数据核实道路Polygon是否为粗覆盖区域，设计允许内部目标的类型条件软方向约束；独立核验地理方向与语言参考系。再在新的组上检验局部边界/距离建模。本轮30条确认已使用，不应继续当作调参确认集。',
'- 当前结果只否定本轮“前缘外侧射线/45°锥+重尾距离”的具体实现能可靠优于旧模型，**不证明完整Polygon信息没有价值**，也不证明可通过翻转坐标解决。没有充分证据时不训练、不进行闭环。']
 (o/'REPORT_zh.md').write_text('\n'.join(L)+'\n');status=dict(completed=['full22002_text_scan','candidate_scope_review','public_geometry_validation','grouped_split','frozen_baselines','development_gate','one_shot_confirmation','visualizations','tests'],training='skipped: three development gate criteria failed',new_checkpoint=None,confirmation_scored_once=True,PGVB_navigation_weights_changed=False,closed_loop_run=False,formal_Val_Test_accessed=False);(o/'STATUS.json').write_text(json.dumps(status,indent=2));print('Chinese report and state saved.')
if __name__=='__main__':main()
