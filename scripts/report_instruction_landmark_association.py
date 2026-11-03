"""Current-session semantic review, stratified estimates, and Chinese corpus report."""
import collections,csv,hashlib,html,json
from pathlib import Path
import numpy as np
P=Path('reports/instruction_landmark_association')
# This session directly read the independently sampled 101 English instructions.
# These are AI semantic-review labels, not human annotations or pixel GT.
OVERRIDES={12:'indirect_geometry',20:'indirect_geometry',26:'indirect_geometry',31:'direct_geometry',36:'direct_geometry',39:'uncertain',43:'direct_geometry',45:'direct_geometry',52:'direct_geometry',56:'direct_geometry',61:'indirect_geometry',68:'direct_geometry',75:'direct_geometry',80:'uncertain'}
DIRECT_UNCERTAIN={89,91,95,99,100}
UNCERTAIN_KEEP={94}
NOTES={12:'车位于停车场后部；停车场在Full Circle右侧，修正错误的关系作用域。',20:'树、棕色建筑等中间参照连接Beche Road；不能把末端across直接附给目标车。',26:'车在停车场内，停车场在Bus Depot街对面。',31:'Master’s Lodge与Cloister Court是两端具名建筑，后缀buildings非新增对象。',36:'逗号后parked in front of Epworth Court更自然修饰目标车；保留邻车用于实例识别。',39:'两层between修饰目标车或yellow object不够明确，保留不确定。',43:'第二句The building明确重指目标且直接on Trumpington Street。',45:'roof/front door是房屋属性；房屋本身在具名街角，并非独立中间地标。',52:'停车场直接between两个具名道路/建筑，Traffic Road/Building是类型后缀。',56:'目标为一排房屋这个集合，整体直接on Grafton Street。',61:'公开目录确有West Midlands Police Custody Suite；小写/plural suites导致字面匹配失败；车→停车场→地标。',68:'同地图公开名The Broadway，原文The Broad way为空格变体；房屋直接on该路。',75:'同地图公开名Acrefield Drive；acre Field Drive为空格变体，车直接off该路。',80:'同地图有Parsonage Street，但personage street是疑似拼写错误；未自动当成同一名字。',89:'playground本身in front of具名Petersfield Mansions。',91:'目标建筑直接位于具名Sport Centre右侧，且道路直接命名Gonville Place。',94:'句子断裂，parked in between的修饰范围有歧义；可见地图名但链路保留不确定。',95:'another building named Kettle’s Yard Gallery绑定参照，目标与其直接next to。',99:'目标楼一侧树、另一侧Jesus College Forum，直接关系，the other指另一侧。',100:'目标黑车直接在Willmore Road左侧，邻车描述为局部消歧。'}

def main():
 records=json.loads((P/'records.json').read_text());summary=json.loads((P/'summary.json').read_text());sample=json.loads((P/'manual_review_sample.json').read_text());assert len(sample)==101
 review=[]
 for i,r in enumerate(sample):
  gold=r['category']
  if i>=81:gold='direct_geometry' if i in DIRECT_UNCERTAIN else 'uncertain' if i in UNCERTAIN_KEEP else 'indirect_geometry'
  gold=OVERRIDES.get(i,gold)
  review.append({'sample_index':i,'episode':r['episode'],'automatic_category':r['category'],'reviewed_category':gold,'instruction':r['instruction'],'reason':NOTES.get(i,'原文有目标→中间区域/实例→具名地标的多跳关系；自动作用域或共指未连通。' if gold!=r['category'] else '按原文目标、具名参照和中间区域核对；与当前分类一致。'),'reviewer':'current Codex session direct semantic review; no external LLM API','human_verified':False})
 (P/'session_semantic_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2))
 cats=['direct_geometry','indirect_geometry','direct_identity','no_confirmed_anchor','uncertain'];counts=summary['mutually_exclusive'];rng=np.random.default_rng(20261103);weighted=np.zeros(4);boot=np.zeros((10000,4));confusion={}
 def group(c):return 0 if c.startswith('direct_') else 1 if c=='indirect_geometry' else 2 if c=='no_confirmed_anchor' else 3
 for c in cats:
  rr=[r for r in review if r['automatic_category']==c];labels=np.array([group(r['reviewed_category']) for r in rr]);confusion[c]=dict(collections.Counter(r['reviewed_category'] for r in rr));freq=np.bincount(labels,minlength=4)/len(labels);weighted+=counts[c]*freq
  for i in range(len(boot)):boot[i]+=counts[c]*np.bincount(rng.choice(labels,len(labels),replace=True),minlength=4)/len(labels)
 estimates={name:{'estimated_tasks':float(weighted[i]),'estimated_percent':float(weighted[i]/len(records)*100),'stratified_bootstrap95_percent':(np.quantile(boot[:,i]/len(records)*100,[.025,.975])).tolist()} for i,name in enumerate(['direct_including_identity','indirect','no_explicit_reliable_anchor','uncertain'])}
 corrected={'review_n':len(review),'reviewer':'current-session AI, not independent human gold','sampling':'30 direct,30 indirect,1 identity,20 no-anchor,20 unresolved; deterministic seed20261105; frozen parser before final review','confusion':confusion,'estimates':estimates,'warning':'Sampling intervals do not include reviewer error or parser design uncertainty; estimates are not exhaustive human counts.'};(P/'review_adjusted_estimates.json').write_text(json.dumps(corrected,ensure_ascii=False,indent=2))
 # Counts for all texts, rather than best available path in a selected subset.
 intermediate=collections.Counter();no_sub=collections.Counter();uncertainty=collections.Counter()
 for r in records:
  if r['category']=='indirect_geometry' and r['paths']:intermediate.update(set(r['paths'][0]['labels'][1:-1]))
  if r['category']=='no_confirmed_anchor':no_sub['generic_visual_reference_only' if r.get('generic_target_references') else 'no_parsed_generic_reference']+=1
  if r['category']=='uncertain':uncertainty[r['reason']]+=1
 (P/'additional_counts.json').write_text(json.dumps({'frequent_intermediate_nodes':intermediate.most_common(20),'no_anchor_subtypes':dict(no_sub),'uncertainty_reasons':dict(uncertainty)},ensure_ascii=False,indent=2))
 with (P/'all_instructions.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['episode','map_name','target_category','category','instruction','target','paths','public_names','unbound_name_cues']);writer.writeheader()
  for r in records:writer.writerow({k:(json.dumps(r[k],ensure_ascii=False) if k=='paths' else r.get(k,'')) for k in ['episode','map_name','target_category','category','instruction','target','paths','unbound_name_cues']}|{'public_names':' | '.join(sorted({n for m in r['mentions'] for n in m['public_names']}))})
 n=len(records);d=counts['direct_geometry']+counts['direct_identity'];i=counts['indirect_geometry'];overlap=summary['overlap']['direct_geometry_path']+summary['overlap']['indirect_geometry_path']-counts['direct_geometry']-counts['indirect_geometry']
 lines=['# CityNav TRAIN：目标与公开地标关联方式分析','', '**结论：绝大多数指令可以利用公开地标，但“直接关联”与“先定位停车场/车排/参照物再定位目标”都占很大比例。当前会话101条分层语义复核加权估计，约51%直接关联、46%间接关联；约1%缺少明确可绑定的地图锚点，约2%仍不确定。这些是抽样校正估计，不是22,002条逐条人工金标准。**','', '## 数据与定义','',f"完整官方TRAIN {n:,}条任务指令，{summary['unique_map_instruction']:,}个唯一map+instruction，{summary['maps']}张地图；未使用Val/Test构建统计。逐任务原文、图关系与匹配名保存在records.json/CSV。仅使用原文、公开CityRefer名称/类型及TRAIN目标类别/实例标注辅助确定分析对象；**没有用目标GT坐标或轮廓筛选地标**，没有模型训练或导航。",'', '“地标”在本报告指可绑定到该地图CityRefer目录的具名道路、建筑等公开对象。未具名的树、汽车、房屋、停车场是普通参照物；它们可以充当中间节点，但不自动等于可定位的公开地图锚点。', '', '- 直接：目标与具名地标有明确空间边，如building on St Peter’s Street；几何部位词side/front/corner不当成独立物体。直接叫出目标自身目录名另列identity。', '- 间接：目标→中间对象/区域→具名地标，例如车→停车场→One Stop，车→车排→停车场→Nike。路口或未知车道等中间构造也计入当前多跳定义；并非所有间接链都必须依赖视觉。', '- 无明确锚点：未发现可靠的公开名称绑定，且无明显待核对专名线索；常见只有颜色、车排、邻车、树木。它不等于绝对无法与地图关联，RGB/区域识别仍可能帮助绑定。', '- 不确定：名称缩写/错拼待核对，或关系作用域/共指未解析；不能将算法解析失败当成指令无关联。', '', '## 全量自动初筛（保守解析结果，非全量语义金标准）','', '|类型|条数|占全部TRAIN|','|---|---:|---:|',f'|直接几何关系|{counts["direct_geometry"]:,}|{counts["direct_geometry"]/n:.2%}|',f'|直接自名称绑定|{counts["direct_identity"]:,}|{counts["direct_identity"]/n:.3%}|',f'|间接关系链|{i:,}|{i/n:.2%}|',f'|未发现明确地图锚点|{counts["no_confirmed_anchor"]:,}|{counts["no_confirmed_anchor"]/n:.2%}|',f'|名称或关系作用域待确认|{counts["uncertain"]:,}|{counts["uncertain"]/n:.2%}|','',f"{summary['overlap']['explicit_public_name']:,}/{n:,}（{summary['overlap']['explicit_public_name']/n:.2%}）包含可字面绑定的公开名称。自动图中{overlap:,}条同时存在直接和间接路径，按最短可靠路径归到直接；两种特征不能直接相加。",'', '自动未判明比例较高，主要是复杂句、共指、拼写变体以及4跳路径截断；抽样中多数仍有明确地标关系。所以不能把23%左右的自动uncertain写成“23%指令根本无法关联地标”。', '', '## 当前会话语义复核及校正估计','', '解析器冻结后，以新seed分层抽取101条，我直接阅读原始英文、解析链与该图公开名称进行复核；没有调用Qwen或外部LLM API。复核不是人类标注，也不能保证零错误。20条自动uncertain中14条可归到间接、5条直接，1条仍保留不确定。20条无锚点样本中，有3条其实能通过名称变体建立关联，另1条疑似错拼仍未知。', '', '|复核校正后的类别|估计数量|估计比例|分层bootstrap95%区间|','|---|---:|---:|---|']
 names={'direct_including_identity':'直接关联（含自身名称）','indirect':'间接关联','no_explicit_reliable_anchor':'缺少明确可靠地图锚点','uncertain':'仍需确认'}
 for c,m in estimates.items():lines.append(f"|{names[c]}|约{m['estimated_tasks']:,.0f}|{m['estimated_percent']:.1f}%|{m['stratified_bootstrap95_percent'][0]:.1f}–{m['stratified_bootstrap95_percent'][1]:.1f}%|")
 lines.extend(['','区间只反映该分层样本的抽样波动，不包括本会话复核错误、解析设计偏差或其他地图分布偏差。建议将结论表述为“约一半直接、接近一半间接”，不把点估计当成精确全量人工统计。', '', '## 目标类别差异（全量自动初筛）','', '|目标类别|总数|直接含名称|间接|无锚点|待确认|','|---|---:|---:|---:|---:|---:|'])
 for c,v in summary['by_target_category'].items():
  total=sum(v.values());lines.append(f"|{c}|{total:,}|{v.get('direct_geometry',0)+v.get('direct_identity',0):,}|{v.get('indirect_geometry',0):,}|{v.get('no_confirmed_anchor',0):,}|{v.get('uncertain',0):,}|")
 lines.extend(['','车辆有5,205条已经解析出的间接关系，明显多于其3,144条直接关系；建筑则直接关系为主。不能直接把这些自动类别比例当作抽样校正后的类别内真实比例，因为抽样未按目标类别联合分层。', '', '## 真实例子',''])
 for ep,explanation in [(7797,'直接：building → on → St Peter’s Street。'),(1027,'间接：黑车 → 第四排/停车场 → One Stop；不能把Goal直接设为One Stop。'),(1911,'无明确具名地图锚点：只能先靠停车区布局和序数，再用观测识别区域。'),(6236,'名称变体：police custody suites与目录West Midlands Police Custody Suite可关联，字面失败不代表无关联。')]:
  r=next(x for x in records if x['episode']==ep);lines.extend([f"- Episode {ep}：`{r['instruction']}`",f'  {explanation}'])
 lines.extend(['','## 对原始GeomBelief的意义','', '1. “没有任何地标信息”不是这批TRAIN指令的主要共性；优先利用已有地标，避免不断扩展模型却忽略关系作用域。', '2. 直接关联不等于目标已经精确定位：on某条道路可能仍对应上百米范围；left/right/front/behind参考系不明确时，不能强制转换为世界方向。', '3. 间接任务尤其车辆需要保留“地标→区域→车排/邻近实例→目标外观”的层级。最值得先改的是语言角色/关系作用域和中间区域表示，而不是把停车场或地标中心直接当车辆目标。', '4. 此次只说明语言结构分布，不证明该模块一定带来SR收益，更不能替代导航失败归因。无名区域也不等于完全无法关联：可观察RGB可能将其绑定到地图位置。', '', '## 复现与边界','', '独立分析分支research/instruction-landmark-association-20261103，旧GraphNav主工作树及其未提交恢复修改未动。CPU spaCy依存解析（版本见summary.json）仅作结构提案，本会话复核仅101条；没有训练任何导航模型。9项基础解析、归一化和身份约束测试通过。所有名称关联均来自同地图公开目录，不使用GT邻近检索；没有检查实际地标距离或图像可见性，因而“附近”仅指指令语义参照，并非已验证50m/100m邻域。', '', '运行scripts.analyze_instruction_landmark_association生成全量图，scripts.report_instruction_landmark_association保存复核、估计、报告和HTML。模型环境独立保存在Experiments/instruction_audit_env，不修改原工程Python环境。'])
 (P/'REPORT_zh.md').write_text('\n'.join(lines))
 # Browsable examples; all 22k are available in CSV/JSON rather than huge DOM.
 page=['<meta charset="utf-8"><style>body{font:16px sans-serif;margin:24px}pre{white-space:pre-wrap}section{border-bottom:1px solid #ddd;padding:12px}</style><h1>TRAIN指令：公开地标直接/间接关联</h1><p>全量自动初筛＋本会话101条AI语义复核；不是人类金标准。关联≠20m定位成功。</p><a href="REPORT_zh.md">中文报告</a> · <a href="all_instructions.csv">全部22,002条CSV</a> · <a href="summary.json">自动统计</a> · <a href="review_adjusted_estimates.json">复核校正估计</a>']
 for r in review:page.extend([f'<section><b>Episode {r["episode"]}: {r["automatic_category"]} → {r["reviewed_category"]}</b>',f'<p>{html.escape(r["instruction"])}</p>',f'<p>{html.escape(r["reason"])}</p></section>'])
 (P/'index.html').write_text('\n'.join(page));print(json.dumps(corrected,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
