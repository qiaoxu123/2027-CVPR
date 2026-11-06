"""Text-only Agent review: not independent human annotation or GT role truth."""
import json
from pathlib import Path
RUN=Path('/home/20261024_1/Workspace/2027-CVPR/GraphNav/runs/relation_bearing_s42_20261106')
notes={11498:'目标主语及opposite链合理；另一具名参考未完整覆盖。',21452:'on Priory Road直接链合理，复杂车辆—路口链仅部分恢复。',21653:'on Abbey Road直接关系合理。',6769:'on Aldridge Road合理；同类房屋的left实例约束仍只在全文。',16228:'raw direct分类不可靠；可信Clarendon链经黑车为间接，Victoria裸to连接不确定。',16967:'corner of City Road合理；另一道路分支不完整。',15871:'到Parkside Gym直接关系合理；Parkside road与Parkside的associated分支有名词作用域歧义。',3630:'Thornbury直接链合理；车序数/卡车约束尚未完全结构化。',9300:'两个直接参考关系可保留；behind坐标系未定义。',7787:'两道路corner关系合理，但角点需多约束联合。',17965:'median中间场景可用；前车/白车约束不完整，属于粗间接链。',6182:'car→parking lot→across from WDM链合理；序数和van关系尚未完整保留为路径。',15094:'lot→grey buildings→Hills Road链合理；其他方向描述不能当成目标方向真值。',909:'car→row→Warehouse路径可作为间接关联，停车场层次被压缩；序数仍靠全文。',8864:'原文多重between且语法含混，抽取链不作为高可靠实例作用域真值。',19082:'car→parking lot→Parsonage Street链合理。',15090:'总体间接合理；black triangle是建筑形状片段，不能直接假设独立实例标签。',5804:'有间接关联，但路径经另一黑车并遗漏parking lot，作用域存在疑点。',3264:'target复杂住宅与section范围难确定，不强制称精确目标解析。',18936:'多层路径且存在not，完整全文保留，但当前路径简写对否定与角色存在局限。'}
sample=json.load(open(RUN/'data/parse_audit_sample.json'));out=[]
for q in sample:
 if q['index'] in notes:out.append(dict(index=q['index'],instruction=q['instruction'],parsed_category=q['category'],target=q['target_span'],review=notes[q['index']],reviewer='current Agent, language-only inspection',GT_used=False))
Path('reports/relation_bearing').mkdir(exist_ok=True)
Path('reports/relation_bearing/text_audit.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(len(out),'reviewed, no GT used')
