"""Text-reviewed fixed cohort selection. This tool never opens target labels."""
import argparse,json,hashlib
from pathlib import Path

# Decisions follow a full reading of all 154 public instructions, before this
# experiment's predictions and scoring. Not a new automatic language parser.
EXCLUDE={
4:'目标→第9排停车位→地标，间接链',5:'目标→第14排停车位→地标，间接链',6:'目标→工地→地标，间接链',8:'left side/三栋建筑组与north的作用域不清',11:'south可能修饰白色建筑或车辆，作用域不清',12:'停车场/两建筑/车辆的south作用域不清',15:'道路far north端点语义与侧向关系混合',16:'facing朝向且显式road-as-horizontal版面参考',17:'道路far north端点位置，不是明确侧向关系',18:'目标→停车场→地标，间接链',19:'目标→停车场→地标，间接链',23:'目标→九车排→地标，间接链',24:'目标→九车排→地标，间接链',25:'目标→九车排→地标，间接链',26:'同名The Coppice混合Building和TrafficRoad，绑定未消歧',27:'Perry Barr footpath词组边界与目录名称需核验',35:'south参照Fitzwilliam Museum而非当前绑定Grove Lodge Cottage',40:'道路/车/Car Park并列导致north作用域不清',59:'目标→四块绿地集合→Hall，间接链',62:'道路far south端点语义',75:'道路far south端点并附房屋参照，非明确侧向关系',93:'目标→对面建筑→Henry Wellcome，间接链',100:'when viewed aerially with Geography...指定版面参考，方向未知',104:'这些车辆集合/当前车辆作用域需核验',116:'车辆集合和建筑的共同south关系，严格组暂不纳入',130:'far south on the left side，缺乏明确target south of地标谓词',135:'目标→大建筑局部→路口north side，间接链',149:'目标→建筑集合/十字空地→道路，间接链',152:'north of可能修饰Abbey Street而非目标，作用域不清'}

def decide(row,public_types=()):
 if row['index'] in EXCLUDE:return False,EXCLUDE[row['index']]
 if len(set(public_types))>1:return False,'同名公开片段类型混合，未消歧：'+','.join(sorted(set(public_types)))
 if row['word'] not in ('north','south','east','west'):return False,'相对方向参考未知'
 return True,'逐条原文核验：目标主语/明确回指或前置位置短语直接连接具名地标；无中间关系'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=json.loads(a.source.read_text());catalog=json.loads(a.catalog.read_text());selected=[];decisions=[]
 for r in rows:
  assert not {'target_world','target_ids','target_type'}&set(r)
  types=[catalog[r['map_name']][i]['object_type'] for i in r['public_object_ids']];keep,reason=decide(r,types);decisions.append(dict(index=r['index'],episode=r['episode'],subset=r['subset'],public_types=types,instruction=r['instruction'],word=r['word'],landmark=r['landmark_name'],keep=keep,reason=reason,reviewer='current assistant textual review; not human gold'))
  if keep:selected.append(r)
 (a.output/'selection.json').write_text(json.dumps(decisions,ensure_ascii=False,indent=2));(a.output/'public.json').write_text(json.dumps(selected,ensure_ascii=False,indent=2));(a.output/'selection_seal.json').write_text(json.dumps(dict(source_sha256=digest(a.source),catalog_sha256=digest(a.catalog),public_sha256=digest(a.output/'public.json'),selection_sha256=digest(a.output/'selection.json'),target_labels_read=False,selection_basis='Original instruction only; inherited public binding. No target direction/GT-distance/outcome selection',scope='Existing 154 TRAIN pair candidate pool, not exhaustive 22002-instruction retrieval',before_current_prediction=True),indent=2));print('Selected',len(selected),'of',len(rows))
if __name__=='__main__':main()
