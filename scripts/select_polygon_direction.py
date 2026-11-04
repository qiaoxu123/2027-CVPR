"""Explicit current-assistant scope review of the complete frozen candidate pool.

No GT coordinates, model predictions or error metrics are inputs. Public
linear map fragments may share types Road/Footpath/Bridge; mixed building/road
identities are not silently merged into a reliable landmark.
"""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter
from scripts.prepare_strict_direct_direction import EXCLUDE as OLD_EXCLUDE

# Candidate IDs refer to the complete 427-entry full-TRAIN scan, reviewed as
# original sentences. This is an auditable cohort annotation, not a general NLP model.
CHAIN={15,22,26,33,38,39,41,42,46,55,56,64,66,67,76,77,78,79,80,85,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,106,107,108,109,110,111,112,113,114,115,125,141,160,161,173,175,185,206,208,210,214,215,216,218,220,221,222,265,283,302,303,306,307,308,328,331,355,375,376,404,424}
ROAD_SCOPE={0,2,3,18,25,32,35,40,178,313,332,338,397,402,425}
PART_OR_GROUP={5,37,82,116,163,287,291,333,362,363,370,382,384,389,415}
BINDING={187:'Fitzwilliam Museum road被绑定到Museum建筑，名称/类别不匹配',231:'Fitzwilliam Museum Department Department具体实体未核验'}
SCOPE={49:'道路far north端点非明确侧向',253:'south of修饰green tree而非目标',271:'south of可能修饰停车场、邻楼或车辆'}

def decide_candidate(r,old_reasons):
 key=(r['episode'],r['word'],r['landmark_name'])
 if key in old_reasons:return False,'此前原文作用域审查：'+old_reasons[key]
 cid=r['candidate_id']
 if cid in CHAIN:return False,'目标经停车场/车排/道路/建筑或集合建立关系，非严格直接关系'
 if cid in ROAD_SCOPE:return False,'方向可能修饰另一条道路或参照实体，目标关系作用域不确定'
 if cid in PART_OR_GROUP:return False,'多个目标/建筑局部与整体的标签作用域未消歧'
 if cid in BINDING:return False,BINDING[cid]
 if cid in SCOPE:return False,SCOPE[cid]
 types=set(r['public_types']);linear={'TrafficRoad','Footpath','Bridge'}
 if len(types)>1 and not types<=linear:return False,'同名公开片段跨建筑/道路等实体类别，未消歧'
 if r['name_binding'].startswith('unique'):
  return True,'当前AI原文复核：唯一公开名称前缀简称，目标直接关联；保留简称来源，非人工金标准'
 return True,'当前AI逐句复核：目标/明确回指或独立位置前置语直接关联公开地标；线性地标全部合法片段保留'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);p.add_argument('--old-public',type=Path,required=True);a=p.parse_args();o=a.directory;candidates=json.loads((o/'candidates.json').read_text());assert len(candidates)==427
 old=json.loads(a.old_public.read_text());old_reasons={(r['episode'],r['word'],r['landmark_name']):OLD_EXCLUDE[r['index']] for r in old if r['index'] in OLD_EXCLUDE};selected=[];decisions=[]
 for r in candidates:
  keep,reason=decide_candidate(r,old_reasons);decisions.append(dict(**r,keep=keep,reason=reason,reviewer='current assistant; original text and public catalog only; not human validation'))
  if keep:selected.append(r)
 for name,data in [('decisions.json',decisions),('selected_text.json',selected)]: (o/name).write_text(json.dumps(data,ensure_ascii=False,indent=2))
 proof=dict(candidates=len(candidates),selected_pairs=len(selected),selected_episodes=len({r['episode'] for r in selected}),counts=dict(Counter(r['word'] for r in selected)),target_coordinates_used=False,source_sha256={str(p):sha(p) for p in [o/'candidates.json',o/'selected_text.json',o/'decisions.json',Path(__file__)]})
 (o/'selection_seal.json').write_text(json.dumps(proof,indent=2));print(json.dumps(proof,indent=2))
if __name__=='__main__':main()
