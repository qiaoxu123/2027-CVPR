"""Shared schema/utilities; public inference inputs are explicitly whitelisted."""
from pathlib import Path
import collections,gzip,hashlib,json,re,sys
import numpy as np
V2=Path(__file__).resolve().parents[1]/'direction_frames_v2'
sys.path.insert(0,str(V2))
from text_relations import Entity,TextCatalog,name_key,parse_landmark_relations,TOKEN,PATTERNS
from geometry_frames import poly,shape_axis,angle,adiff,vertices,OFFSETS,IMAGE,transforms
BASE=Path(__file__).parent;SEED=20261106;MAIN={'left','right','front','behind'}
def readj(path):return json.loads(Path(path).read_text())
def writej(path,data):Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def readlines(path):
 with gzip.open(path,'rt',encoding='utf8') as fp:return [json.loads(x) for x in fp]
def writelines(path,rows):
 with gzip.open(path,'wt',encoding='utf8') as fp:
  for r in rows:fp.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def fold_id(cluster):return int(hashlib.sha256(cluster.encode()).hexdigest()[:8],16)%5
def cluster_ci(rows,values,B=1000):
 import pandas as pd
 if not len(rows):return [None,None]
 df=pd.DataFrame({'g':[r['cluster'] for r in rows],'v':values});g=df.groupby('g').v.agg(['sum','count']);rng=np.random.default_rng(SEED);ix=rng.integers(0,len(g),(B,len(g)));a=g['sum'].values[ix].sum(1)/g['count'].values[ix].sum(1);return np.quantile(a,[.025,.975]).tolist()
def text_key(s):return ' '.join(s.lower().split())
ROAD_SUFFIX={'road','street','lane','drive','way','avenue','close','terrace','walk','path','gardens','causeway','roundabout','place'}
class TypedRoadCatalog(TextCatalog):
 """Semantic type disambiguation only. No positions/GT are available here."""
 def spans(self,text):
  hits=super().spans(text)
  for h in hits:
   trailing=text[h['end']:h['end']+15]
   h['explicit_road_type']=h['name'].split()[-1] in ROAD_SUFFIX or bool(re.match(r'\s+road\b',trailing,re.I))
  return hits
 def resolve(self,hit):
  ids=hit['ids'];road=[i for i in ids if self.entities[i].category=='TrafficRoad']
  if hit.get('explicit_road_type') and road:
   if len(road)==1:return road,'semantic_road_type_unique',.97
   return road,'semantic_road_type_all_segments',.85
  return super().resolve(hit)
