"""Re-audit the four original candidate contradictions using raw TRAIN evidence."""
import argparse,gzip,json,collections
from pathlib import Path
import numpy as np
from shapely.geometry import Point
from shapely.ops import nearest_points
from text_relations import TextCatalog,Entity,name_key
from geometry_frames import poly,shape_axis,angle,adiff
from extract import dump
BASE=Path(__file__).parent
NOTES=[
 {'scope':'两条都是房屋相对于具名教堂的方向，含第二/两个房屋的沿街计数。不是车—车关系。','local_frame':'连排房屋与Aston Lane局部排列可能影响计数方向；道路没有独立的行进正向或观察者位置。','image_check':'真实正射图显示目标为教堂旁的灰屋顶连排房屋部分，目标与教堂轮廓分离；图像中目标位于教堂的−X侧。','viewpoint':'原始TRAIN描述没有给出这两条的描述相机姿态。教堂朝街或观察者面朝教堂的左右可相反，但不能据目标方向挑选。','verdict':'绑定与方向修饰对象可确认；共享同一有向世界参考系时不兼容。局部计数/视角是否不同不可确定，不能确认真实标注矛盾。'},
 {'scope':'两条主语均为同一蓝车，参照为唯一Camden House；第一条的其他left/right描述邻车，已排除在本实验之外。','local_frame':'场地含草坪、停车区及相邻建筑，front/behind可能描述立面、场地或视点；没有可靠入口/正面朝向记录。','image_check':'蓝车、停车区与Camden House完整轮廓可见；蓝车位于参照位置−X/+Y侧。原图没有独立确认建筑正面。','viewpoint':'同目标另有TRAIN描述“car parked facing the Camden House”，它给出车朝向与地标的联系，但不能证明两条front/behind使用相同视角。','verdict':'同一对象—地标的表面正反用词成立；未发现同名误绑。真实语义参考系未固定，尚不能确认标注矛盾。'},
 {'scope':'两条主语均为同一棕屋顶建筑，参照为唯一Pembroke College Boat Club。','local_frame':'同目标的另一条TRAIN描述明确指定“stand at Pembroke College Boat Club facing the stream ... on your right”；这是独立的文本视角证据。','image_check':'目标与船屋在河岸附近并列，目标位于船屋−X侧；唯一Water对象可独立提供面向水面的候选方向。','viewpoint':'面向河流候选轴只由船屋位置和唯一Water轮廓的最近点构成，不使用目标GT选择轴或符号；它不是恢复的原始相机yaw。','verdict':'不应当把这组直接认定为真实标注矛盾；存在已明说的河流面向参考系，能检验视角解释，但不能把第2条的视角自动赋给第4/5条。'},
 {'scope':'front与at the back均直接修饰蓝车相对于Broadmeadows建筑的关系；邻车left不纳入主统计。','local_frame':'建筑北侧（仅指图像上侧，不声称地理北）停车区与建筑固有front/back可能不同；建筑轮廓主轴没有正反。','image_check':'同一蓝车、黑车和红车排列及Broadmeadows轮廓可见，蓝车位于建筑位置−X/+Y侧。','viewpoint':'未找到入口、立面正向或两条描述的相机记录，不能凭车辆位置定义建筑front。','verdict':'唯一绑定可确认；这组是未确定参考系的正反用词候选，不是已证实的同参考系真实标注矛盾。'}]

def run(root,out):
 old=json.loads((BASE.parent/'direction_frames/results/same_pair_opposite_descriptions.json').read_text());O=json.loads((root/'cityrefer_original/objects.json').read_text());E=json.loads((root/'original_citynav/citynav_train_seen.json').read_text());rows=[json.loads(x) for x in gzip.open(out/'text_relations.jsonl.gz','rt')];idx={r['occurrence_id']:r for r in rows};epindex=collections.defaultdict(list)
 for i,e in enumerate(E):epindex[(f"{e['area']}_block_{e['block']}",str(e['object_ids'][0]))].append((i,e))
 result=[]
 for j,c in enumerate(old):
  m=c['map'];obs=O[m];t=obs[c['target_id']];ref=obs[c['ref_id']];same=[i for i,o in obs.items() if o['name'] and name_key(o['name'])==name_key(ref['name'])];assert same==[c['ref_id']]
  cat=TextCatalog([Entity(str(i),o['name'],o['object_type']) for i,o in obs.items() if o['name']]);annotations=[]
  for i,e in epindex[m,c['target_id']]:
   a=e['ann_ids'][0];txt=e['descriptions'][0];p=e['trajectory'][0];yaw=angle(p[3:5]);rs=[r for r in rows if r['episode']==i]
   annotations.append({'episode':i,'ann_id':a,'instruction':txt,'aligned_original_annotation':' '.join(txt.lower().split())==' '.join(t['descriptions'][a].lower().split()),'initial_yaw_deg':yaw,'initial_pose':p,'marker_xyz':e['marker_positions'][0],'gt_xyz':e['target_positions'][0],'landmark_relations':[{'word':r['word'],'status':r['status'],'reference_ids':r['reference_ids'],'subject_role':r['subject_role'],'parse_confidence':r['parse_confidence'],'reason':r['reason']} for r in rs]})
  focus=[]
  for rid,txt in [(c['first'],c['instruction1']),(c['second'],c['instruction2'])]:
   hits=[h for h in cat.spans(txt) if c['ref_id'] in h['ids']];assert len(hits)==1 and hits[0]['ids']==[c['ref_id']]
   r=idx.get(rid);assert r is not None
   focus.append({'occurrence_id':rid,'instruction':txt,'unique_reference_ids':hits[0]['ids'],'v2_status':r['status'],'v2_reason':r['reason'],'v2_confidence':r['parse_confidence'],'word':r['word']})
  d=np.array(t['position'][:2])-np.array(ref['position'][:2]);tp,rp=poly(t),poly(ref)
  x={**c,'case_number':j+1,'notes':NOTES[j],'same_named_reference_ids':same,'target_object':t,'reference_object':ref,'source_annotations_TRAIN_only':annotations,'focus_relations':focus,'delta_xy':d.tolist(),'world_angle_deg':angle(d),'target_polygon_valid':tp is not None,'reference_polygon_valid':rp is not None,'polygon_distance_m':tp.distance(rp) if tp and rp else None,'reference_unsigned_axis':shape_axis(rp),'true_annotation_contradiction_confirmed':False,'shared_reference_frame_verified':False,'original_annotator_image_available':False,'available_image':'actual orthographic TRAIN map raster, not annotation-camera capture'}
  if j==2:
   water=[(i,o,poly(o)) for i,o in obs.items() if o['object_type']=='Water'];x['water_candidate_ids']=[i for i,o,p in water]
   if len(water)==1 and water[0][2] is not None:
    rxy=np.array(ref['position'][:2]);q=nearest_points(Point(*rxy),water[0][2])[1];h=angle(np.array(q.coords[0])-rxy);x['independent_river_facing']={'water_id':water[0][0],'water_polygon':water[0][1]['contour'],'water_boundary_point':list(q.coords[0]),'heading_deg':h,'right_expected_world_deg':(h-90)%360,'right_error_deg':float(adiff(angle(d),h-90)),'uses_target_GT_to_choose_heading':False,'source':'reference -> nearest boundary of unique Water polygon; stream-facing direction is a geometry proxy, not recovered camera pose'}
  result.append(x)
 dump(out/'four_case_reaudit.json',result);print('four cases re-audited; confirmed same-frame annotation contradictions = 0')
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--data-root',required=True,type=Path);a=ap.parse_args();run(a.data_root,BASE/'results')
