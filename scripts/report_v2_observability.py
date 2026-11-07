"""Coverage and sampling-size proxies, not an instance-recognition benchmark."""
import argparse,json,sys,html
from pathlib import Path
import numpy as np,cv2
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame


def main():
 p=argparse.ArgumentParser()
 for k in ['run','data','sbf-source','initial-eval','output']:p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);(a.output/'cases').mkdir(exist_ok=True);initial=json.load(open(a.run/'observability/initial_summary.json'));history=json.load(open(a.run/'observability/causal_history_summary.json'));assert initial['n']==history['n']==1911
 for scope in ['initial','causal_history']:
  (a.output/(scope+'_summary.json')).write_text((a.run/'observability'/(scope+'_summary.json')).read_text());(a.output/(scope+'_rows.json')).write_text((a.run/'observability'/(scope+'_rows.json')).read_text())
 keys=['center_in_footprint','geometric_overlap','valid_roi','clarity_proxy_8','clarity_proxy_16','clarity_proxy_32'];labels=['目标中心几何入镜','Contour几何交集','有有效像素且无明显配准冲突','8px/64像素尺寸代理','16px/256像素尺寸代理','32px/1024像素尺寸代理'];table='\n'.join(f'|{label}|{initial["initial"][k]} ({100*initial["initial"][k]/1911:.1f}%)|{history["any_legal_observation"][k]} ({100*history["any_legal_observation"][k]/1911:.1f}%)|' for k,label in zip(keys,labels));cat='\n'.join(f'|{c}|{v["n"]}|{initial["by_category"][c]["initial"]["clarity_proxy_16"]}|{v["any_legal_observation"]["clarity_proxy_16"]}|' for c,v in history['by_category'].items())
 text=f'''# 1911条NMS Top-5命中任务的真实观测支持\n\n这是完整Val-Seen中原始V2空间去重Top-5世界20m命中的固定1911条。GT身份/位置/Contour仅用于事后量测，不参与视点或候选选择。\n\n|指标|真实起点|原始V2合法历史累计（含起点）|\n|---|---:|---:|\n{table}\n\n|目标类别|任务数|起点16px代理|历史16px代理|\n|---|---:|---:|---:|\n{cat}\n\n所有原始观测均为384×384；阈值8/16/32表示有效投影区域BBox短边至少k像素、面积至少k²像素、该投影Mask内有效RGB/Depth比例至少80%。不代表物体全貌清晰或语义可辨认。3条目标中心与Contour距离超过20m，不计为可靠ROI，保留异常。\n\n**真正“足够清晰且能确认正确实例”的数量仍未知**：没有独立像素可见性/实例标注；树木遮挡、地面Footprint/屋顶差异、粗Contour、多实例歧义仍可能使代理高估。这里不测模型识别准确率，不使用插值放大冒充细节。几何上不入镜不直接算检测漏检。\n\n历史限定为本轮原始V2实际闭环的已观测决策Pose，不是教师轨迹、规划地标视点或GT选择视点。对Horizon/停滞最后移动后尚未观测的Pose明确排除。此处反映该控制器走过的观测机会，不是所有可能合法探索轨迹的上界。训练后V2的不同轨迹未混入这1911条统计。\n\n世界20m候选命中≠候选ROI真的覆盖目标，后续候选裁剪尺寸和实际视觉匹配还需独立验证；本报告只回答当前/历史图像是否提供几何与像素支持。\n'''
 hr=json.load(open(a.run/'observability/causal_history_rows.json'));outcomes={}
 for state in ['success','failure']:
  rr=[r for r in hr if bool(json.load(open(a.run/'original_v2/rows'/f'{r["index"]:05d}.json'))['metrics']['SR'])==(state=='success')]
  outcomes[state]=dict(n=len(rr),historical_16=sum(any(f['clarity_proxy_16'] for f in r['frames']) for r in rr))
 (a.output/'by_navigation_outcome.json').write_text(json.dumps(outcomes,indent=2))
 text+=f'\n导航结果分组：1911条中成功{outcomes["success"]["n"]}、失败{outcomes["failure"]["n"]}；失败中{outcomes["failure"]["historical_16"]}条曾满足历史16px代理。该数仅提示待核验视觉纠错样本，不是可恢复SR上界，不证明实例可辨认。\n'
 (a.output/'REPORT_zh.md').write_text(text)
 sys.path.insert(0,str(a.sbf_source));from sbfnav.dataset import CityNavDataset
 from sbfnav.mapdata import GROUND_LEVEL
 ds=CityNavDataset(a.data,'val_seen',version='processed_citynav');objects=json.load(open(a.data/'cityrefer/objects.json'));rows=json.load(open(a.run/'observability/initial_rows.json'));historyrows={r['index']:r for r in json.load(open(a.run/'observability/causal_history_rows.json'))};renderer=CityNavObservedViewRenderer(a.data/'rgbd',a.output/'observations',image_size=384);choices=[]
 for pred in [lambda r:r['category']=='Car' and r['frames'][0]['clarity_proxy_16'],lambda r:r['category']=='Car' and r['frames'][0]['valid_roi'] and not r['frames'][0]['clarity_proxy_8'],lambda r:r['category']=='Building' and r['frames'][0]['clarity_proxy_32'],lambda r:not r['frames'][0]['geometric_overlap'] and any(f['clarity_proxy_16'] for f in historyrows[r['index']]['frames'])]:choices.extend([r for r in rows if pred(r)][:2])
 cards=[]
 for r in {r['index']:r for r in choices}.values():
  i=r['index'];ep=ds[i];nav=json.load(open(a.run/'original_v2/rows'/f'{i:05d}.json'));best=max(historyrows[i]['frames'],key=lambda f:f['pixel_area']);ims=[]
  for label,step in [('start',0),('history',best['step'])]:
   obs=renderer.current_rgbd(ep.map_name,np.array(nav['poses'][step]),GROUND_LEVEL[ep.map_name]);raw=obs['bgr'];cv2.imwrite(str(a.output/'cases'/f'{i}_{label}_raw.png'),raw);vis=raw.copy();contour=np.array(objects[ep.map_name][str(ep.object_id)]['contour']);pix=world_to_frame(contour,obs['footprint_world'],raw.shape);cv2.polylines(vis,[np.rint(pix).clip(-100000,100000).astype(np.int32)],True,(0,180,255),1);cv2.putText(vis,f'{label} step={step} CONTOUR PROXY',(5,18),cv2.FONT_HERSHEY_SIMPLEX,.43,(0,180,255),1);fn=f'cases/{i}_{label}_proxy.png';cv2.imwrite(str(a.output/fn),vis);ims.append(f'<a href="cases/{i}_{label}_raw.png"><img src="{fn}" width="384"></a>')
  cards.append(f'<article><h3>{i} {r["category"]}: {html.escape(ep.instruction)}</h3>'+''.join(ims)+'<p>橙线是未核验Contour投影，点击查看原始384px；历史时刻按事后最大支持面积展示，不用于推理/视点选择。</p></article>')
 (a.output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>候选观测支持代理</title><h1>1911任务真实观测审计</h1><a href="REPORT_zh.md">完整报告</a><p>图中标记不是人工像素GT，不宣称可见性/识别准确率。</p>'+''.join(cards));print(text)
if __name__=='__main__':main()
