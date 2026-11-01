"""Summarize official-pair adaptation with grouped new-head audit statistics."""
import json,sys,hashlib
from pathlib import Path
import numpy as np
from graphnav.workspace_paths import PROJECT


def main():
 folders={'official':Path('aerialvg_pair_pilot_20261101'),'plain':Path('aerialvg_plain_pilot_20261101')}
 for path in folders.values():assert (path/'complete.json').exists(),str(path)
 split=[json.loads((p/'audit_split.json').read_text()) for p in folders.values()]
 assert split[0]['audit_episodes']==split[1]['audit_episodes']
 sys.path.insert(0,str(PROJECT/'Baselines/SBFNav/runs/sbfnav_main_e20_bs16_s0_20260929/source'))
 from sbfnav.dataset import CityNavDataset
 source=CityNavDataset(PROJECT.parent/'DATA/sbfnav-data','train_seen',version='processed_citynav')
 manifest=json.loads((PROJECT/'GraphNav/data/rebuilt_stage1_20261025/frames/manifest.json').read_text())
 episodes=manifest['protocol']['episode_indices']
 group_cache={}
 def cluster(index):
  episode=episodes[index//4]
  if episode not in group_cache:
   record=source[episode];group_cache[episode]=hashlib.sha256(json.dumps((record.map_name,record.instruction)).encode()).hexdigest()
  return group_cache[episode]
 def read(folder,control,audit):
  return json.loads((folder/('predictions_'+('audit_' if audit else '')+control+'.json')).read_text())
 def compare(left,right):
  assert left['dev_indices']==right['dev_indices']
  difference=(np.array(left['errors_m'])<=20).astype(float)-(np.array(right['errors_m'])<=20)
  ids=np.array([cluster(i) for i in left['dev_indices']]);_,inverse=np.unique(ids,return_inverse=True)
  sums=np.bincount(inverse,weights=difference);counts=np.bincount(inverse);rng=np.random.default_rng(0);draws=[]
  for _ in range(50):
   sampled=rng.integers(0,len(sums),size=(100,len(sums)))
   draws.extend((100*sums[sampled].sum(1)/counts[sampled].sum(1)).tolist())
  return dict(difference_pp=100*difference.mean(),group_bootstrap_95ci_pp=np.quantile(draws,[.025,.975]).tolist(),helped=int((difference>0).sum()),harmed=int((difference<0).sum()),groups=len(sums),states=len(difference))
 results={};lines=['# 官方 AerialVG 关系模块导航适配短训练\n','本轮直接复用官方 `CrossSelfRelationTransformer` 源码及预训练权重；冻结该模块与原导航主干，只训练投影、零初始化门及原视觉残差头。对照使用相同初始化、损失、2轮×128训练batch、seed0、batch16及分组隔离。\n']
 for audit in (False,True):
  name='audit' if audit else 'dev';filename='audit_controls.json' if audit else 'controls.json'
  metrics={tag:json.loads((folder/filename).read_text()) for tag,folder in folders.items()}
  comparisons={label:compare(read(folders['official'],'real',audit),read(folder,control,audit)) for label,folder,control in [('vs_plain',folders['plain'],'real'),('vs_no_relation',folders['official'],'no_relation'),('vs_zero',folders['official'],'zero'),('vs_shuffle',folders['official'],'shuffle')]}
  results[name]=dict(metrics=metrics,comparisons=comparisons)
  lines.extend([f'\n## {name}\n','\n|模型/条件|状态数|R@1(20m)|NE(m)|\n|---|---:|---:|---:|\n'])
  for tag,controls in metrics.items():
   for control,m in controls.items():lines.append(f"|{tag}/{control}|{m['samples']}|{100*m['corrected_R1']:.4f}%|{m['corrected_NE']:.4f}|\n")
  lines.append('\n配对差值，按map+instruction组做5,000次bootstrap：\n\n')
  for label,m in comparisons.items():lines.append(f"- {label}: {m['difference_pp']:+.4f}pp，95%CI {m['group_bootstrap_95ci_pp']}，帮助{m['helped']} / 损害{m['harmed']}状态。\n")
 lines.append('\n## 解释边界\n\nDev参与checkpoint选择，其区间是探索性描述。audit按map+instruction从非Dev Train分离，未参与本轮新残差头训练或选择；原冻结主干及过去实验曾使用Train，因此它不是整个系统的独立未知场景测试。所有指标为离线前缀状态，不是闭环SR。未引入层级空间图HCA，也未声称复现官方未发布的训练代码。\n')
 Path('AERIALVG_PAIR_ADAPTATION_RESULTS_20261101.md').write_text(''.join(lines))
 Path('aerialvg_pair_adaptation_summary_20261101.json').write_text(json.dumps(results,indent=2))
 print(json.dumps({k:v['comparisons'] for k,v in results.items()},indent=2))


if __name__=='__main__':main()
