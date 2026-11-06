"""Frozen-angle experiment report: TRAIN Oracle separate from predicted-only SR."""
import argparse,csv,html,json,shutil
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256


def main():
 p=argparse.ArgumentParser()
 for k in ('run','baseline-evaluation','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();r=a.run;out=a.output;out.mkdir(parents=True,exist_ok=True)
 train=json.loads((r/'full/report.json').read_text());manifest=train['manifest'];hist=json.loads((r/'full/history.json').read_text());frozen=json.loads((r/'FROZEN_BEFORE_VAL.json').read_text())
 assert file_sha256(r/'full/last.pt')==frozen['angle_head_sha256']
 for path,sha in frozen['code'].items():assert file_sha256(path)==sha,f'Frozen inference code changed: {path}'
 nav=r/'val_seen_predicted_angles';metric=json.loads((nav/'metrics.json').read_text());base_metric=json.loads((a.baseline_evaluation/'metrics.json').read_text());assert metric['coverage']['completed']==metric['coverage']['expected']==2470
 current=json.loads((nav/'full_relation_manifest.json').read_text());old=json.loads((a.baseline_evaluation/'full_relation_manifest.json').read_text())
 for key in ('stage1','original_v2','trained_relation','split','data_root','relation_control','relation_visual_control','geometry_source','model_code','relation_runtime_code','config','public_catalog','text_adapter','text_inputs','text_parser'):assert current[key]==old[key],f'Baseline source mismatch {key}'
 assert current['angle_checkpoint']==frozen['angle_head_sha256'];assert current['trained_relation']==manifest['baseline_sha256']
 rows={x['index']:x for x in map(json.loads,(nav/'predictions.jsonl').read_text().splitlines())};baseline={x['index']:x for x in map(json.loads,(a.baseline_evaluation/'predictions.jsonl').read_text().splitlines())};assert len(rows)==len(baseline)==2470 and set(rows)==set(baseline)
 paired=[];stats=Counter();groups=defaultdict(list);delta_max=[]
 for i in sorted(rows):
  b=baseline[i];n=rows[i];assert b['episode_id']==n['episode_id'] and b['poses'][0]==n['poses'][0]
  br=json.loads((a.baseline_evaluation/f'belief_steps/{i:05d}/public_record.json').read_text());nr=json.loads((nav/f'belief_steps/{i:05d}/public_record.json').read_text());assert br['landmarks']==nr['landmarks'] and br['instruction']==nr['instruction'] and br['world_bounds']==nr['world_bounds']
  diag=n['step_diagnostics'];assert all(d['angle_GT_used_at_inference'] is False and d['modality']=='predicted_angular_direction_only' for d in diag)
  stats['fixes']+=int(not b['metrics']['SR'] and n['metrics']['SR']);stats['damage']+=int(b['metrics']['SR'] and not n['metrics']['SR']);stats['inference_error']+=int(n['termination']=='inference_error');stats['OSR_success_SR_failed']+=int(n['metrics']['OSR'] and not n['metrics']['SR']);stats[n['termination']]+=1
  if diag:
   stats['angle_supported_episodes']+=int(diag[0]['angle_supported']);stats['angle_changed_initial_Top1']+=int(diag[0]['baseline_top1']!=diag[0]['refined_top1']);delta_max.append(diag[0]['angle_delta_abs_max'])
  paired.append(dict(index=i,episode_id=json.dumps(n['episode_id']),baseline_SR=b['metrics']['SR'],predicted_angle_SR=n['metrics']['SR'],baseline_NE=b['metrics']['NE'],predicted_angle_NE=n['metrics']['NE'],termination=n['termination']))
  groups[tuple(n['episode_id'][:2])].append(i)
 stats['net']=stats['fixes']-stats['damage'];rng=np.random.default_rng(42);g=list(groups.values());ci=[]
 for _ in range(2000):
  ix=[i for j in rng.integers(0,len(g),len(g)) for i in g[j]];ci.append(np.mean([rows[i]['metrics']['SR']-baseline[i]['metrics']['SR'] for i in ix]))
 summary=dict(TRAIN_only=train['TRAIN_Birmingham_diagnostic'],Val_Seen=dict(baseline=base_metric['metrics'],predicted_angle=metric['metrics']),paired=dict(stats),SR_delta_physical_target_bootstrap95CI=np.quantile(ci,[.025,.975]).tolist(),physical_target_groups=len(g),single_seed=True,angle_delta_max=max(delta_max) if delta_max else None,angle_delta_median=float(np.median(delta_max)) if delta_max else None,head_checkpoint_SHA256=frozen['angle_head_sha256'],Oracle_navigation_executed=False)
 atomic_json(summary,out/'summary.json')
 with (out/'paired2470.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
 for source,name in [(r/'full/report.json','TRAIN_report.json'),(r/'full/history.json','history.json'),(r/'FROZEN_BEFORE_VAL.json','FROZEN_BEFORE_VAL.json'),(nav/'metrics.json','navigation_metrics.json'),(nav/'episodes.json','episodes2470.json'),(r/'CPU_tests.txt','CPU_tests.txt'),(r/'full/gradient_audit.json','gradient_audit.json'),(r/'runtime_smoke/smoke.json','runtime_smoke.json')]:shutil.copyfile(source,out/name)
 fig,axs=plt.subplots(1,2,figsize=(8,3));axs[0].plot([x['epoch'] for x in hist],[x['loss'] for x in hist],'o-');axs[0].set(xlabel='Epoch',ylabel='TRAIN combined loss');axs[1].plot([x['epoch'] for x in hist],[x['learned_gain'] for x in hist],'o-');axs[1].set(xlabel='Epoch',ylabel='Learned angular fusion gain');fig.tight_layout();fig.savefig(out/'training_curve.png',dpi=150);plt.close(fig)
 fig,axs=plt.subplots(1,2,figsize=(8,3));labels=['Baseline','Predicted angle']
 for ax,k in zip(axs,['SR','NE']):ax.bar(labels,[summary['Val_Seen'][m][k]*(100 if k=='SR' else 1) for m in ('baseline','predicted_angle')]);ax.set_ylabel(k+(' (%)' if k=='SR' else ' (m)'))
 fig.tight_layout();fig.savefig(out/'navigation.png',dpi=150);plt.close(fig)
 tr=summary['TRAIN_only'];methods=[('原全文冻结基线',tr['baseline']),('预测8方向',tr['predicted']),('Oracle8方向（仅TRAIN离线）',tr['oracle_GT_direction_OFFLINE_ONLY'])]
 offline_table='\n'.join(f'|{name}|{m["R20"]*100:.3f}%|{m["NE_m"]:.4f}|' for name,m in methods)
 nav_table='\n'.join(f'|{name}|{m["SR"]*100:.3f}%|{m["OSR"]*100:.3f}%|{m["SPL"]:.4f}|{m["NE"]:.4f}|{m["nDTW"]:.4f}|' for name,m in summary['Val_Seen'].items())
 steps=[json.loads(x) for x in (r/'full/steps.jsonl').read_text().splitlines()];stepmean=float(np.mean([s['step_ms'] for s in steps[20:]]))
 coverage=json.loads((out/'data_coverage.json').read_text())
 text=f'''# PR25：8方向方位角预测与Belief融合

## 设置与输入安全

全部21878条Revised TRAIN、seed42、3轮，冻结原Unified V2 Geometry/Relation/PGVB、BERT和原导航控制器；只训练{manifest['trainable_parameters']}个新增参数。Loss=Goal CE+0.5 Direction CE+0.05 KL，FP32且TF32关闭。3轮每轮覆盖21878条，实际最后训练记录耗时{hist[-1]['elapsed_seconds']:.2f}秒，训练内GPU步骤均值{stepmean:.3f}ms（不含DataLoader等待），峰值GPU分配{train['peak_gpu_GiB']:.3f}GiB。未修改原权重。

新增头只接收冻结全文/公开名称嵌入、公开候选方位角、有效Anchor Mask和原Belief；无dx/dy、距离、距离桶或GT目标位置。atan2在候选公开几何预处理/训练标签中计算，坐标差和模长不进入新增头。原冻结主模型仍保留原几何输入，本实验不是全模型去除距离。方向为地图+X=0度、逆时针8类，未证明地理北；1m内重合Anchor没有有效方向标签。

修复原PR按目标ID索引processed_descriptions读地标列表的路径：TRAIN及推理均从原文匹配完整公开目录，使用同一public_text_record/full_text，原私有目标ID和GT选锚不进入模型。GT只在forward后构造方向/Goal损失和TRAIN离线Oracle。无Anchor全部回退原Belief并入分母。Anchor采用原特征约定的完整Polygon顶点均值（不是面积质心或GT最近边界）；同名多Polygon独立保留并平均角度相容性，超容量不按GT选择。与已有45.75%基线核对同Checkpoint、公开绑定、起点及控制协议，复用其全部真实轨迹。

## TRAIN内定位诊断（非独立Dev）

Birmingham7619条也参与训练，只能作为拟合诊断。Oracle无闭环开关，不在Val上计算GT方向。

|方法|R20|NE(m)|
|---|---:|---:|
{offline_table}

预测方向分类准确率{tr['direction_class_accuracy']*100:.3f}%，有效角度标签{tr['direction_labeled_anchors']}，有角度支持任务{tr['angular_supported_queries']}。预测定位纠正{tr['fixes']}、破坏{tr['breaks']}，净{tr['net']}。Oracle只评估相同已训练有界融合公式中的GT8方向贡献，不是任意位置Oracle上界。

同名多Anchor的事后监督歧义统计：{coverage}。同一指令和同一名称会给预测头相同输入；如果不同片段的GT方向类不同，当前头只能给出相同的混合方向分布，不能识别片段身份。该统计仅用于事后解释，不据GT选择Anchor或修正预测。

## 全2470/2470 Val-Seen真实闭环

冻结第3轮，不用Val调参；只输入预测角度，Oracle从未进入导航。保留原20m、Horizon20、Stop0.3、1mm到达守卫、Stable Goal与动作策略；RGB更新关闭，与原语言/几何基线一致。

|方法|SR|OSR|SPL|NE(m)|nDTW|
|---|---:|---:|---:|---:|---:|
{nav_table}

配对纠正{stats['fixes']}、破坏{stats['damage']}、净{stats['net']}。实际错误{stats['inference_error']}，初始Top1变化{stats['angle_changed_initial_Top1']}条，有角度支持{stats['angle_supported_episodes']}条；残差最大{summary['angle_delta_max']:.6f}、中位{summary['angle_delta_median']:.6f}logit。OSR成功而SR失败{stats['OSR_success_SR_failed']}条。SR差值物理目标组bootstrap95%区间{summary['SR_delta_physical_target_bootstrap95CI']}；一个训练种子，不宣称Unseen或多种子泛化提升。

新增闭环推理平均{metric['metrics']['inference_ms']:.3f}ms、耗时{metric['seconds']:.2f}秒；旧基线是历史运行，不能作同一时段速度提升结论。

## 实现核查与局限

CPU专项与全测试见CPU_tests.txt；梯度审计证实新增头非零、原模型无梯度；训练后原模型逐张量完全一致。2条TRAIN真实闭环仅作接口验收，使用只暴露公开objects的catalog，无法调用目标索引接口，不评分。修复public-text模式覆盖角度Policy的问题、最终TopK/坐标/128维Context同步及角度修正热图捕获问题。普通默认导航未改变。

视觉实例身份与Polygon投影像素准确性不在本轮验证范围。多Anchor平均、粗8类和±1有界logit是原PR设计，未根据Val调强度或改结构。详见TRAIN24例Polygon/方向概率/三组Belief热图及反事实JSON。固定长度箭头仅供显示，不是输入距离或预测距离。

角度Checkpoint：{r/'full/last.pt'}
SHA256：{frozen['angle_head_sha256']}
原模型SHA256：{manifest['baseline_sha256']}
服务器保留全部日志、Checkpoint、轨迹和逐步字段：{r}。未访问Test、未合并main。
'''
 (out/'REPORT_zh.md').write_text(text)
 cases=json.loads((out/'cases.json').read_text())['cases']
 content=''.join(f'<article><h3>TRAIN {x["TRAIN_index"]}</h3><p>{html.escape(x["original_instruction"])}</p><a href="{x["figure"]}"><img width="100%" src="{x["figure"]}"></a></article>' for x in cases)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Angle-only TRAIN Oracle / Val predicted navigation</title><body style="max-width:1500px;margin:auto"><h1>方位角实验</h1><p>Oracle仅TRAIN离线，不是导航SR。地图轴未核实地理北；橙色箭头定长40m仅为显示。</p><a href="REPORT_zh.md">中文报告</a><img src="training_curve.png"><img src="navigation.png">'+content+'</body>')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
