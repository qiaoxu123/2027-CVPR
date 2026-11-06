"""Render live/final V3 TRAIN curves; never treats partial evaluation as full results."""
import argparse,json,math,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=a.run;o=a.output;o.mkdir(parents=True,exist_ok=True)
    events=[]
    for line in (r/'run.log').read_text(errors='replace').splitlines():
        try:v=json.loads(line)
        except json.JSONDecodeError:continue
        if isinstance(v,dict) and 'epoch' in v and ('mean_loss' in v or 'mean_train_loss' in v):events.append(v)
    fit=json.loads((r/'full_split_expected.json').read_text());nfit=fit['fit'];epochs=[x for x in events if 'mean_train_loss' in x]
    points=[x for x in events if 'mean_loss' in x]
    x=[(v['epoch']-1)*nfit+v['update'] for v in points];y=[v['mean_loss'] for v in points]
    assert all(math.isfinite(v) for v in y)
    fig,ax=plt.subplots(figsize=(9,4));ax.plot(x,y,label='Within-epoch cumulative mean (every 1000 updates)')
    if epochs:ax.plot([v['epoch']*nfit for v in epochs],[v['mean_train_loss'] for v in epochs],'o-',label='Completed epoch mean')
    ax.set(xlabel='Cumulative optimizer updates',ylabel='TRAIN weighted loss',title='V3 real GPU training; no Val selection');ax.legend(fontsize=7);fig.tight_layout();fig.savefig(o/'training_loss.png',dpi=150);plt.close(fig)
    keys=['geo_ce','geo_polygon','fused_ce','fused_polygon','text_visual','gt_image_visual'];fig,axes=plt.subplots(2,3,figsize=(13,6))
    for ax,k in zip(axes.flat,keys):
        vals=[(xx,v['components_mean'][k]) for xx,v in zip(x,points) if k in v.get('components_mean',{})]
        if vals:ax.plot(*zip(*vals))
        ax.set(title=k,xlabel='Cumulative updates',ylabel='Within-epoch mean')
    fig.tight_layout();fig.savefig(o/'loss_components.png',dpi=150);plt.close(fig)
    final_path=r/'train/report.json';final=json.loads(final_path.read_text()) if final_path.exists() else None
    initial_path=r/'val_seen_initial_full/summary.json';initial=json.loads(initial_path.read_text()) if initial_path.exists() else None
    nav={}
    for name in ['geometry','fused']:
        path=r/f'navigation_{name}/metrics.json'
        if path.exists():
            d=json.loads(path.read_text());assert d['coverage']=={'completed':2470,'expected':2470};nav[name]=d
    status=dict(train_finished=final is not None,completed_epochs=len(epochs),last_logged_update=points[-1] if points else None,split=fit,initial_evaluation=initial,navigation=nav)
    if final:
        status['final_training_manifest']=final['manifest'];status['checkpoint_sha256']=file_sha256(r/'train/last.pt')
    atomic_json(status,o/'status.json')
    for src,dst in [('cache_audit.json','cache_audit.json'),('cache/manifest.json','cache_manifest.json'),('train/losses.json','losses.json'),('train/invocation.json','invocation.json')]:
        if (r/src).exists():shutil.copyfile(r/src,o/dst)
    report=['# UnifiedPGVB-V3 完整TRAIN训练与评测','',f'完整TRAIN输入{fit["total"]}条；本阶段拟合{fit["fit"]}条、物理目标留出{fit["dev"]}条，物理分组重叠0。初始化V2曾使用TRAIN，内部留出不是完全未见泛化。','',f'当前已完成完整Epoch：{len(epochs)}；训练结束：{final is not None}。','', 'Geometry和轻量视觉模块训练，BERT/SigLIP2冻结并读取缓存；seed42，原权重保留。四小时累计训练预算，至少三轮后连续两轮相对Loss改善<0.5%提前停止。未修改Stop=0.3、1mm守卫或Horizon。','', '训练视点由公开地标选择，属于离线地图引导观测；不能作为无人机真实已看到的图像。导航/初始Goal评测使用真实起点及因果历史，保留原yaw，存在需检查的训练—部署分布差异。Polygon/Depth均是弱监督/2.5D代理，不是人工像素真值。','', '|Epoch|拟合样本|TRAIN Loss|本轮Wall秒|GPU峰值GiB|','|---|---:|---:|---:|---:|']
    for v in epochs:report.append(f'|{v["epoch"]}|{v["fit"]}|{v["mean_train_loss"]:.6f}|{v["elapsed_s"]:.2f}|{v["peak_cuda_GiB"]:.3f}|')
    report+=['','![训练Loss](training_loss.png)','![Loss分量](loss_components.png)','', '## 全量评测状态','']
    if initial:
        assert initial['coverage']=={'completed':2470,'expected':2470}
        report += [f'全量初始定位覆盖2470/2470，推理错误{initial["errors"]}条。','', '|方法|Top1≤20m|Top5≤20m|Top20≤20m|NE(m)|','|---|---:|---:|---:|---:|']
        for name,m in initial['metrics'].items():report.append(f'|{name}|{m["Top1_20m"]*100:.2f}%|{m["Top5_20m"]*100:.2f}%|{m["Top20_20m"]*100:.2f}%|{m["NE"]}|')
        report += ['',f'55%导航门槛通过：{initial["navigation_gate_passed"]}。这是初始Goal指标，不是SR。']
    else:report+=['尚无全部2470条Val-Seen定位结果；不报告子集作为正式效果。']
    if nav:
        report+=['','|方法|SR|OSR|SPL|NE|nDTW|','|---|---:|---:|---:|---:|---:|']
        for name,m in nav.items():
            z=m['metrics'];report.append(f'|{name}|{z["SR"]*100:.2f}%|{z["OSR"]*100:.2f}%|{z["SPL"]:.4f}|{z["NE"]:.3f}|{z["nDTW"]:.4f}|')
    else:report+=['','尚无完整闭环结果，不宣称SR提升。']
    report+=['','训练原始日志/权重留在服务器：`'+str(r)+'`。本报告只汇总真实文件，不预测未完成结果。']
    (o/'REPORT_zh.md').write_text('\n'.join(report)+'\n');(o/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>V3 Training</title><h1>V3 TRAIN曲线；不是导航SR</h1><a href="REPORT_zh.md">中文报告</a><br><img src="training_loss.png"><img width="1200" src="loss_components.png">')
    print(json.dumps(dict(completed_epochs=len(epochs),train_finished=final is not None),indent=2))
if __name__=='__main__':main()
