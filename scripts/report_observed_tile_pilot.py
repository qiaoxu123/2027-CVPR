"""Actual RGB and separate biased TRAIN cohorts; no navigation claims."""
import argparse
import html
import json
from pathlib import Path
import shutil
import textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
from graphnav.landmark_projection import world_to_frame


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if (a.output/'REPORT_zh.md').exists():raise FileExistsError('preserve report')
    a.output.mkdir(parents=True,exist_ok=True);r=a.root;page=[];runtime=[];data={};private_all={};forward_paths={}
    for cohort,stem in [('fit24','observed_tile_fit24'),('hard12','observed_tile_hard12')]:
        selection=r/(stem+'_selection_v1');forward=r/(stem+'_forward_v1');scores=r/(stem+'_scores_v1')
        d=json.loads((scores/'results.json').read_text());data[cohort]=d;private=json.loads((selection/'private_scoring.json').read_text());pub=json.loads((selection/'public_selection.json').read_text());inputs=torch.load(selection/'public_inputs.pt',map_location='cpu',weights_only=True)
        times=json.loads((forward/'runtime.json').read_text());runtime.extend(times);private_all.update(private)
        shutil.copy2(scores/'results.json',a.output/(cohort+'_results.json'));shutil.copy2(forward/'prediction_seal.json',a.output/(cohort+'_prediction_seal.json'));shutil.copy2(forward/'run_manifest.json',a.output/(cohort+'_run_manifest.json'))
        for row in pub:
            key=row['case_key'];forward_paths[key]=forward
            paths={}
            for kind in ('low384','native768'):
                filename=key+'_'+kind+'.png';shutil.copy2(forward/'images'/filename,a.output/filename);paths[kind]=filename
            frame=inputs[key]['frame'];target=np.asarray(private[key]['target_world'][:2]);pixel=world_to_frame(target[None],frame['footprint_world'],(384,384,3))[0]
            page.append(dict(key=key,cohort=cohort,instruction=row['instruction'],category=private[key]['category'],images=paths,
                pixel=pixel.tolist(),metrics={m:next(x for x in rs if x['case_key']==key) for m,rs in d['rows'].items()}))
    lines=['# 真实高清分块与视觉候选：冻结原SigLIP2结果','','**当前冻结头没有显示稳定的高清净收益，不扩大候选预算或直接进入闭环。** 困难Car中出现1个高清纠正，但首批混合组同时有1个误改；两个群组抽样方式不同，不能合并宣称总体性能。','',
           '所有图像保持同XY、高度、yaw0和Footprint，当前768图来自原始栅格重新采样；插值对照由同一384图生成。768不是源数据最高可用分辨率。4块均实际以384×384输入冻结SigLIP2，池化物理范围对齐，共用有效观测交集。A0及V3权重均冻结。','',
           '## 两组结果必须分开读','']
    for cohort,d in data.items():
        lines += [f'### {cohort}', '', '首批12个Car+其他类别各4，Car偏向已有几何支持。' if cohort=='fit24' else '额外12个A0误差>20m的拟合Car，排除首批物理目标/图像；这是按失败条件筛选的机制分析。', '', '|方法|Goal正确|NE(m)|局部位置Top1/Top5（20m）|','|---|---:|---:|---:|']
        for m,s in d['summary'].items():lines.append(f"|{m}|{s['correct']}/{s['cases']}|{s['goal_ne_m']:.4f}|{s['visual_world_20m_top1']}/{s['visual_world_20m_top5']}|")
    lines += ['', '首批Car的Goal均12/12正确，不能用它检验车辆纠错能力。困难组真实768/144相对插值纠正1条、误改0条；增加到576候选后反而丢失2条原正确结果。目标20m/1m附近的世界位置支持都不是实际实例匹配。没有人工像素GT，因此真实实例准确率未知。','',
        '## 开销与数值核验','',f"视觉编码中位耗时：Whole384 {np.median([x['whole384']['seconds'] for x in runtime])*1000:.1f}ms；真实768分块 {np.median([x['native768']['seconds'] for x in runtime])*1000:.1f}ms。原始输入为1×3×384×384或4×3×384×384，高清编码峰值分配显存{max(x['native768']['gpu_peak_allocated_gib'] for x in runtime):.3f}GiB。计时包括本原型576位置共享池化，不含A0/BERT、渲染和控制器。", '',
        '旧缓存按batch4编码，当前Whole384按batch1；重提取平均余弦约0.999，但有局部向量明显偏差。对困难组重复同一384图4次，仅取首图tokens，缓存最小余弦恢复到约0.9999995，确认BF16计算batch形状影响。12条Goal正确数与NE未变化；不把数值差当新视觉信息，也未更改普通运行时。','',
        '## 更重要的结构约束','',
        '另对原900拟合状态做必要覆盖界限检查：当前315个Goal失败中，140个正确20m区域没有局部更新支持，且覆盖之外的先验最佳位置也是错的。只改变现有局部得分无法让正确区域成为单帧Goal最高峰。其余19个属于覆盖外先验最高位置已正确、156个有可更新的正确格点；后者仅是乐观可能，双线性投影会耦合格点，不能写成可恢复156条导航。','',
        '这是单帧Goal argmax的结构性质，不是轨迹SR上限；后续移动可以改变覆盖范围。界限计算与真实V3预测逐帧核对，没有出现模型成功却被判不可达的矛盾。GT只用于诊断成功区域。','',
        '## 下一步决策','',
        '优先验证轻量的全局参照证据传递：保持原目标外观和局部头，利用当前可观测参照物的视觉特征、完整指令关系和相对世界几何，对覆盖之外的目标候选也产生可学习证据。以相同容量的局部支持版本作对照，先检验能否触及上述结构限制，并验证RGB打乱/关系关闭后的变化。','',
        '这比继续增大候选预算更直接检验地标→区域→目标链。尚未证明可行，也不能把140个状态全称为可修复：参照物是否真的可见、身份/作用域是否正确仍待确认。只有TRAIN独立收益与配对闭环支持后，才进入新的冻结Val评估。','',
        '全部36条真实图像可在index.html切换、放大查看。红圈为TRAIN地图点投影代理，GT中心裁剪仅用于本报告，模型ROI始终为固定网格，不读取这些标记。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    reach=json.loads((r/'fit_visual_reachability_v1/results.json').read_text());(a.output/'reachability.json').write_text(json.dumps(reach,indent=2))
    hard_correct=data['hard12']['paired']['Interpolated768_144 -> Native768_144']['corrected']
    chosen=[page[0]['key'],page[1]['key']]+hard_correct
    chosen += [x['key'] for x in page if x['cohort']=='hard12' and not x['metrics']['Native768_144']['goal_success']][:2]
    chosen += data['fit24']['paired']['Whole384_144 -> Native768_144']['harmed'][:1]
    for entry in page:
        if entry['key'] not in chosen:continue
        key=entry['key'];frame_dir=forward_paths[key]/'images';low=np.array(Image.open(frame_dir/(key+'_low384.png')));high=np.array(Image.open(frame_dir/(key+'_native768.png')));inter=np.array(Image.open(frame_dir/(key+'_interpolated768.png')))
        point=np.array(entry['pixel']);inside=np.isfinite(point).all() and ((point>=0)&(point<384)).all();center=point if inside else np.array([191.5,191.5]);x0,y0=np.clip(np.rint(center-24).astype(int),0,336);x1,y1=x0+48,y0+48
        hx0,hy0=np.rint(np.array([x0,y0])*767/383).astype(int);hx1,hy1=np.rint(np.array([x1,y1])*767/383).astype(int)
        fig,axes=plt.subplots(1,4,figsize=(13,4))
        for ax,image,title,origin,scale in zip(axes,[low,low[y0:y1,x0:x1],inter[hy0:hy1,hx0:hx1],high[hy0:hy1,hx0:hx1]],['Actual whole384','Same-world crop: 384','Same-world crop: interpolated768','Same-world crop: native768'],[(0,0),(x0,y0),(hx0,hy0),(hx0,hy0)],[1,1,767/383,767/383]):
            ax.imshow(image,interpolation='nearest');ax.axis('off');ax.set_title(title,fontsize=8)
            if inside:ax.scatter(*(point*scale-np.array(origin)),facecolors='none',edgecolors='red',s=60)
        caption='\n'.join(textwrap.wrap(entry['instruction'],130));fig.suptitle(key+'\n'+caption+'\nRed circle = map-point proxy; not verified pixel GT. Display crops are posthoc only.',fontsize=7);fig.tight_layout();fig.savefig(a.output/(key+'_comparison.png'),dpi=150);plt.close(fig)
    (a.output/'runtime.json').write_text(json.dumps(runtime,indent=2))
    encoded=json.dumps(page,ensure_ascii=False).replace('<','\\u003c')
    content='''<!doctype html><meta charset="utf-8"><title>CityNav 冻结高清分块对照</title><style>body{font:16px sans-serif;margin:24px;background:#f5f5f5}#views{display:grid;grid-template-columns:1fr 1fr;gap:16px}.pane{overflow:auto;height:65vh;background:#222}img{display:block;width:100%;max-width:none}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #aaa}</style><h2>真实TRAIN当前观测：低清与原生768</h2><p>两组抽样偏向不同，不能合并为总体性能。无人工像素GT；红圈对照图只用于事后诊断。这里不是导航SR。</p><select id="case"></select> 放大 <input id="zoom" type="range" min="1" max="5" step="0.25" value="1"><p id="text"></p><div id="views"><div><h3>原始384</h3><div class="pane"><img id="low"></div></div><div><h3>真实768（同一视域）</h3><div class="pane"><img id="high"></div></div></div><div id="metrics"></div><script>const cases=DATA;const sel=document.querySelector('#case');cases.forEach((c,i)=>{let o=document.createElement('option');o.value=i;o.textContent=c.cohort+' '+c.category+' '+c.key;sel.appendChild(o)});function show(){let c=cases[sel.value];document.querySelector('#text').textContent=c.instruction;document.querySelector('#low').src=c.images.low384;document.querySelector('#high').src=c.images.native768;let t='<table><tr><th>方法</th><th>Goal NE(m)</th><th>20m内</th></tr>';for(let [name,m] of Object.entries(c.metrics))t+='<tr><td>'+name+'</td><td>'+m.goal_ne_m.toFixed(3)+'</td><td>'+m.goal_success+'</td></tr>';document.querySelector('#metrics').innerHTML=t+'</table>'}sel.onchange=show;document.querySelector('#zoom').oninput=e=>document.querySelectorAll('img').forEach(x=>x.style.width=(100*e.target.value)+'%');show();</script>'''
    (a.output/'index.html').write_text(content.replace('DATA',encoded))
    print(json.dumps(dict(cases=len(page),panels=chosen)))


if __name__=='__main__':main()
