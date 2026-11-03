"""Posthoc one-shot confirmation report and actual TRAIN RGB cases."""
import argparse
from collections import Counter
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
from graphnav.checkpointing import file_sha256
from scripts.report_world_relation_factor import world_grid_extent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('evaluation','observations','cache','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve previous report')
    result=json.loads((a.evaluation/'results.json').read_text());seal=json.loads((a.evaluation/'prediction_seal.json').read_text())
    assert seal['predictions_sha256']==file_sha256(a.evaluation/'predictions_public.json')
    scores=result['results'];base=scores['FrozenV3']['rows']
    a.output.mkdir(parents=True)
    for filename in ('results.json','prediction_seal.json'):shutil.copy2(a.evaluation/filename,a.output/filename)
    lines=['# 原始完整指令目标通道：独立TRAIN确认结果','','## 决策','','**未通过预登记闭环准入条件，不开展本候选闭环或Val/Test，不根据这65个目标继续调参。** FullTarget优于旧冻结V3，但未超过同预算Role对照，不能把新增训练的收益归因于补回全文。','',
           '65个新物理目标，3张留出TRAIN地图；Car30、Building35，无Parking/Ground。原定260个参考轨迹均匀采样位置，3个Car位置海拔无效，得到257个可渲染状态（Car117、Building140）；全部65个目标保留。均非真实闭环，不能报告为SR。','',
           '先生成并封存所有模型预测及SHA，再读取私有TRAIN标签评分；prediction_seal.json记录封存证据。原A0预训练使用合法TRAIN，这里的独立性只针对新增局部头拟合与本轮选方案。','',
           '|方法|正确状态|Goal R1|NE(m)|20m概率质量|纠正/误改A0|','|---|---:|---:|---:|---:|---:|']
    a0_ne=np.mean([x['baseline_ne_m'] for x in base]);a0_n=sum(x['baseline_success'] for x in base)
    lines.append(f'|A0|{a0_n}/257|{a0_n/257:.2%}|{a0_ne:.4f}|未单独保存|—|')
    for name,data in scores.items():
        m=data['metrics'];lines.append(f"|{name}|{sum(x['success'] for x in data['rows'])}/257|{m['goal_r1']:.2%}|{m['goal_ne_m']:.4f}|{m['probability_mass_20m']:.4f}|{m['corrections']}/{m['harms']}|")
    lines+=['','## 配对与不确定性','']
    for name,pair in result['paired'].items():
        lo,hi=pair['target_group_bootstrap95_r1_delta'];lines.append(f"- FullTarget相对{name}：纠正{len(pair['corrected'])}、破坏{len(pair['harmed'])}个状态；R1差95%目标分组bootstrap区间[{lo*100:.2f}, {hi*100:.2f}]个百分点。")
    lines+=['','这些区间按物理目标分组，条件于当前3张地图；并未证明跨地图显著性，同目标4时刻不是4个独立目标。', '',
            '|方法|Car正确/状态|Car NE|Building正确/状态|Building NE|','|---|---:|---:|---:|---:|']
    for name,data in scores.items():
        c=[x for x in data['rows'] if x['category']=='Car'];b=[x for x in data['rows'] if x['category']=='Building']
        lines.append(f"|{name}|{sum(x['success'] for x in c)}/{len(c)}|{np.mean([x['ne_m'] for x in c]):.4f}|{sum(x['success'] for x in b)}/{len(b)}|{np.mean([x['ne_m'] for x in b]):.4f}|")
    lines+=['','## 机制结论','','1. 目标通道全文补回后，开发集的小幅收益没有超越独立组的Role对照。不能只凭已知5对输入碰撞就认定它是导航主要根因。','2. 独立组FullTarget真实RGB优于同模型打乱RGB，但这证明的是世界位置预测利用了视觉对应，不是精确实例识别或数数能力。','3. 原始SigLIP2文本上限64 token，完整原文仍可能截断；没有修改预训练长度规则，也没有重新训练视觉骨干。','4. 保留旧冻结V3，不因Role对照此次离线更好就事后将其选成确认胜者。本轮不恢复任何暂停评测。','',
            '## 下一步','','当前证据不支持继续扩大全文目标通道训练。后续若研究序数/实例链，应先建立可验证的对象/行列对应监督和参考系，再检验现有patch特征是否足够；不能把地图点核心或20m定位成功充当实例对应。优先改进真实局部监督与世界位置证据，而不是继续全局语言替换或增加关系网络。已消费的65确认目标不能再次用于选方案。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4));names=list(scores)
    axes[0].bar(names,[scores[n]['metrics']['goal_r1']*100 for n in names]);axes[0].set_ylabel('Goal R1 (%)');axes[0].set_ylim(0,100)
    axes[1].bar(names,[scores[n]['metrics']['goal_ne_m'] for n in names]);axes[1].set_ylabel('Goal NE (m)')
    for ax in axes:ax.tick_params(axis='x',labelrotation=20)
    fig.suptitle('One-shot TRAIN confirmation: 65 targets / 257 reference observations');fig.tight_layout();fig.savefig(a.output/'confirmation_metrics.png',dpi=150);plt.close(fig)
    indexed={name:{r['frame_key']:r for r in data['rows']} for name,data in scores.items()}
    changed=result['paired']['TrainedRole'];chosen=changed['corrected'][:1]+changed['harmed'][:2]
    frames={r['frame_key']:r for r in map(json.loads,(a.observations/'public_frames.jsonl').read_text().splitlines())}
    labels={r['frame_key']:r for r in map(json.loads,(a.observations/'labels_private.jsonl').read_text().splitlines()) if r['status']=='captured'}
    meta=json.loads((a.cache/'manifest.json').read_text());features={}
    for chunk in meta['chunks']:
        for row in torch.load(a.cache/chunk['file'],map_location='cpu',weights_only=True)['records']:
            if row['frame_key'] in chosen:features[row['frame_key']]=row
    posterior=torch.load(a.evaluation/'posteriors_public.pt',map_location='cpu',weights_only=True)
    order={r['frame_key']:i for i,r in enumerate(scores['TrainedFullTarget']['rows'])}
    for key in chosen:
        frame=frames[key];path=a.observations/frame['image'];assert file_sha256(path)==frame['image_sha256']
        rgb=np.array(Image.open(path).convert('RGB'));target=np.asarray(labels[key]['target_world'][:2]);pixel=world_to_frame(target[None],frame['footprint_world'],rgb.shape)[0]
        fig,axes=plt.subplots(1,2,figsize=(11,5));axes[0].imshow(rgb)
        if np.isfinite(pixel).all() and ((pixel>=0)&(pixel<384)).all():axes[0].scatter(*pixel,c='red',marker='x',s=70)
        axes[0].axis('off');axes[0].set_title('Actual 384px RGB; red x=map-point proxy')
        world=features[key]['candidate_world'][:784].reshape(28,28,2)
        axes[1].imshow(posterior['TrainedFullTarget'][order[key]].reshape(28,28).numpy(),extent=world_grid_extent(world),origin='lower',cmap='magma')
        axes[1].scatter(*target,c='red',marker='x',label='GT posthoc')
        for name,color in [('FrozenV3','cyan'),('TrainedRole','lime'),('TrainedFullTarget','white')]:
            xy=indexed[name][key]['predicted_world'];axes[1].scatter(*xy,c=color,marker='o',s=24,label=f"{name}: {indexed[name][key]['ne_m']:.1f}m")
        fp=np.asarray(frame['footprint_world']);fp=np.vstack([fp,fp[0]]);axes[1].plot(fp[:,0],fp[:,1],c='deepskyblue',lw=1)
        axes[1].set_aspect('equal');axes[1].legend(fontsize=7);axes[1].set_title('World posterior; same fixed world axes')
        fig.suptitle(key+'\n'+'\n'.join(textwrap.wrap(frame['instruction'],100)),fontsize=8);fig.tight_layout();fig.savefig(a.output/f'{key}.png',dpi=150);plt.close(fig)
    print(json.dumps(dict(navigation_gate=result['navigation_gate'],figures=chosen)))


if __name__=='__main__':main()
