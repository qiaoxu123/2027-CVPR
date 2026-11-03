"""Frozen transport results, corrected ablations and real offline RGB examples."""
import argparse
import copy
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
from graphnav.checkpointing import file_sha256
from graphnav.reference_evidence_transport import ReferenceTransportPositionHead
from graphnav.visual_chain_position import WorldLikelihoodPositionHead
from scripts.train_local_chain_position import records,collate
from scripts.train_reference_evidence_transport import public_inputs
from scripts.report_world_relation_factor import world_grid_extent


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','suite','control-fix','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('preserve report')
    status=json.loads((a.suite/'status.json').read_text())
    if status['status']!='complete':raise ValueError('terminal complete suite required')
    fix=json.loads((a.control_fix/'wrong_frame_corrected.json').read_text());r=a.root;data={};training={}
    a.output.mkdir(parents=True)
    for mode in ('local','global','constant'):
        folder=a.suite/(mode+'_s0');original=json.loads((folder/'evaluation.json').read_text());resolved=copy.deepcopy(original)
        assert original['checkpoint_sha256']==fix['results'][mode]['checkpoint_sha256']
        resolved['results']['wrong-frame']=fix['results'][mode]['result'];resolved['original_wrong_frame_invalid']=True
        data[mode]=resolved;training[mode]=json.loads((folder/'training.json').read_text())
        for filename in ('config.json','training.json','evaluation.json'):
            shutil.copy2(folder/filename,a.output/(mode+'_'+('evaluation_original_invalid_wrong_frame.json' if filename=='evaluation.json' else filename)))
    (a.output/'resolved_results.json').write_text(json.dumps(data,indent=2));shutil.copy2(a.control_fix/'wrong_frame_corrected.json',a.output/'wrong_frame_corrected.json');shutil.copy2(a.suite/'status.json',a.output/'suite_status.json')
    bound=json.loads((r/'fit_visual_reachability_v1/results.json').read_text());blocked={x['frame_key'] for x in bound['rows'] if not x['bound']['possible']}
    ref=json.loads((r/'public_reference_weak_audit_v1/records.json').read_text())
    reference_stats=dict(blocked_frames=len(blocked),accepted_public_named_object_frames=len({x['frame_key'] for x in ref if x['frame_key'] in blocked}),
        positive_geometry_frames={str(s):len({x['frame_key'] for x in ref if x['frame_key'] in blocked and x['scale_counts'][i]['positive']>0}) for i,s in enumerate((1,3,7))},
        warning='Conservative public named-object geometry support, not observed instance proof; missing support does not prove absence. Ordinary unnamed references are not covered.')
    (a.output/'blocked_reference_support.json').write_text(json.dumps(reference_stats,indent=2))
    lines=['# 全局参照视觉证据传递：同容量训练结果','','**不采用新原型，不进入独立确认、闭环或Val/Test。** 扩展世界证据支持的能力已实现，但新增收益未泛化；Global开发组不如原冻结头、Local或固定图像对照。保留原权重与所有负结果。','',
        '原900拟合状态/351目标/18图，92开发状态/25目标/3图；seed0，固定3×159更新、每步8样本，有放回采样共3816次状态呈现/模型。每组新增131,361参数，仅新模块训练；A0、SigLIP2、原V3逐参数未变。无GT参照实例监督。目标坐标只用于世界Goal损失与事后评分。','',
        '|方法|拟合Goal R1|开发Goal R1|开发NE(m)|开发纠正/破坏原V3|','|---|---:|---:|---:|---:|',
        '|冻结原V3|65.00%|85.87%|17.2714|—|']
    for mode,d in data.items():
        f=d['fit']['metrics'];m=d['results']['real']['metrics'];lines.append(f"|{mode}|{f['goal_r1']:.2%}|{m['goal_r1']:.2%}|{m['goal_ne_m']:.4f}|{m['corrected']}/{m['harmed']}|")
    lines+=['','拟合组本身比开发组更难，因此不能仅比较绝对训练/开发准确率判断过拟合；这里的关键是新增模块在拟合组改善、在开发组破坏原正确结果。','',
        '## Global消融（旧V3始终接收原真实图像）','','|新增分支输入|开发Goal R1|NE(m)|纠正/破坏|','|---|---:|---:|---:|']
    for control,value in data['global']['results'].items():
        m=value['metrics'];lines.append(f"|{control}|{m['goal_r1']:.2%}|{m['goal_ne_m']:.4f}|{m['corrected']}/{m['harmed']}|")
    lines+=['','真实输入未超过打乱、均值或错帧，不能证明新增分支学到了可靠参照对应。关闭关系分支更好也不作为选择新方案的依据；没有按开发结果改模型或继续加轮次。','',
        '## 局部支持限制被解除了吗？','']
    for mode,d in data.items():
        recovered=sum(x['success'] for x in d['fit']['rows'] if x['frame_key'] in blocked)
        lines.append(f'- {mode}：140个原单帧结构受限状态中，拟合组位置成功{recovered}个。')
    lines+=['','Local为0与必要支持界限一致；Global为3，说明覆盖外重排能力确实存在。但固定图像对照也有2个，不能把拟合纠正直接归因为视觉关系推理。','',
        '140个状态中，仅84个进入既有“唯一名称+有效公开多边形”审计；其中1/3/7patch尺度分别仅7/5/3个状态有保守正区域支持。不能把其余状态当作参照不存在，普通汽车/停车场等无名参照也不在这项统计内；它表明当前缺少可核实的参照对应监督。Global纠正的3个结构受限拟合状态，也没有保守的具名参照正窗口，不能据此宣称已学会地标链。','',
        '## 原有高度范围分层','','预先已有public_deployment_filter，仅作事后解释，完整92状态仍为主结果：','','|方法|合法高度/初始例外68状态正确|其余24状态正确|','|---|---:|---:|','|原V3|59|20|']
    allowed=set(json.loads((r/'public_deployment_filter.json').read_text())['frame_keys'])
    for mode,d in data.items():
        rs=d['results']['real']['rows'];lines.append(f"|{mode}|{sum(x['success'] for x in rs if x['frame_key'] in allowed)}|{sum(x['success'] for x in rs if x['frame_key'] not in allowed)}|")
    lines+=['','Global在合法范围也退化，不能把负结果全部归因于教师低空分布。Local在合法范围多1个，但不足以在本轮事后晋级该对照。','',
        '## 工程修正与真实GPU证据','','v1 Smoke在更新前因分开两次旧头输出逐值比较而退出；新增证据实际严格为0。CUDA重复局部scatter实验最大差9.54e-7、确定性模式为0。修正为同一次基线核验，要求零证据及输出逐值相等，未放松任何性能/Stop阈值。v1原失败目录保留。v2 Smoke、恢复及3组正式训练均退出0。','',
        '原wrong-frame列漏传已构造的错帧张量，属于无效对照。新增真实GPU集成回归先失败再修复；只冻结重算此列，不重训、不覆盖原评估。resolved_results.json与wrong_frame_corrected.json为修正后的结果，原无效列另存供审计。','']
    for mode,t in training.items():lines.append(f"- {mode}：训练循环{t['history'][-1]['seconds']:.2f}s，读出梯度{t['score_gradient']:.6f}、图像投影梯度{t['image_gradient']:.6f}，峰值{t['peak_allocated_gib']:.3f}GiB（缓存特征头训练，不含SigLIP编码常驻权重）。")
    lines+=['','## 下一步决策','','结构支持不是唯一瓶颈。当前原型缺少可靠的参照身份/可观测性绑定，弱Goal监督允许它学习错误的图像—关系偏移，导致真实RGB反而增加开发误改；具体错绑仍需逐例核验，不能只由消融断言全部原因。','',
        '下一轮先准备更充分的TRAIN参照观测与绑定数据：按指令中的公开具名对象和真实当前Footprint选取已有教师轨迹中的参照支持帧，保留像素配准、遮挡和高度不确定性；不按GT目标位置选择推理ROI，不把无名对象自动绑定到具名地标。先检查真实RGB与关系作用域，再决定是否增加参照监督或扩大训练。','',
        '不继续盲目增加全局自由度、Attention或训练轮数。最终验收仍需独立实例确认、真实闭环及冻结Val128严格>65%的门槛；目前没有达到该目标。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for mode,t in training.items():axes[0].plot([x['epoch'] for x in t['history']],[x['loss'] for x in t['history']],'.-',label=mode)
    axes[0].set_xlabel('Fixed epoch');axes[0].set_ylabel('Goal loss');axes[0].legend()
    labels=list(data);fit_delta=[(data[m]['fit']['metrics']['goal_r1']-.65)*100 for m in labels];dev_delta=[(data[m]['results']['real']['metrics']['goal_r1']-79/92)*100 for m in labels]
    xs=np.arange(3);axes[1].bar(xs-.15,fit_delta,.3,label='Fit');axes[1].bar(xs+.15,dev_delta,.3,label='Development');axes[1].set_xticks(xs,labels);axes[1].axhline(0,color='black');axes[1].set_ylabel('Goal R1 change (percentage points)');axes[1].legend();fig.tight_layout();fig.savefig(a.output/'training_and_generalization.png',dpi=160);plt.close(fig)
    torch.set_num_threads(4);roles=torch.load(r/'reference_vectors_v2.pt',map_location='cpu',weights_only=True)
    fit_rows=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_fit.jsonl','fit',roles,'reference',allowed);dev_rows=records(r/'chain_train400_features',r/'fixed_yaw_train400_s1/labels_dev.jsonl','dev',roles,'reference')
    by_key={x['frame_key']:x for x in fit_rows+dev_rows};frames={x['frame_key']:x for x in map(json.loads,(r/'fixed_yaw_train400_s1/public_frames.jsonl').read_text().splitlines())}
    harms=sorted([x for x in data['global']['results']['real']['rows'] if x['baseline_success'] and not x['success']],key=lambda x:x['ne_m']-x['baseline_ne_m'],reverse=True)[:2]
    fixed=[x for x in data['global']['fit']['rows'] if x['frame_key'] in blocked and x['success']][:1];selected=harms+fixed
    state=torch.load(a.suite/'global_s0/final.pt',map_location='cpu',weights_only=True);model=ReferenceTransportPositionHead(WorldLikelihoodPositionHead(),mode='global').cuda().eval();model.load_state_dict(state['model'])
    case_records=[]
    for saved in selected:
        key=saved['frame_key'];row=by_key[key];batch=collate([row])
        with torch.no_grad():new,energy,base=model(**public_inputs(batch,'cuda'),return_base=True)
        world=row['candidate_world'][:784].reshape(28,28,2);target=row['target_world'].numpy();old_choice=int(base[0,:784].argmax());new_choice=int(new[0,:784].argmax())
        old_xy=row['candidate_world'][old_choice].numpy();new_xy=row['candidate_world'][new_choice].numpy();extent=world_grid_extent(world)
        image=r/'fixed_yaw_train400_s1'/frames[key]['image'];assert file_sha256(image)==frames[key]['image_sha256']
        fig,axes=plt.subplots(1,4,figsize=(15,4));axes[0].imshow(Image.open(image));axes[0].axis('off');axes[0].set_title('Actual TRAIN observed RGB')
        for ax,values,title in [(axes[1],base[0,:784].softmax(-1),'Frozen V3'),(axes[2],new[0,:784].softmax(-1),'Global transport'),(axes[3],energy[0],'Added evidence')]:
            ax.imshow(values.cpu().reshape(28,28).numpy(),extent=extent,origin='lower',cmap='magma');ax.scatter(*target,c='red',marker='x',s=35);ax.scatter(*old_xy,c='cyan',s=18);ax.scatter(*new_xy,c='lime',s=18);fp=np.asarray(frames[key]['footprint_world']);fp=np.vstack([fp,fp[0]]);ax.plot(fp[:,0],fp[:,1],color='deepskyblue',lw=.7);ax.set_title(title);ax.set_aspect('equal')
        fig.suptitle(key+' | offline only; red=GT posthoc, cyan=base, green=new\n'+'\n'.join(textwrap.wrap(frames[key]['instruction'],135)),fontsize=7);fig.tight_layout();fig.savefig(a.output/(key+'.png'),dpi=150);plt.close(fig)
        case_records.append(dict(frame_key=key,scope='fit' if key in blocked else 'development',original_evaluation=saved,recomputed_goal_ne_m=float(np.linalg.norm(new_xy-target))))
    (a.output/'cases.json').write_text(json.dumps(case_records,indent=2));print(json.dumps({m:d['results']['real']['metrics'] for m,d in data.items()}))


if __name__=='__main__':main()
