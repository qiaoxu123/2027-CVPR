"""Post-evaluation report for COMPLETE, frozen, matched Val-Seen experiments."""
import argparse,csv,gzip,hashlib,json,sys,shutil
from html import escape
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.landmark_projection import world_to_frame

ARMS=('original_geometry','geometry','relation','rgb')
LABELS={'original_geometry':'Original Geometry','geometry':'Trained Geometry','relation':'Geometry+Relation','rgb':'Geometry+Relation+RGB'}
KEYS=('SR','OSR','SPL','NE','nDTW','inference_ms','steps','path_length')


def main():
    p=argparse.ArgumentParser()
    for k in ('run','data-root','sbf-source','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--group-inventory',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);sys.path.insert(0,str(a.sbf_source))
    from sbfnav.mapdata import GROUND_LEVEL
    raw=json.loads((a.data_root/'processed_citynav/citynav_val_seen.json').read_text());N=len(raw);assert N==2470
    categories=json.loads((a.data_root/'cityrefer/objects.json').read_text());group_rows=json.loads((a.group_inventory/'tasks_public.json').read_text());groups={x['index']:x['group'] for x in group_rows};strict_ids={x['index'] for x in group_rows if x['anchors']}
    results={};summary={};case_analysis={};allrows=[]
    for arm in ARMS:
        path=a.run/f'eval_{arm}';report=json.loads((path/'metrics.json').read_text())
        rr=[json.loads(l) for l in (path/'predictions.jsonl').read_text().splitlines()];assert len(rr)==N and len({r['index'] for r in rr})==N
        results[arm]={r['index']:r for r in rr};summary[arm]={k:float(np.mean([r['metrics'][k] for r in rr])) for k in KEYS};summary[arm]['N']=N
        summary[arm]['terminations']=dict(Counter(r['termination'] for r in rr));analysis=[]
        for row in rr:
            idx=row['index'];r=raw[idx];target=np.asarray(r['target_positions'][0][:2]);diags=row['step_diagnostics'];poses=np.array(row['poses']);goals=[d['selected_goal_xy'] for d in diags if 'selected_goal_xy' in d]
            gd=np.linalg.norm(np.asarray(goals)-target,axis=-1) if goals else np.array([]);pd=np.linalg.norm(poses[:,:2]-target,axis=-1)
            replan=sum(d.get('stable_navigation_goal',{}).get('event')=='replanned' for d in diags);rgb_updates=sum(bool(d.get('rgb_active')) for d in diags)
            if row.get('error'):cause='inference_error'
            elif row['metrics']['SR']:cause='success'
            elif row['metrics']['OSR']:cause='entered_20m_but_not_final_success'
            elif len(gd) and (gd<=20).any():cause='correct_goal_ever_but_not_reached'
            else:cause='never_correct_navigation_goal'
            first=int(np.where(pd<=20)[0][0]) if (pd<=20).any() else None
            first_stop=None
            if first is not None and first<len(diags):
                d=diags[first];first_stop=dict(goal_correct=bool(np.linalg.norm(np.array(d['selected_goal_xy'])-target)<=20),stop_probability=d.get('stop_probability'),predicted_goal_distance=d.get('predicted_goal_distance'),altitude_guard=d.get('stop_altitude_guard'),arrival_guard=(d.get('predicted_goal_distance') is None or d['predicted_goal_distance']<=1e-3))
            category=categories[row['map_name']].get(str(r['object_ids'][0]),{}).get('object_type','Unknown')
            item=dict(index=idx,arm=arm,group=groups[idx],category=category,cause=cause,RGB_updates=rgb_updates,replans=replan,
                first_20m_step=first,first_entry_stop=first_stop,ever_correct_goal=bool((gd<=20).any()),goal_jump_count=int((np.linalg.norm(np.diff(np.asarray(goals),axis=0),axis=-1)>20).sum()) if len(goals)>1 else 0,
                no_public_entity=bool(diags and diags[0].get('relation_anchor_count')==0),visual_anchor_available=bool(diags and diags[0].get('anchor_audit',{}).get('visual_anchor_available')),metrics=row['metrics'])
            analysis.append(item);allrows.append(item)
        case_analysis[arm]=analysis
        summary[arm]['failure_causes']=dict(Counter(x['cause'] for x in analysis));summary[arm]['RGB_updates']=sum(x['RGB_updates'] for x in analysis);summary[arm]['replans']=sum(x['replans'] for x in analysis)
        summary[arm]['RGB_reasons']=dict(Counter(d.get('rgb_reason','unknown') for row in rr for d in row['step_diagnostics']))
        summary[arm]['replan_block_reasons']=dict(Counter(d.get('stable_navigation_goal',{}).get('replan_block_reason','unknown') for row in rr for d in row['step_diagnostics']))
        summary[arm]['strata']={}
        for group in ('direct','indirect','unresolved_or_no_relation','no_public_entity'):
            selected=[x for x in analysis if x['group']==group] if group!='no_public_entity' else [x for x in analysis if x['no_public_entity']]
            summary[arm]['strata'][group]=dict(N=len(selected),**{k:float(np.mean([x['metrics'][k] for x in selected])) if selected else None for k in KEYS})
        selected=[x for x in analysis if x['index'] in strict_ids]
        summary[arm]['strata']['previous_strict_direct_56']=dict(N=len(selected),**{k:float(np.mean([x['metrics'][k] for x in selected])) for k in KEYS})
        summary[arm]['target_categories']={}
        for category in sorted({x['category'] for x in analysis}):
            selected=[x for x in analysis if x['category']==category]
            summary[arm]['target_categories'][category]=dict(N=len(selected),**{k:float(np.mean([x['metrics'][k] for x in selected])) for k in KEYS})
        shutil.copyfile(path/'full_relation_manifest.json',a.output/f'freeze_{arm}.json')
        shutil.copyfile(path/'metrics.json',a.output/f'full_metrics_{arm}.json')
        with (path/'predictions.jsonl').open('rb') as source,gzip.open(a.output/f'full_trajectories_{arm}.jsonl.gz','wb') as dest:
            shutil.copyfileobj(source,dest)
    frozen=[json.loads((a.run/f'eval_{arm}/full_relation_manifest.json').read_text()) for arm in ARMS]
    for key in ('stage1','original_v2','split','data_root','evaluation_code','model_code','relation_runtime_code','config'):
        assert len({x[key] for x in frozen})==1, 'Paired protocol/source differs: '+key
    assert len({file_sha256(a.run/f'eval_{arm}/episodes.json') for arm in ARMS})==1
    atomic_json(dict(same_all_2470_episode_start_manifests=True,same_controller_and_runtime_source=True,
        weight_or_declared_ablation_differences_only=True),a.output/'paired_protocol_audit.json')
    paired=[]
    for idx in range(N):
        rec=dict(index=idx,map=results['rgb'][idx]['map_name'],group=groups[idx],instruction=raw[idx]['descriptions'][0])
        for arm in ARMS:rec[arm]=results[arm][idx]['metrics']
        paired.append(rec)
    summary['paired']={}
    comparisons=list(zip(ARMS,ARMS[1:]))+[('original_geometry','rgb'),('geometry','rgb')]
    for left,right in comparisons:
        correct=sum(not bool(x[left]['SR']) and bool(x[right]['SR']) for x in paired);damage=sum(bool(x[left]['SR']) and not bool(x[right]['SR']) for x in paired)
        delta=np.array([x[right]['SR']-x[left]['SR'] for x in paired]);maps=sorted({x['map'] for x in paired});rng=np.random.default_rng(42);boots=[]
        for _ in range(2000):
            chosen=rng.choice(maps,len(maps),replace=True);sample=[x[right]['SR']-x[left]['SR'] for m in chosen for x in paired if x['map']==m];boots.append(np.mean(sample))
        summary['paired'][left+'->'+right]=dict(correct=correct,damage=damage,net=correct-damage,SR_delta=float(delta.mean()),map_bootstrap_95CI=np.percentile(boots,[2.5,97.5]).tolist())
    atomic_json(summary,a.output/'metrics.json');atomic_json(paired,a.output/'paired_episodes.json');atomic_json(case_analysis,a.output/'failure_analysis.json')
    with (a.output/'episodes.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['arm','index','group','category','cause','RGB_updates','replans',*KEYS])
        for x in allrows:writer.writerow([x[k] for k in ('arm','index','group','category','cause','RGB_updates','replans')]+[x['metrics'][k] for k in KEYS])
    fig,axes=plt.subplots(1,3,figsize=(14,4));colors=['gray','#377eb8','#ff7f00','#4daf4a']
    for ax,metric in zip(axes,('SR','NE','inference_ms')):ax.bar([LABELS[x] for x in ARMS],[summary[x][metric] for x in ARMS],color=colors);ax.set_title(metric);ax.tick_params(axis='x',rotation=22)
    fig.tight_layout();fig.savefig(a.output/'metrics.png',dpi=150);plt.close(fig)
    history=json.loads((a.run/'core/progress.json').read_text())['history'];fig,ax=plt.subplots(figsize=(6,4));ax.plot([x['epoch'] for x in history],[x['train_loss'] for x in history],marker='o');ax.set(xlabel='Epoch',ylabel='Full TRAIN loss');fig.tight_layout();fig.savefig(a.output/'training_curve.png',dpi=150);plt.close(fig)
    # Include gains, damage and remaining failures; never only successes.
    selected=[]
    conditions=[('branch_gain',lambda i:not results['geometry'][i]['metrics']['SR'] and results['rgb'][i]['metrics']['SR']),('branch_damage',lambda i:results['geometry'][i]['metrics']['SR'] and not results['rgb'][i]['metrics']['SR']),
        ('training_gain',lambda i:not results['original_geometry'][i]['metrics']['SR'] and results['geometry'][i]['metrics']['SR']),('training_damage',lambda i:results['original_geometry'][i]['metrics']['SR'] and not results['geometry'][i]['metrics']['SR']),
        ('remaining_failure',lambda i:not results['rgb'][i]['metrics']['SR']),('success',lambda i:bool(results['rgb'][i]['metrics']['SR']))]
    for kind,pred in conditions:
        candidates=[i for i in range(N) if pred(i) and i not in [x[0] for x in selected]];picked=[];maps_seen=set()
        for i in candidates:
            map_name=results['rgb'][i]['map_name']
            if map_name not in maps_seen:picked.append(i);maps_seen.add(map_name)
            if len(picked)==2:break
        if len(picked)<2:picked.extend(i for i in candidates if i not in picked)
        selected.extend((i,kind) for i in picked[:2])
    if len(selected)<12:
        chosen={i for i,k in selected};selected.extend((i,'additional_case') for i in range(N) if i not in chosen and len(selected)<12)
    selected=selected[:12]
    renderer=CityNavObservedViewRenderer(a.data_root/'rgbd',a.output/'render',384);gallery=[];instance_stats=[]
    for idx,kind in selected:
        target=np.array(raw[idx]['target_positions'][0][:2]);map_name=results['rgb'][idx]['map_name'];fig,axes=plt.subplots(1,4,figsize=(17,4));case=a.output/f'case_{idx:04d}';case.mkdir(exist_ok=True)
        maxima=[]
        for arm in ARMS:
            files=sorted((a.run/f'eval_{arm}/belief_steps/{idx:05d}').glob('*.npz'))
            if files:
                z=np.load(files[-1]);key='geometry' if arm in ('original_geometry','geometry') else 'relation' if arm=='relation' else 'full';pp=np.exp(z[key]-z[key].max());maxima.append(float(pp.max()/pp.sum()))
        common_max=max(maxima) if maxima else 1.
        for ax,arm in zip(axes,ARMS):
            record=results[arm][idx];poses=np.array(record['poses']);root=a.run/f'eval_{arm}/belief_steps/{idx:05d}';files=sorted(root.glob('*.npz'))
            if files:
                field=np.load(files[-1]);public=json.loads((root/'public_record.json').read_text());b=np.array(public['world_bounds']);key='geometry' if arm in ('original_geometry','geometry') else 'relation' if arm=='relation' else 'full';prob=np.exp(field[key]-field[key].max());prob/=prob.sum()
                ax.imshow(prob.reshape(28,28),origin='lower',extent=[b[0],b[2],b[1],b[3]],vmin=0,vmax=common_max)
                for lm in public['landmarks']:
                    contour=b[:2]+np.array(lm['contour'])*(b[2:]-b[:2]);contour=np.vstack((contour,contour[0]));ax.plot(*contour.T,'c-',lw=.7)
            ax.plot(poses[:,0],poses[:,1],'w.-',lw=1);ax.scatter(*target,c='red',marker='x',s=60);ax.scatter(*poses[0,:2],c='lime',s=25);ax.set_title(f'{LABELS[arm]}\nSR={record["metrics"]["SR"]:.0f}, NE={record["metrics"]["NE"]:.1f}m')
            ax.add_patch(plt.Circle(target,20,fill=False,color='red',ls='--',alpha=.5))
            if files:ax.set_xlim(b[0],b[2]);ax.set_ylim(b[1],b[3])
        fig.suptitle(f'{idx} {kind}: '+raw[idx]['descriptions'][0][:120],fontsize=9);fig.tight_layout();fig.savefig(case/'belief_trajectories.png',dpi=140);plt.close(fig)
        files=sorted((a.run/f'eval_rgb/belief_steps/{idx:05d}').glob('*.npz'));rgbfiles=[]
        if files:
            inds=sorted(set([0,len(files)//2,len(files)-1]))
            public=json.loads((a.run/f'eval_rgb/belief_steps/{idx:05d}/public_record.json').read_text());b=np.array(public['world_bounds'])
            fig,axes=plt.subplots(1,len(inds),figsize=(5*len(inds),5));axes=np.atleast_1d(axes)
            for ax,j in zip(axes,inds):
                fields=np.load(files[j]);obs=renderer.current_rgbd(map_name,fields['pose'],GROUND_LEVEL[map_name]);sha=hashlib.sha256(obs['bgr'].tobytes()).hexdigest();assert sha==str(fields['rgb_SHA256']), 'Re-render must equal actual rollout RGB'
                import cv2
                name=f'RGB_{j:02d}.png';cv2.imwrite(str(case/name),obs['bgr']);rgbfiles.append(name);ax.imshow(cv2.cvtColor(obs['bgr'],cv2.COLOR_BGR2RGB));ax.set_title(f'Step{j}, z={fields["pose"][2]:.1f}m, yaw={fields["pose"][3]:.2f}');ax.axis('off')
                for lm in public['landmarks']:
                    contour=b[:2]+np.asarray(lm['contour'])*(b[2:]-b[:2]);contour=np.vstack((contour,contour[0]));px=world_to_frame(contour,fields['footprint'],obs['bgr'].shape);ax.plot(px[:,0],px[:,1],'c-',lw=.8)
                gp=world_to_frame(target[None],fields['footprint'],obs['bgr'].shape)[0]
                if (gp>=0).all() and (gp<=383).all():ax.scatter(*gp,c='red',marker='x',s=65)
                diagnostics=results['rgb'][idx]['step_diagnostics']
                if j<len(diagnostics):
                    point=world_to_frame(np.array(diagnostics[j]['selected_goal_xy'])[None],fields['footprint'],obs['bgr'].shape)[0]
                    if (point>=0).all() and (point<=383).all():ax.scatter(*point,c='blue',marker='+',s=65)
                ax.set_xlim(-.5,383.5);ax.set_ylim(383.5,-.5)
            fig.tight_layout();fig.savefig(case/'RGB_timeline.png',dpi=130);plt.close(fig)
        gallery.append(dict(index=idx,type=kind,images=[f'case_{idx:04d}/belief_trajectories.png']+([f'case_{idx:04d}/RGB_timeline.png'] if files else []),native_images=[f'case_{idx:04d}/{name}' for name in rgbfiles]))
    atomic_json(gallery,a.output/'cases.json')
    # Aggregate spatial (not pixel-instance) corrections from saved per-step Beliefs.
    for arm in ARMS:
        corrections=damages=0;shift=[];prob_delta=[];centroid_shifts=[];variation=[];states=0;raw_correct_episodes=set();raw_correct_not_selected=set()
        for idx in range(N):
            root=a.run/f'eval_{arm}/belief_steps/{idx:05d}';files=sorted(root.glob('*.npz'))
            if not files:continue
            public=json.loads((root/'public_record.json').read_text());b=np.array(public['world_bounds']);target=np.array(raw[idx]['target_positions'][0][:2])
            for path in files:
                field=np.load(path);world=b[:2]+field['grid']*(b[2:]-b[:2]);distance=np.linalg.norm(world-target,axis=-1);g=int(field['geometry'].argmax());f=int(field['full'].argmax());corrections+=int(distance[g]>20 and distance[f]<=20);damages+=int(distance[g]<=20 and distance[f]>20);shift.append(float(np.linalg.norm(world[g]-world[f])))
                if distance[f]<=20:
                    raw_correct_episodes.add(idx)
                    if distance[int(field['selected_ids'][0])]>20:raw_correct_not_selected.add(idx)
                mass=[];probabilities=[]
                for key in ('geometry','full'):
                    pp=np.exp(field[key]-field[key].max());pp/=pp.sum();mass.append(float(pp[distance<=20].sum()));probabilities.append(pp)
                centroid_shifts.append((probabilities[1]-probabilities[0])@world)
                variation.append(float(np.abs(probabilities[1]-probabilities[0]).sum()/2))
                prob_delta.append(mass[1]-mass[0]);states+=1
        instance_stats.append(dict(arm=arm,states=states,position20m_corrections=corrections,position20m_damage=damages,mean_Goal_shift_m=float(np.mean(shift)),mean_GT20m_mass_delta=float(np.mean(prob_delta)),
            mean_probability_centroid_shift_m=float(np.mean(np.linalg.norm(centroid_shifts,axis=-1))),mean_centroid_XY_delta=np.mean(centroid_shifts,axis=0).tolist(),mean_total_variation=float(np.mean(variation)),not_pixel_instance_accuracy=True,
            failed_with_any_raw_correct_belief=sum(not bool(results[arm][i]['metrics']['SR']) for i in raw_correct_episodes),raw_correct_but_selected_wrong_episodes=sorted(raw_correct_not_selected)))
    atomic_json(instance_stats,a.output/'belief_shift.json')
    relation_scale=[]
    for idx in range(N):
        paths=sorted((a.run/f'eval_relation/belief_steps/{idx:05d}').glob('*.npz'))
        if not paths:continue
        item=np.load(paths[0]);g=item['geometry'];r=item['relation'];gap=float(np.sort(g)[-1]-np.sort(g)[-2]);delta=float(np.abs(r-g).max())
        relation_scale.append(dict(index=idx,Geometry_top2_gap=gap,Relation_max_abs_delta=delta,top1_changed=bool(g.argmax()!=r.argmax()),gap_exceeds_twice_delta=gap>2*delta))
    atomic_json(relation_scale,a.output/'relation_scale_per_episode.json')
    atomic_json(dict(episodes=len(relation_scale),median_Geometry_gap=float(np.median([x['Geometry_top2_gap'] for x in relation_scale])),median_Relation_max_delta=float(np.median([x['Relation_max_abs_delta'] for x in relation_scale])),
        guaranteed_same_top1=sum(x['gap_exceeds_twice_delta'] for x in relation_scale),changed_top1=sum(x['top1_changed'] for x in relation_scale)),a.output/'relation_scale_summary.json')
    for src,name in [(a.run/'core/manifest.json','core_manifest.json'),(a.run/'core/progress.json','core_progress.json'),(a.run/'visual/manifest.json','visual_manifest.json'),(a.run/'visual/progress.json','visual_progress.json'),(a.run/'runtime_smoke/smoke.json','runtime_smoke.json')]:shutil.copyfile(src,a.output/name)
    artifacts={str(a.run/'core/last.pt'):file_sha256(a.run/'core/last.pt'),str(a.run/'visual/last.pt'):file_sha256(a.run/'visual/last.pt')};atomic_json(artifacts,a.output/'checkpoint_SHA256.json')
    lines=['# UnifiedPGVB完整TRAIN联合训练与完整Val-Seen闭环','', '仅seed42。所有模型及参数在Val评分前冻结，不用Val选择Checkpoint或调整参数；Test未访问。','', '## 完整闭环','|方法|N|SR|OSR|SPL|NE(m)|nDTW|时延ms|','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        x=summary[arm];lines.append(f'|{LABELS[arm]}|{x["N"]}|{100*x["SR"]:.2f}%|{100*x["OSR"]:.2f}%|{x["SPL"]:.4f}|{x["NE"]:.3f}|{x["nDTW"]:.4f}|{x["inference_ms"]:.2f}|')
    lines+=['','原始Geometry是额外完整基线；其余三组共享训练后的Geometry。Relation与RGB的Geometry/Relation权重逐张量相同，只有PGVB阶段权重变化。全部2470任务计入，空参考和推理异常不删除。','', '## 逐任务配对与不确定性','```json',json.dumps(summary['paired'],ensure_ascii=False,indent=2),'```','单随机种子不能证明训练随机性稳健；地图分组bootstrap仅描述这批评测任务的采样不确定性，不等于多种子显著性。','', '## 训练范围', 'Phase1：完整Revised TRAIN 21878条×5轮，0转换跳过。Geometry和AerialVG-inspired Relation联合训练。不是官方AerialVG预训练参数。BERT/SigLIP2冻结，既有PGVB在Phase1冻结；不是全视觉骨干端到端训练。', 'Phase2：249帧、205条fit指令的真实当前RGB缓存训练PGVB三轮，Geometry与Relation冻结。支持局部Anchor的筛选与运行时一致，不制造画外视觉证据；此子集较小并偏向直接关系，不宣称全样本视觉训练。', '原始Stop概率阈值0.3、到达守卫1mm、StableGoal margin0.35/连续2帧、20步Horizon及20m成功半径均不变。', '', '## 失败分析', '互斥主分类仅按轨迹/Goal可证事实：推理错误、从未有正确Goal、曾正确Goal但未到达、进入20m后未最终成功。它们不能单凭NE区分像素实例错误、错误Polygon绑定或遮挡；无独立标注时原因未知。', '详细分类与各分层在failure_analysis.json、metrics.json；保存了首次进入20m时Goal、Stop概率、到达守卫及高度守卫，不自动修改Stop。', 'Belief 20m纠正/破坏是位置指标，不等于真实实例匹配准确率；公开Polygon是地图弱监督，不是人工像素GT。', '', '## 决策', f'Geometry→Relation净纠正{summary["paired"]["geometry->relation"]["net"]:+d}条；Relation→RGB净纠正{summary["paired"]["relation->rgb"]["net"]:+d}条。完整负结果同样保留，不根据这次Val结果追加参数搜索。', '公开Polygon与语言作用域、RGB确认是否真能识别目标、观察不足及原控制器Goal保持均需分别看实际轨迹；不能从Relation残差非零就宣布理解关系。', '', '## 复现与图像说明','Checkpoint路径与SHA256见checkpoint_SHA256.json；冻结协议见freeze_*.json；CPU与GPU Smoke证据均随报告保存。所有案例RGB由实际轨迹位姿重渲染并逐帧SHA校验与在线真实观测一致，384×384原始观测，未冒充高清。地图轴为工程World XY，不声称其已完成地理北向配准。', '所有图包含成功、破坏及剩余失败；红叉为事后GT目标位置、青色为公开Polygon、白线为实际预测轨迹。']
    (a.output/'REPORT_zh.md').write_text('\n'.join(lines)+'\n')
    with (a.output/'REPORT_zh.md').open('a') as doc:
        doc.write('\n## 分层完整结果\n\n直接/间接为自动语言路径分组，非人工关系真值；另列历史严格56条子集。各分母不混用。\n\n|分组|方法|N|SR|OSR|NE|\n|---|---|---:|---:|---:|---:|\n')
        for arm in ARMS:
            for group,d in summary[arm]['strata'].items():
                if d['N']:doc.write(f'|{group}|{LABELS[arm]}|{d["N"]}|{100*d["SR"]:.2f}%|{100*d["OSR"]:.2f}%|{d["NE"]:.3f}|\n')
        doc.write('\n## 失败与实际控制\n\n|方法|从未正确Goal|曾正确Goal但未进入|进入后未最终成功|推理错误|正确PolicyStop|重规划|RGB更新帧|\n|---|---:|---:|---:|---:|---:|---:|---:|\n')
        for arm in ARMS:
            counts=summary[arm]['failure_causes'];correct_stop=sum(x['metrics']['SR'] and results[arm][x['index']]['termination']=='policy_stop' for x in case_analysis[arm]);doc.write(f'|{LABELS[arm]}|{counts.get("never_correct_navigation_goal",0)}|{counts.get("correct_goal_ever_but_not_reached",0)}|{counts.get("entered_20m_but_not_final_success",0)}|{counts.get("inference_error",0)}|{correct_stop}|{summary[arm]["replans"]}|{summary[arm]["RGB_updates"]}|\n')
        doc.write('\n错误Goal处停止不能自动归因为Stop网络失败；需结合首次20m进入时的Goal与守卫证据。没有可靠像素实例标注，Polygon/RGB错配或真实目标不可见的原因保留未知。\n')
    html='<!doctype html><meta charset="utf-8"><title>UnifiedPGVB全量闭环</title><style>body{font-family:sans-serif;max-width:1600px;margin:auto}img{max-width:100%}</style><h1>UnifiedPGVB：完整Val-Seen</h1><a href="REPORT_zh.md">中文报告</a><img src="metrics.png"><img src="training_curve.png">'
    for case in gallery:
        html+=f'<h2>{case["index"]} {case["type"]}</h2><p>'+escape(raw[case['index']]['descriptions'][0])+'</p>'+''.join(f'<a href="{im}"><img src="{im}"></a>' for im in case['images'])+' '.join(f'<a href="{im}">原生RGB</a>' for im in case['native_images'])
    (a.output/'index.html').write_text(html);print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
