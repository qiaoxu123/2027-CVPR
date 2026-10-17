"""Publication figures from real CityNav records and archived executed traces."""
import json,textwrap,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon,Circle
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.closedloop import PublicEpisode
from graphnav.a3_runtime import public_static_record
from graphnav.dynamic_runtime import NativeObservationBuilder
from graphnav.workspace_paths import WORKSPACE,PROJECT
from scripts.polygon_ablation_navigation import resources
from scripts.geombelief_map_experiments import OUT,OLD,HERE

def get_row(path,eid):
    with Path(path).open() as f:
        for line in f:
            r=json.loads(line)
            if r['episode_id']==eid:return r
    raise KeyError(eid)

def draw_map(ax,record,target,trajectories=(),goals=()):
    b=np.asarray(record['world_bounds']);lo,span=b[:2],b[2:]-b[:2]
    labelled=set()
    for i,l in enumerate(record['landmarks']):
        xy=lo+np.asarray(l['contour'])*span
        ax.add_patch(Polygon(xy,closed=True,facecolor='#93b6ca',edgecolor='#376078',alpha=.45))
        if l['name'] not in labelled:
            center=xy.mean(0);ax.text(*center,l['name'],fontsize=7,ha='center',clip_on=True);labelled.add(l['name'])
    ax.scatter(*target[:2],marker='*',s=140,c='#269653',label='Ground-truth goal',zorder=7)
    ax.add_patch(Circle(target[:2],20,fill=False,color='#269653',ls='--',lw=1,label='20 m success region'))
    for label,row,color in trajectories:
        poses=np.asarray(row['poses']);ax.plot(poses[:,0],poses[:,1],'-o',c=color,markersize=2,lw=1.5,label=label)
        ax.scatter(*poses[0,:2],c='black',marker='s',s=18,zorder=6)
    for label,goal,color in goals:ax.scatter(*goal,marker='x',s=70,c=color,label=label,zorder=8)
    ax.set(xlabel='World X (m)',ylabel='World Y (m)',aspect='equal');ax.autoscale_view();ax.legend(fontsize=6,loc='best');ax.grid(alpha=.2)

def main():
    out=OUT/'figures';out.mkdir(parents=True,exist_ok=True)
    plan=json.loads((OLD/'frozen_plan.json').read_text())
    old=json.loads((HERE/'visual_residual_unseen_validation/metrics.json').read_text())
    examples=old['typical_episodes'];chosen=[]
    for outcome in ('improved','degraded'):
        choices=[x for x in examples if x['split']=='val_unseen' and x['outcome']==outcome]
        if not choices:choices=[x for x in examples if x['outcome']==outcome]
        if choices:chosen.append(('visual_'+outcome,choices[0]['split'],choices[0]['episode_id']))
    # First formal Unseen false-stop in immutable episode order, not worst case.
    baseline=PROJECT/'GraphNav/runs/graphnav_v1_stage1/full_unseen/predictions.jsonl'
    with baseline.open() as f:
        fail=next(json.loads(l) for l in f if json.loads(l)['metrics']['false_stop'])
    chosen.append(('false_stop','val_unseen',fail['episode_id']))
    manifest=[];resource={}
    for name,split,eid in chosen:
        if split not in resource:
            data,catalog,base,controller,protocol,ground=resources(plan,split)
            resource[split]=(data,catalog,base,ground,NativeObservationBuilder(base,WORKSPACE/'DATA/sbfnav-data/rgbd',out/'rendered'))
        data,catalog,base,ground,builder=resource[split]
        record=next(data[i] for i in range(len(data)) if list(data[i].episode_id)==eid)
        public=PublicEpisode(record.map_name,record.instruction,record.start_pose.copy(),tuple(catalog.referenced_landmarks(record)),tuple(catalog.referenced_landmark_names(record)))
        static=public_static_record(public,base.transform(public.map_name),catalog,max_landmarks=16)
        if name.startswith('visual'):
            stem='calibrated_residual_full_'+('seen' if split=='val_seen' else 'unseen')
            real_path=HERE/'visual_evaluations'/f'{stem}_real/predictions.jsonl'
            other_path=HERE/'visual_evaluations'/f'{stem}_no-belief/predictions.jsonl'
            real=get_row(real_path,eid);other=get_row(other_path,eid)
            paths=[('RGB residual',real,'#2373ac'),('No residual',other,'#df8543')]
            goals=[('Residual final goal',real['step_diagnostics'][-1]['selected_goal_xy'],'#2373ac'),('No-residual final goal',other['step_diagnostics'][-1]['selected_goal_xy'],'#df8543')]
        else:
            real_path=baseline;other_path=None;real=fail;paths=[('Executed trajectory',real,'#b54452')];goals=[('Predicted stop goal',real['step_diagnostics'][-1]['selected_goal_xy'],'#b54452')]
        # Display the last actually observed pose, not an unobserved final action.
        step=len(real['step_diagnostics'])-1
        obs=builder.build_from_poses(public,np.asarray(real['poses'])[:step+1]).current_observation
        fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
        draw_map(axes[0],static,record.target_position,paths,goals)
        axes[0].set_title('Public map geometry and executed paths')
        axes[1].imshow(obs['bgr'][...,::-1]);axes[1].axis('off');axes[1].set_title(f'Visited orthophoto crop at decision {step}')
        fig.suptitle('\n'.join(textwrap.wrap(record.instruction,110))+'\nEpisode: '+':'.join(map(str,eid)),fontsize=9)
        for suffix in ('png','pdf'):fig.savefig(out/f'{name}.{suffix}',dpi=180)
        plt.close(fig)
        # Independent input overview, no invented observations or trajectories.
        if name=='visual_improved':
            obs0=builder.build_from_poses(public,np.asarray(real['poses'])[:1]).current_observation
            fig,axes=plt.subplots(1,2,figsize=(11,4.8),constrained_layout=True)
            draw_map(axes[0],static,record.target_position);axes[0].set_title('Language-grounded public landmark geometry')
            axes[1].imshow(obs0['bgr'][...,::-1]);axes[1].axis('off');axes[1].set_title('Initial-pose orthophoto crop')
            fig.suptitle('\n'.join(textwrap.wrap(record.instruction,110)),fontsize=9)
            for suffix in ('png','pdf'):fig.savefig(out/f'language_map_goal.{suffix}',dpi=180)
            plt.close(fig)
        manifest.append(dict(case=name,split=split,episode_id=eid,instruction=record.instruction,map_name=record.map_name,
            image_step=step,image_pose=real['poses'][step],trace_path=str(real_path),trace_sha256=file_sha256(real_path),
            comparator_trace=str(other_path) if other_path else None,actual_metrics=real['metrics'],
            comparator_metrics=other['metrics'] if other_path else None,figure=str(out/f'{name}.png'),public_map=static,
            selection='First archived typical example of the requested outcome; formal false-stop is first failed policy_stop in fixed episode order. Illustrative, not representative causal evidence.'))
    atomic_json(manifest,OUT/'qualitative_manifest.json')
    text='# GeomBelief qualitative cases\n\nImages are crops of the real CityNav orthophoto at actual executed poses. The environment uses an orthographic crop and a surface-height-derived vertical-depth proxy, not a perspective occlusion renderer or a separately captured onboard photograph. Only the visited crop and its support metadata enter the policy. Map polygons, language, goal annotations and trajectories come from recorded public inputs and evaluation-only GT. English figure labels; no invented paths.\n\n'
    for r in manifest:text+=f'## {r["case"]}\n\nEpisode `{r["episode_id"]}`; {r["split"]}; NE {r["actual_metrics"]["NE"]:.2f} m. Source trace and SHA: qualitative_manifest.json.\n\n![{r["case"]}](figures/{r["case"]}.png)\n\n'
    text+='## Correct versus shuffled geometry belief\n\nPending all static models being frozen. The heatmap uses predeclared Seed0 and the first publicly changed geometry query, never a favorable seed or outcome. No fabricated probability field is substituted for missing logits.\n'
    previous=(OUT/'qualitative_cases.md').read_text() if (OUT/'qualitative_cases.md').exists() else ''
    marker='## Correct versus shuffled geometry belief'
    if marker in previous and 'belief_figure_manifest.json' in previous:
        text=text.split(marker)[0]+marker+previous.split(marker,1)[1]
    (OUT/'qualitative_cases.md').write_text(text)

if __name__=='__main__':main()
