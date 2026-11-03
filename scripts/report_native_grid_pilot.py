"""Frozen native-grid evidence and actual RGB proxy-ranking cases."""
import argparse
import html
import json
from pathlib import Path
import shutil
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,file_sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','annotations','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();r=a.root;a.output.mkdir(parents=True,exist_ok=True)
    forward=r/'native_grid_weak27_forward_v1';scores=r/'native_grid_weak27_scores_v1';inter=r/'native_grid_weak27_rank_intervention_v1'
    summary=json.loads((scores/'summary.json').read_text());rows=json.loads((scores/'records_private.json').read_text());intervention=json.loads((inter/'results.json').read_text())
    pred=torch.load(forward/'predictions_public.pt',weights_only=True);queries=torch.load(r/'native_grid_weak27_selection_v1/inputs_public.pt',weights_only=True)
    boxes={x['frame_key']:x for x in json.loads(a.annotations.read_text())};rng=np.random.default_rng(0);maps=sorted({x['map_name'] for x in rows});bootstrap={}
    groups={m:[x for x in rows if x['map_name']==m] for m in maps}
    for metric in ('native_top1','goal_ne'):
        values=[]
        for _ in range(5000):
            sampled=[x for m in rng.choice(maps,len(maps),replace=True) for x in groups[m]]
            if metric=='native_top1':d=[int(x['methods']['Native576']['proxies']['cosine']['0']['top1'])-int(x['methods']['Native144']['proxies']['cosine']['0']['top1']) for x in sampled]
            else:d=[x['methods']['Native576']['goal_ne_m']-x['methods']['Native144']['goal_ne_m'] for x in sampled]
            values.append(np.mean(d)*(100 if metric=='native_top1' else 1))
        bootstrap[metric]=dict(map_group_bootstrap95=np.quantile(values,[.025,.975]).tolist(),warning='Selected TRAIN AI appearance proxies only; intervals omit annotation error and selection bias.')
    atomic_json(bootstrap,a.output/'paired_uncertainty.json')
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(3);width=.22
    for i,(name,score) in enumerate([('144 native','cosine'),('576 native','cosine'),('576 learned','visual_score')]):
        mode='Native144' if i==0 else 'Native576';v=summary['methods'][mode]['proxies'][score]['0'];axes[0].bar(x+(i-1)*width,[v['top1'],v['top5'],v['top15']],width,label=name)
    axes[0].set_xticks(x,['Top1','Top5','Top15']);axes[0].set_ylabel('AI weak-box center hits /27');axes[0].legend(fontsize=8)
    modes=['Native144','Native576','Native144Shuffle','Native576Shuffle'];axes[1].bar(np.arange(4),[summary['methods'][m]['goal_ne_m'] for m in modes]);axes[1].set_xticks(np.arange(4),['144','576','144 shuffled','576 shuffled']);axes[1].set_ylabel('Registered Goal NE (m)');axes[1].tick_params(axis='x',rotation=15)
    fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=160);plt.close(fig)
    # Selected AFTER scoring as descriptive cases, not outcome-based parameter tuning.
    common_gain=next(x['frame_key'] for x in rows if x['methods']['Native144']['proxies']['cosine']['0']['candidate_present'] and not x['methods']['Native144']['proxies']['cosine']['0']['top1'] and x['methods']['Native576']['proxies']['cosine']['0']['top1'])
    harm=next(x['frame_key'] for x in intervention['rows'] if x['results']['Native576']['Original']['success'] and not x['results']['Native576']['NativeRank']['success'])
    selected=list(dict.fromkeys([common_gain,'train_17051_0008',harm,'train_12163_0029']));page=['<meta charset="utf-8"><h1>冻结原生144/576候选：TRAIN27弱外观代理</h1><p>不是像素金标准、真实实例准确率或导航SR。青色框为AI弱外观框；黄点原生相似度Top5，紫点学习评分Top5。</p><a href="REPORT_zh.md">中文报告</a><br><img width="1000" src="comparison.png">'];cases=[]
    for key in selected:
        q=queries[key];frame=q['frame'];path=Path(q['observations'])/frame['image'];assert file_sha256(path)==frame['image_sha256'];image=cv2.cvtColor(cv2.imread(str(path)),cv2.COLOR_BGR2RGB);box=boxes[key]['boxes_px'][0]
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,mode in zip(axes[:2],('Native144','Native576')):
            ax.imshow(image);ax.add_patch(Rectangle((box[0],box[1]),box[2]-box[0],box[3]-box[1],fill=False,edgecolor='cyan',linewidth=1.5))
            item=pred[key][mode]
            for score,color,marker in [('cosine','yellow','o'),('visual_score','magenta','x')]:
                order=item[score].masked_fill(~item['valid'],float('-inf')).argsort(descending=True)[:5];xy=item['pixels'][order].numpy();ax.scatter(xy[:,0],xy[:,1],c=color,marker=marker,s=14)
            record=next(x for x in rows if x['frame_key']==key)['methods'][mode];ax.set_title(mode+' / Goal NE %.2fm'%record['goal_ne_m']);ax.set_xlim(0,384);ax.set_ylim(384,0)
        ax=axes[2]
        for mode,color in [('Native144','blue'),('Native576','purple')]:
            record=next(x for x in rows if x['frame_key']==key)['methods'][mode];ax.scatter(*record['goal_world'],c=color,label=mode)
        # Private Goal shown only AFTER predictions; source labels never enter model.
        label=next(x for x in map(json.loads,(Path(q['observations'])/'labels_private.jsonl').read_text().splitlines()) if x.get('frame_key')==key)
        rank_case=next(x for x in intervention['rows'] if x['frame_key']==key)
        rank_goal=rank_case['results']['Native576']['NativeRank']['goal_world'];ax.scatter(*rank_goal,c='green',marker='^',label='576 native rank (diagnostic)')
        target=label['target_world'][:2];ax.scatter(*target,c='red',marker='x',label='TRAIN Goal posthoc');foot=np.asarray(frame['footprint_world']);foot=np.vstack([foot,foot[0]]);ax.plot(foot[:,0],foot[:,1],c='gray');ax.set_aspect('equal');ax.legend(fontsize=8);ax.set_title('Fixed world XY / actual footprint')
        fig.suptitle(key);fig.tight_layout();name=key+'.png';fig.savefig(a.output/name,dpi=150);plt.close(fig)
        cases.append(dict(frame_key=key,instruction=frame['instruction'],source_image_sha256=frame['image_sha256'],pixel_gold=False))
        page+=['<h2>'+html.escape(key)+'</h2><p>'+html.escape(frame['instruction'])+'</p><a href="'+name+'"><img width="1300" src="'+name+'"></a>']
    atomic_json(cases,a.output/'cases.json');(a.output/'index.html').write_text('\n'.join(page))
    for src,name in [(scores/'summary.json','summary.json'),(scores/'records_private.json','records_private.json'),(inter/'results.json','ranking_intervention.json'),(forward/'run_manifest.json','run_manifest.json'),(forward/'prediction_seal.json','prediction_seal.json'),(forward/'runtime.json','runtime.json'),(inter/'normal_prediction_seal.json','rank_prediction_seal.json')]:shutil.copyfile(src,a.output/name)
    extra_losses=0
    for key,q in queries.items():
        for mode in ('Native144','Native576'):
            item=pred[key][mode];uv=(item['world']-q['world_bounds'][:2])/(q['world_bounds'][2:]-q['world_bounds'][:2])
            extra_losses+=int((item['valid']&~((uv>=0).all(-1)&(uv<=1).all(-1))).sum())
    atomic_json(dict(world_bounds_filter_extra_losses=extra_losses,proof='All depth-valid points in both lattices lie within public world bounds on selected27.',pooling_shape_min_cosine_usable=False,reason='v1 min includes unsupported zero vectors; predictions use EXACT same dense feature subset, no standalone sparse pooling features.'),a.output/'diagnostic_notes.json')

if __name__=='__main__':main()
