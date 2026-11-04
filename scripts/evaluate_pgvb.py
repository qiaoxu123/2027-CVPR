"""One frozen evaluation of city-disjoint PGVB with content/shape controls."""
import argparse,collections,json,time
from pathlib import Path
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from graphnav.checkpointing import atomic_json,atomic_save,file_sha256
from graphnav.landmark_visual_only import visual_only_world_belief
from graphnav.polygon_guided_visual import PolygonGuidedVisualBelief
from graphnav.pgvb_training import pgvb_batch
from scripts.audit_visual_only_landmarks import prediction_index


def aggregate(rows):
    if not rows:return dict(n=0)
    return dict(n=len(rows),correct=sum(x['success'] for x in rows),R20=float(np.mean([r['success'] for r in rows])),
        NE=float(np.mean([r['ne'] for r in rows])),polygon_mass=float(np.mean([r['mass'] for r in rows])),
        proxy_top1_hits=sum(r['proxy_top1_label']==1 for r in rows),proxy_top1_unknown=sum(r['proxy_top1_label']==-1 for r in rows),
        proxy_top5_hits=sum(r['proxy_top5'] for r in rows),maps=len({r['map_name'] for r in rows}),
        named_groups=len({r['query_key'] for r in rows}))


def pair_metrics(data,scores):
    grouped=collections.defaultdict(list)
    for index in scores:
        r=data['records'][index];grouped[(r['frame_id'],r['type'])].append(index)
    results=[]
    for (fid,typ),indices in grouped.items():
        for i in sorted(indices):
            for j in sorted(indices):
                if j<=i or j not in data['records'][i]['hard_negative_indices']:continue
                a=data['records'][i]['labels'].numpy()==1;b=data['records'][j]['labels'].numpy()==1
                ma=float(scores[i][a].mean()-scores[i][b].mean());mb=float(scores[j][b].mean()-scores[j][a].mean())
                results.append(dict(frame_id=fid,type=typ,first=i,second=j,margin_a=ma,margin_b=mb,
                    both_correct=ma>1e-6 and mb>1e-6,query_score=((ma>1e-6)+.5*(abs(ma)<=1e-6)+(mb>1e-6)+.5*(abs(mb)<=1e-6))/2,
                    tied_queries=int(abs(ma)<=1e-6)+int(abs(mb)<=1e-6)))
    return dict(summary={typ:dict(pairs=len(rr),frames=len({r['frame_id'] for r in rr}),strict_both_correct=sum(r['both_correct'] for r in rr),
        query_preference_accuracy=float(np.mean([r['query_score'] for r in rr])),tied_queries=sum(r['tied_queries'] for r in rr))
        for typ in ('Building','Road','all') if (rr:=[r for r in results if typ=='all' or r['type']==typ])},rows=results)


def main():
    p=argparse.ArgumentParser()
    for k in ('data','training','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    if a.output.exists():raise FileExistsError('Frozen evaluation cannot be overwritten')
    proof=json.loads((a.data/'manifest.json').read_text());freeze=json.loads((a.training/'freeze.json').read_text())
    if file_sha256(a.data/'cache.pt')!=proof['cache_sha256'] or file_sha256(a.training/'best.pt')!=freeze['best_sha256']:raise ValueError('Unsealed input')
    data=torch.load(a.data/'cache.pt',weights_only=True,map_location='cpu');state=torch.load(a.training/'best.pt',weights_only=True,map_location='cpu')
    for path,h in state['config']['code_sha256'].items():
        if file_sha256(path)!=h:raise ValueError('Training source changed: '+path)
    model=PolygonGuidedVisualBelief().cuda().eval().requires_grad_(False);model.load_state_dict(state['model']);a.output.mkdir(parents=True)
    all_rows=[];pairs={};maps={};timings={};subsets=('fit','dev')
    for subset in subsets:
        indices=[i for i,r in enumerate(data['records']) if r['subset']==subset]
        # Fit is descriptive only; same frozen checkpoint, never new tuning.
        for control in ('real','coverage_only','zero_rgb','shuffle_rgb','polygon_zero','wrong_polygon','wrong_name'):
            chosen=[i for i in indices if data['records'][i]['hard_negative_indices']] if control in ('wrong_polygon','wrong_name') else indices
            score_cache={};duration=[]
            with torch.inference_mode():
                for start in range(0,len(chosen),32):
                    ids=chosen[start:start+32];batch,labels,rr=pgvb_batch(data,ids,control=control)
                    torch.cuda.synchronize();t=time.perf_counter()
                    score=model(**batch)
                    if control=='coverage_only':score=torch.zeros_like(score)
                    uv=torch.stack([data['frames'][r['frame_id']]['local_uv'] for r in rr]).cuda()
                    field=visual_only_world_belief(local_uv=uv,scores=score,valid=batch['valid'])
                    torch.cuda.synchronize();duration.append((time.perf_counter()-t)/len(ids))
                    for j,(index,r) in enumerate(zip(ids,rr)):
                        f=data['frames'][r['frame_id']];world=f['global_world'].numpy();valid=f['valid'].numpy()
                        prob=field['probabilities'][j].cpu().numpy();covered=field['covered'][j].cpu().numpy();s=score[j].cpu().numpy()
                        if not bool(field['has_view'][j]):raise ValueError('Eligible cache unexpectedly empty')
                        best=prediction_index(prob,covered,world);ne=float(r['distance'][best])
                        localuv=f['local_uv'].numpy();localxy=np.asarray(f['bounds'])[:2]+localuv*(np.asarray(f['bounds'])[2:]-np.asarray(f['bounds'])[:2])
                        locprob=torch.from_numpy(s).masked_fill(~f['valid'],float('-inf')).softmax(-1).numpy()
                        local=prediction_index(locprob,valid,localxy);top5=np.argsort(-np.where(valid,s,-np.inf))[:5]
                        item=dict(index=index,subset=subset,control=control,frame_id=r['frame_id'],query_key=r['query_key'],type=r['type'],map_name=f['map_name'],
                            ne=ne,success=ne<=20,mass=float(prob@r['area_fraction'].numpy()),world_xy=world[best].tolist(),
                            proxy_top1_label=int(r['labels'][local]),proxy_top5=bool((r['labels'][top5]==1).any()),actual_visibility='unknown',human_pixel_gt=False)
                        all_rows.append(item);score_cache[index]=s
                        if subset=='dev' and control in ('real','coverage_only','zero_rgb','wrong_polygon'):
                            maps[(index,control)]=dict(probability=torch.from_numpy(prob),covered=torch.from_numpy(covered))
            pairs[subset+'::'+control]=pair_metrics(data,score_cache);timings[subset+'::'+control]=float(np.mean(duration))
    summary={s:{c:{typ:aggregate([r for r in all_rows if r['subset']==s and r['control']==c and (typ=='all' or r['type']==typ)]) for typ in ('all','Building','Road')}
        for c in ('real','coverage_only','zero_rgb','shuffle_rgb','polygon_zero','wrong_polygon','wrong_name')} for s in subsets}
    changes={}
    real={r['index']:r for r in all_rows if r['control']=='real' and r['subset']=='dev'}
    for control in ('coverage_only','zero_rgb','polygon_zero','wrong_polygon','wrong_name'):
        other={r['index']:r for r in all_rows if r['control']==control and r['subset']=='dev'};keys=sorted(real.keys()&other.keys())
        changes[control]=dict(paired=len(keys),real_corrected=sum(real[i]['success'] and not other[i]['success'] for i in keys),
            real_lost=sum(not real[i]['success'] and other[i]['success'] for i in keys),
            real_NE=float(np.mean([real[i]['ne'] for i in keys])),control_NE=float(np.mean([other[i]['ne'] for i in keys])))
    output=dict(summary=summary,pairs={k:v['summary'] for k,v in pairs.items()},dev_paired=changes,head_seconds_per_query=timings,
        checkpoint_sha256=freeze['best_sha256'],best_epoch=state['next_epoch'],training_config=state['config'],
        model_frozen_before_evaluation=True,no_formal_val_test=True,no_navigation=True,weak_pixel_labels=True,
        evaluation_code_sha256=file_sha256(__file__))
    atomic_json(output,a.output/'metrics.json');atomic_json(all_rows,a.output/'predictions.json');atomic_json(pairs,a.output/'same_image_pairs.json')
    atomic_save(maps,a.output/'visual_maps.pt')
    picks=[]
    for typ in ('Building','Road'):
        for success in (True,False):
            row=next((r for r in real.values() if r['type']==typ and r['success']==success),None)
            if row:picks.append(row)
    # At least one same-image multi-query case independent of outcome.
    for typ in ('Building','Road'):
        pair=next((r for r in pairs['dev::real']['rows'] if r['type']==typ),None)
        if pair:
            for idx in (pair['first'],pair['second']):
                if all(r['index']!=idx for r in picks):picks.append(real[idx])
    case_info=[]
    for ci,row in enumerate(picks):
        index=row['index'];r=data['records'][index];f=data['frames'][r['frame_id']];q=data['queries'][r['query_key']]
        if file_sha256(f['image_path'])!=f['frame']['image_sha256']:raise ValueError('RGB changed')
        bgr=cv2.imread(f['image_path']);fig,axes=plt.subplots(1,4,figsize=(16,4));axes[0].imshow(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB));axes[0].set_title('Current real RGB');axes[0].axis('off')
        # Polygon overlays are only in diagnostic figures, never input RGB.
        from graphnav.landmark_projection import world_to_frame
        for contour in q['contours']:
            px=world_to_frame(np.asarray(contour),f['frame']['footprint_world'],bgr.shape);axes[0].plot(px[:,0],px[:,1],'c-',lw=.8)
        bounds=f['bounds'];extent=[bounds[0],bounds[2],bounds[1],bounds[3]]
        for ax,c,title in zip(axes[1:],('coverage_only','zero_rgb','real'),('Coverage only','RGB zero','PGVB')):
            packed=maps[(index,c)];values=np.where(packed['covered'].numpy(),packed['probability'].numpy(),np.nan).reshape(28,28)
            ax.imshow(values,origin='lower',extent=extent,cmap='magma')
            for contour in q['contours']:
                xy=np.asarray(contour);ax.plot(xy[:,0],xy[:,1],'c-',lw=.8)
            item=next(x for x in all_rows if x['index']==index and x['control']==c);ax.scatter(*item['world_xy'],color='red',s=20)
            ax.set_title(f'{title}: NE {item["ne"]:.1f}m, mass {item["mass"]:.3f}',fontsize=9)
        fig.suptitle(q['name']+' | '+r['type']+' | public Polygon weak reference, not pixel GT');fig.tight_layout()
        fig.savefig(a.output/f'case_{ci}.png',dpi=150);plt.close(fig);cv2.imwrite(str(a.output/f'case_{ci}_rgb.png'),bgr)
        case_info.append(dict(index=index,query_key=r['query_key'],type=r['type'],frame_id=r['frame_id'],file=f'case_{ci}.png'))
    atomic_json(case_info,a.output/'cases.json');print(json.dumps(output,indent=2),flush=True)


if __name__=='__main__':main()
