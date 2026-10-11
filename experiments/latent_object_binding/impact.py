"""Actual score changes at the frozen output interface; no gain sweep."""
import json,numpy as np,torch
from experiments.semantic_polygon_grounding.audit import save
from .resources import OUT
from .evaluate import materialize
from .inference import posterior

def main():
    torch.set_num_threads(1);maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True);weights=torch.load(OUT/'training/rgb/best.pt',weights_only=True)['weights'];rows=[];summary={}
    with torch.no_grad():
        for split in ('train','dev'):
            data=torch.load(OUT/'observed'/f'{split}_public.pt',weights_only=True,mmap=True);labs={l['sample_id']:l for l in torch.load(OUT/'observed'/f'{split}_labels.pt',weights_only=True)};rs=[]
            for d in data:
                meta=d['metadata'];g=maps[meta['map_name']];p={v:posterior(materialize(d['versions'][v],g,d['a0']),weights) for v in ('real','black','shuffled')};lab=labs[meta['sample_id']];tid=g['ids'].index(lab['target_object_id']);world=g['world'].numpy();dist=np.linalg.norm(world-np.array(lab['target_world']),axis=-1);mask=dist<=20;base=torch.softmax(d['a0'].float(),0);base_correct=bool(mask[int(base.argmax())]);entry=dict(sample_id=meta['sample_id'],family_id=meta['family_id'],map_name=meta['map_name'],split=split,static_wrong=not base_correct,gate=p['real']['gate'],known_mass=p['real']['known_mass'])
                for a,b in [('real','black'),('real','shuffled')]:
                    pa,pb=p[a],p[b];ha,hb=pa['heatmap'].numpy(),pb['heatmap'].numpy();ba,bb=pa['object_belief'].numpy(),pb['object_belief'].numpy();entry[a+'_minus_'+b]=dict(goal_top1_changed=int(ha.argmax())!=int(hb.argmax()),object_top1_changed=int(ba.argmax())!=int(bb.argmax()),max_abs_heat_delta=float(abs(ha-hb).max()),goal_l1_delta=float(abs(ha-hb).sum()),target_goal_mass_delta=float(ha[mask].sum()-hb[mask].sum()),target_object_probability_delta=float(ba[tid]-bb[tid]),target_rank_real=int(np.flatnonzero(np.argsort(-ba,kind='stable')==tid)[0])+1,target_rank_control=int(np.flatnonzero(np.argsort(-bb,kind='stable')==tid)[0])+1)
                rs.append(entry)
            summary[split]={}
            for pair in ('real_minus_black','real_minus_shuffled'):
                v=[r[pair] for r in rs];summary[split][pair]=dict(families=len(rs),object_top1_changes=sum(r['object_top1_changed'] for r in v),goal_top1_changes=sum(r['goal_top1_changed'] for r in v),target_rank_improved=sum(r['target_rank_real']<r['target_rank_control'] for r in v),target_rank_worsened=sum(r['target_rank_real']>r['target_rank_control'] for r in v),target_rank_unchanged=sum(r['target_rank_real']==r['target_rank_control'] for r in v),max_abs_heat_delta_quantiles=np.quantile([r['max_abs_heat_delta'] for r in v],[0,.5,.9,1]).tolist(),positive_target_mass_changes=sum(r['target_goal_mass_delta']>1e-8 for r in v),negative_target_mass_changes=sum(r['target_goal_mass_delta']<-1e-8 for r in v))
            rows+=rs
    save(OUT/'visual_goal_impact.json',dict(summary=summary,rows=rows,interface_unchanged=True,gt_only_for_offline_target_mass_rank_diagnostics=True))
if __name__=='__main__':main()
