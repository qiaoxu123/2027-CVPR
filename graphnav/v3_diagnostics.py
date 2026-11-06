"""Private-label POSTHOC analysis. Never import from model/policy inference."""
import torch

@torch.no_grad()
def visual_correction_capacity(output,target_xy,world_bounds,visual_gain,radius_m=20.):
    """Optimistic upper bound under existing support and bounded logit correction.

    Allows independent +/-amplitude per observed cell, which the real network
    generally cannot realize. This diagnoses limited expressivity, NOT accuracy.
    GT is used only to define the scoring region AFTER the frozen forward.
    """
    geom=output.geometry_global.float();fused=output.fused_logits.float();support=output.observed_mask.bool()
    b,h,w=geom.shape
    if target_xy.shape!=(b,2) or world_bounds.shape!=(b,4):raise ValueError('Posthoc label/bounds shape mismatch')
    yy,xx=torch.meshgrid((torch.arange(h,device=geom.device)+.5)/h,(torch.arange(w,device=geom.device)+.5)/w,indexing='ij')
    xy=torch.stack((xx,yy),-1)
    distance=((xy[None]-target_xy[:,None,None])*(world_bounds[:,None,None,2:]-world_bounds[:,None,None,:2])).norm(dim=-1)
    correct=distance<=radius_m
    amplitude=float(torch.as_tensor(visual_gain).detach().abs())
    allowed=amplitude*support.float()
    rows=[]
    for i in range(b):
        region=correct[i];outside=~region
        if not region.any() or not outside.any():
            rows.append(dict(GT_POSTHOC_ONLY=True,region_defined=False));continue
        good=geom[i].masked_fill(~region,float('-inf')).max()
        bad=geom[i].masked_fill(~outside,float('-inf')).max()
        # Correct cells can be raised and wrong cells lowered only if observed.
        optimistic_good=(geom[i]+allowed[i]).masked_fill(~region,float('-inf')).max()
        optimistic_bad=(geom[i]-allowed[i]).masked_fill(~outside,float('-inf')).max()
        idx=int(fused[i].flatten().topk(min(20,h*w)).indices[0]);old=int(geom[i].flatten().topk(min(20,h*w)).indices[0])
        visual=getattr(output,"visual_global",fused-geom)[i].masked_fill(~support[i],float('-inf'))
        visual_hit=bool(region.flatten()[int(visual.flatten().topk(min(20,h*w)).indices[0])]) if support[i].any() else None
        margin=float(optimistic_good-optimistic_bad)
        rows.append(dict(GT_POSTHOC_ONLY=True,region_defined=True,
            visual_only_top1_correct_region_proxy=visual_hit,
            geometry_correct=bool(region.flatten()[old]),fused_correct=bool(region.flatten()[idx]),
            correct_region_observed=bool((region&support[i]).any()),
            wrong_minus_correct_geometry_peak=float(bad-good),
            amplitude_bound=amplitude,
            theoretical_optimistic_margin=margin,
            bound_insufficient=margin<0, capacity_tie_undetermined=margin==0,
            can_flip_under_independent_bounded_updates=margin>0,
            actual_delta_max=float((fused[i]-geom[i]).abs().max())))
    return rows
