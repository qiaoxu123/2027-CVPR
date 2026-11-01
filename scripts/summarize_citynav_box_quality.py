"""Human quality (NA if missing) and physical-instance grouped proxy recall."""
import argparse,json
from collections import Counter,defaultdict
from pathlib import Path
import torch
from torchvision.ops import box_iou
from graphnav.checkpointing import atomic_json
from graphnav.citynav_box_diagnostics import authorize_box


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['annotations','reviews','records','output']:p.add_argument('--'+n,required=True)
    a=p.parse_args()
    if Path(a.output).exists():raise ValueError('No overwrite')
    annotations={r['sample_id']:r for r in map(json.loads,Path(a.annotations).read_text().splitlines())}
    reviews=json.loads(Path(a.reviews).read_text())['reviews'];counts=Counter();quality=defaultdict(list);primary_visible=0
    for review in reviews:
        ann=annotations.get(review['sample_id'])
        if ann is None or ann['image_sha']!=review.get('image_sha'):raise ValueError('Review identity mismatch')
        human=(review.get('source')=='human' and review.get('human_attested') is True and review.get('reviewer','').strip())
        if not human:continue
        counts[review['status']]+=1
        approved=authorize_box('verified',ann,review,'train')
        if approved:
            primary_visible+=1
            iou=float(box_iou(torch.tensor(ann['auto_bbox_xyxy'],dtype=torch.float32)[None],torch.tensor(approved.bbox)[None])[0,0]) if ann['auto_bbox_xyxy'] else 0.
            quality[ann['category']].append(iou)
    records=list(map(json.loads,Path(a.records).read_text().splitlines()));eligible=[r for r in records if r['oracle_authorized']]
    groups=defaultdict(list)
    for r in eligible:groups[r['instance_group']].append(r)
    macro={}
    for stage in ['raw900','valid900','nms','top15','top30','top60']:
        macro[stage]=sum(sum(r['stages'][stage]['covered'] for r in items)/len(items) for items in groups.values())/len(groups) if groups else None
    cohort_counts={}
    for name,subset in [('all',records),('auto_hypotheses',eligible),('A0_wrong',[r for r in records if not r['methods']['A0']['correct']]),('hypothesis_A0_wrong',[r for r in eligible if not r['methods']['A0']['correct']])]:
        cohort_counts[name]=dict(states=len(subset),episodes=len({r['episode'] for r in subset}),instructions=len({r['instruction_group'] for r in subset}),instances=len({r['instance_group'] for r in subset}))
    observed_outside=[r for r in records if annotations[r['sample_id']]['geometric_visibility_hint']=='outside']
    ref_supported=[r for r in observed_outside if any(ref['projection']['valid_roi'] for ref in annotations[r['sample_id']]['references'])]
    indirect=dict(geometric_target_outside_states=len(observed_outside),outside_with_supported_public_reference=len(ref_supported),
                  outside_reference_and_explicit_world_relation=sum(bool(annotations[r['sample_id']]['relations']) for r in ref_supported),
                  A0_wrong_in_ref_supported=sum(not r['methods']['A0']['correct'] for r in ref_supported),
                  Native_corrected_in_ref_supported=sum(r['methods']['Native15']['corrected_A0'] for r in ref_supported),
                  Native_damaged_in_ref_supported=sum(r['methods']['Native15']['damaged_A0'] for r in ref_supported),
                  verified_indirect_visual_evidence=None,note='Geometry only; no human reference identity/visibility labels, no validated indirect inference claim')
    reviewed=sum(counts.values())
    atomic_json(dict(human_reviewed=reviewed,human_approved_visible=primary_visible,human_status_counts=dict(counts),
        human_auto_confirmed_rate=counts['confirmed']/reviewed if reviewed else None,
        human_quality_IoU_by_category={k:dict(samples=len(v),mean_IoU=sum(v)/len(v)) for k,v in quality.items()} if reviewed else None,
        human_main_Oracle_ABC='NA — no verified visible samples' if not primary_visible else 'Run --oracle-mode verified with this review file',
        grouped_counts=cohort_counts,proxy_recall_physical_instance_macro=macro,indirect_reference_geometry=indirect,
        caveat='All proxy recalls still conditional on UNREVIEWED automatic boxes; grouping does not validate labels'),Path(a.output))


if __name__=='__main__':main()
