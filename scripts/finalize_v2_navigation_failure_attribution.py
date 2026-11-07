"""Add raw candidate-budget, subgroup, binding, and corrected gallery outputs.

All target-dependent calculations happen after loading the frozen public field.
No label changes query generation or the saved rollout/action sequence.
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def load_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        Path(path).write_text(""); return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list,dict)) else v for k,v in r.items()})


def parsed(v):
    try: return json.loads(v)
    except Exception: return v


def group_summary(rows, keys, raw_key_prefix="raw_candidate_recall_at_", nms_key_prefix="initial_nms_recall_at_"):
    groups = defaultdict(list)
    for row in rows: groups[row[keys]].append(row)
    out = {}
    for group, values in sorted(groups.items()):
        out[group] = {
            "n": len(values),
            "SR": float(np.mean([x["sr"] for x in values])),
            "OSR": float(np.mean([x["osr"] for x in values])),
            "nms_recall": {str(k):float(np.mean([x[nms_key_prefix+str(k)] for x in values])) for k in (1,5,10,20)},
            "raw_candidate_recall": {str(k):float(np.mean([x[raw_key_prefix+str(k)] for x in values])) for k in (1,5,10,20)},
            "never_entered_20m": sum(not x["osr"] for x in values),
            "never_entered_nms_top5": sum(not x["osr"] and x[nms_key_prefix+"5"] for x in values),
            "never_entered_nms_top20_miss": sum(not x["osr"] and not x[nms_key_prefix+"20"] for x in values),
            "failed_nms_top5_not_selected": sum(not x["sr"] and x[nms_key_prefix+"5"] and not x["selected_target_candidate_ever_within20"] for x in values),
        }
    return out


def main():
    p=argparse.ArgumentParser()
    for name in ("output","run","data","sbf-source","config","source-v2","trained-v2","stage1","legacy-queries","expanded-queries"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args(); out=a.output
    sys.path.insert(0,str(a.sbf_source.resolve()))
    from sbfnav.config import load_config, apply_overrides
    from sbfnav.dataset import CityNavDataset, CityReferCatalog
    from sbfnav.mapdata import GROUND_LEVEL
    from graphnav.renderer import CityNavObservedViewRenderer
    from graphnav.checkpointing import file_sha256
    from scripts.evaluate_v2_expanded_public_navigation import FrozenV2GeometryPolicy
    from scripts.analyze_v2_navigation_failure_attribution import spatial_nms, target_object, draw_case, make_plots, write_dashboard

    episodes=json.loads((out/"episode_attribution.json").read_text())["episodes"]
    steps=json.loads((out/"per_step_replay.json").read_text())["steps"]
    observations=json.loads((out/"historical_rgbd_projection_proxy.json").read_text())["episodes"]
    legacy_q, expanded_q=read_jsonl(a.legacy_queries),read_jsonl(a.expanded_queries)
    run_metrics=json.loads((a.run/"metrics.json").read_text())
    legacy_rows=[json.loads((a.run/"legacy"/"rows"/f"{i:05d}.json").read_text()) for i in range(2470)]
    new_rows=[json.loads((a.run/"expanded"/"rows"/f"{i:05d}.json").read_text()) for i in range(2470)]
    ds=CityNavDataset(a.data,"val_seen",version="processed_citynav")
    catalog=CityReferCatalog(a.data/"cityrefer")

    # Exact original policy proposal budget vs spatially de-duplicated full-grid list.
    for i,row in enumerate(episodes):
        field=np.load(out/"expanded_field_cache"/f"{i:05d}.npz")
        ids=field["candidate_ids"].astype(int); xy=field["world_xy"].astype(float)
        logits=field["logits"].astype(float).reshape(-1)
        target=ds[i].target_position[:2]
        d=np.linalg.norm(xy[ids]-target[None],axis=1)
        rank=next((j+1 for j,x in enumerate(d) if x<=20),None)
        for k in (1,5,10,20): row[f"raw_candidate_recall_at_{k}"]=int(bool((d[:k]<=20).any()))
        row["raw_candidate_first_hit_rank20"]=rank
        row["raw_candidate_head_budget_has_target20"]=int(rank is not None)
        nms_ids=spatial_nms(np.argsort(-logits,kind="stable"),xy,radius=20.0,k=20)
        nms_dist=np.linalg.norm(xy[np.asarray(nms_ids)]-target[None],axis=1)
        row["spatial_nms_top20_posthoc"]=[{"grid_id":int(g),"x":float(xy[g,0]),"y":float(xy[g,1]),
            "logit":float(logits[g]),"distance_to_gt_center_m":float(dd)} for g,dd in zip(nms_ids,nms_dist)]
        row["raw_head_top20_posthoc"]=[{"grid_id":int(g),"x":float(xy[g,0]),"y":float(xy[g,1]),
            "logit":float(logits[g]),"distance_to_gt_center_m":float(dd)} for g,dd in zip(ids[:20],d[:20])]
        # The frozen guarded-blend policy with explore_candidates=0 always executes candidate slot 0.
        row["selected_candidate_slot_zero_only"] = int(all(int(s["selected_candidate_slot"])==0 for s in steps if int(s["index"])==i))
        oldq,newq=legacy_q[i],expanded_q[i]
        names=sorted(set(x["name"] for x in newq.get("landmarks",[])))
        objmap=catalog.objects[row["map_name"]]
        homonyms=[]
        for name in names:
            ids_same=[str(o.get("id")) for o in objmap.values() if str(o.get("name","")).casefold()==str(name).casefold()]
            if len(set(ids_same))>1: homonyms.append({"name":name,"object_ids":sorted(set(ids_same))})
        row["catalog_homonymous_anchor_names"] = homonyms
        row["catalog_homonymous_anchor_count"] = len(homonyms)
        row["query_anchor_name_polygon_fragment_counts"] = dict(Counter(x["name"] for x in newq.get("landmarks",[])))
        row["input_landmark_query_changed_vs_legacy"] = int(oldq.get("landmarks",[])!=newq.get("landmarks",[]))
        row["bbox_projection_geometry_support"] = observations[i]["first_geometric_projection_step"] is not None
        row["bbox_projection_long16_any"] = observations[i]["ever_long_side_16px"]
        row["bbox_projection_long32_short16_any"] = observations[i]["ever_long32_short16_proxy"]
        row["first_projection_long32_short16_remaining_steps"] = observations[i]["remaining_actions_after_proxy"]

    write_csv(out/"per_episode_attribution.csv",episodes)
    # Repeat the static NMS rank at each causal decision and also expose the head's raw budget.
    by_index={r["index"]:r for r in episodes}
    for step in steps:
        row=by_index[step["index"]]
        for k in (1,5,10,20):
            step[f"static_belief_nms_recall_at_{k}"]=int(row[f"initial_nms_gt_rank20m"] is not None and row[f"initial_nms_gt_rank20m"]<=k)
            step[f"raw_head_candidate_recall_at_{k}"]=row[f"raw_candidate_recall_at_{k}"]
    write_csv(out/"per_step_replay.csv",steps)

    never=[r for r in episodes if not r["osr"]]
    failed=[r for r in episodes if not r["sr"]]
    def hits(rows,prefix):
        return {str(k):sum(r[prefix+str(k)] for r in rows) for k in (1,5,10,20)}
    top5_fail=[r for r in failed if r["initial_nms_recall_at_5"]]
    top5_never=[r for r in never if r["initial_nms_recall_at_5"]]
    raw_top5_never=[r for r in never if r["raw_candidate_recall_at_5"]]
    summary=json.loads((out/"diagnostic_summary.json").read_text())
    summary["raw_candidate_proposal_budget"]={
        "definition":"The 20 raw cells supplied to the frozen trajectory/rank head, scored by distance to target only after frozen inference.",
        "recall_all_counts":hits(episodes,"raw_candidate_recall_at_"),
        "recall_all":{str(k):hits(episodes,"raw_candidate_recall_at_")[str(k)]/2470 for k in (1,5,10,20)},
        "recall_failed_counts":hits(failed,"raw_candidate_recall_at_"),
        "recall_never_entered_counts":hits(never,"raw_candidate_recall_at_"),
        "never_entered_nms_top5_and_raw_budget_contains_target":sum(r["initial_nms_recall_at_5"] and r["raw_candidate_head_budget_has_target20"] for r in never),
        "never_entered_nms_top5_but_target_not_in_raw_head_budget":sum(r["initial_nms_recall_at_5"] and not r["raw_candidate_head_budget_has_target20"] for r in never),
        "failed_nms_top5_and_raw_budget_contains_target":sum(r["initial_nms_recall_at_5"] and r["raw_candidate_head_budget_has_target20"] for r in failed),
        "failed_nms_top5_but_target_not_in_raw_head_budget":sum(r["initial_nms_recall_at_5"] and not r["raw_candidate_head_budget_has_target20"] for r in failed),
        "policy_selected_slot_zero_steps":sum(int(x["selected_candidate_slot"])==0 for x in steps),
        "policy_total_decisions":len(steps),
        "policy_actual_candidate_reranking_enabled":False,
    }
    summary["candidate_recall_grouped_by_target_category_proxy"]=group_summary(episodes,"target_category_proxy")
    summary["candidate_recall_grouped_by_relation_rule_proxy"]=group_summary(episodes,"relation_complexity_proxy")
    summary["top5_failure_upper_bound"]={
        "failed_with_nms_top5_location_proxy":len(top5_fail),
        "failed_top5_within_actual_raw_head_budget":sum(r["raw_candidate_head_budget_has_target20"] for r in top5_fail),
        "never_entered_with_nms_top5_location_proxy":len(top5_never),
        "never_entered_top5_within_actual_raw_head_budget":sum(r["raw_candidate_head_budget_has_target20"] for r in top5_never),
        "warning":"20m target-center proximity is a location proxy, not verified physical-instance matching or a promise that reranking will recover SR.",
    }
    summary["historical_observation_by_failure_subset"]={}
    for name, subset in (
        ("never_entered",never),
        ("never_entered_nms_top5",top5_never),
        ("all_failed_nms_top5",top5_fail),
        ("never_entered_nms_rank6_20",[r for r in never if r["initial_nms_gt_rank20m"] and 6<=r["initial_nms_gt_rank20m"]<=20]),
        ("never_entered_nms_top20_miss",[r for r in never if not r["initial_nms_recall_at_20"]]),
    ):
        subset_obs=[observations[r["index"]] for r in subset]
        remaining=[x["remaining_actions_after_proxy"] for x in subset_obs if x["remaining_actions_after_proxy"] is not None]
        co=[x["similar_scale_same_type_objects_best_proxy_frame"] for x in subset_obs if x["similar_scale_same_type_objects_best_proxy_frame"] is not None]
        summary["historical_observation_by_failure_subset"][name]={
            "n":len(subset),
            "target_polygon_projects_into_any_logged_frame":sum(x["first_geometric_projection_step"] is not None for x in subset_obs),
            "max_dimension_at_least_16px_proxy":sum(x["ever_long_side_16px"] for x in subset_obs),
            "long_side32_short_side16px_proxy":sum(x["ever_long32_short16_proxy"] for x in subset_obs),
            "depth_valid_polygon_pixels_any_logged_frame":sum(x["any_depth_valid_target_pixels"] for x in subset_obs),
            "remaining_actions_at_first_long32_short16_proxy_mean":float(np.mean(remaining)) if remaining else None,
            "remaining_actions_at_first_long32_short16_proxy_median":float(np.median(remaining)) if remaining else None,
            "co_visible_similar_size_same_object_type_polygon_proxy_mean":float(np.mean(co)) if co else None,
            "warning":"All are geometry/pixel projection proxies from only logged poses. No occlusion model, segmentation label, or visual recognizability ground truth.",
        }
    # Binding changes are public-input deltas. No independent reference graph labels exist here.
    corrected=[];damaged=[];variant_types=Counter()
    old_set_results=[]
    for i,(old_row,new_row) in enumerate(zip(legacy_rows,new_rows)):
        old_names=set(old_row.get("public_reference_names",[]));new_names=set(new_row.get("public_reference_names",[]))
        changed=legacy_q[i].get("landmarks",[])!=expanded_q[i].get("landmarks",[])
        outcome="corrected" if old_row["metrics"]["SR"]==0 and new_row["metrics"]["SR"]==1 else ("damaged" if old_row["metrics"]["SR"]==1 and new_row["metrics"]["SR"]==0 else "unchanged")
        if outcome in ("corrected","damaged"):
            kind="anchor_added_from_empty" if not old_names and new_names else ("additional_anchor" if old_names and old_names<new_names else ("anchor_replacement_or_scope_change" if old_names!=new_names else "non-name_input_change"))
            variant_types[(outcome,kind)]+=1
            item={"index":i,"outcome":outcome,"instruction":new_rows[i]["instruction"],
                  "legacy_names":sorted(old_names),"expanded_names":sorted(new_names),
                  "added_names":sorted(new_names-old_names),"removed_names":sorted(old_names-new_names),
                  "landmark_query_changed":changed,"primary_failure_class":episodes[i]["primary_failure_class"],
                  "nms_rank":episodes[i]["initial_nms_gt_rank20m"],"target_category":episodes[i]["target_category_proxy"],
                  "case_image":f"cases/case_{i:04d}.jpg"}
            (corrected if outcome=="corrected" else damaged).append(item)
        old_set_results.append({"index":i,"landmark_query_changed":int(changed),"outcome":outcome})
    summary["binding_pair_audit"]={
        "query_geometry_or_semantics_changed":sum(x["landmark_query_changed"] for x in old_set_results),
        "corrected":len(corrected),"damaged":len(damaged),"net":len(corrected)-len(damaged),
        "all_corrected_cases_public_query_changed":all(x["landmark_query_changed"] for x in corrected),
        "all_damaged_cases_public_query_changed":all(x["landmark_query_changed"] for x in damaged),
        "corrected_damaged_change_types":{f"{k[0]}::{k[1]}":v for k,v in variant_types.items()},
        "semantic_error_labels":"No independently human-labelled relation/scope gold is available; outputs only show what public binding changed and its post-hoc navigation outcome.",
    }
    summary["primary_failure_counts_failed_only"] = dict(Counter(r["primary_failure_class"] for r in failed))
    summary["stop_and_goal_audit"]={
        "all_termination_counts":summary["closedloop_pair"]["termination_counts"],
        "trajectory_goal_switch_over20_count":summary["all_episode_goal_hop_over20"],
        "terminal_stop_guard_anomalies":summary["final_failures_stop_guard_anomalies"],
        "failures_stopped_at_selected_goal_outside_target_radius":sum("stopped_at_goal_over_20m_from_gt" in r.get("tags",[]) for r in episodes),
        "interpretation":"A policy stop outside the target radius is not alone evidence of a Stop Head error. All logged terminations are policy_stop; controller replay verifies stop probability/arrival/altitude guard at the terminal decision.",
    }
    summary["precision_metadata_audit"]={
        "prior_manifest_declared":"FP32; TF32 disabled",
        "actual_runtime":"CUDA BF16 autocast is enabled in frozen Geometry and trajectory-head forward; logits/metric scoring are converted to FP32; TF32 disabled.",
        "impact":"Both paired arms used the same actual precision, so their paired comparison is internally controlled; the previous precision label was inaccurate and no full split was rerun in FP32.",
    }
    summary["causal_replay_verification"]={
        "coverage":2470,"decision_rows":len(steps),"saved_candidate_mode_goal_stop_probability_reproduced":True,
        "belief_update_per_step":"None: fixed Geometry field per instruction; actual RGB-D frames were not consumed by V2 policy.",
        "candidate_execution":"guarded-blend with explore_candidates=0 forces candidate slot 0 at every decision; trajectory/rank head does not choose among its Top-20 proposals in this experiment.",
    }
    summary["decision_upper_bounds"]={
        "no_enter_miss_nms_top20":sum(not r["initial_nms_recall_at_20"] for r in never),
        "no_enter_correct_location_in_nms_top20":sum(r["initial_nms_recall_at_20"] for r in never),
        "no_enter_correct_location_in_raw_head_top20":sum(r["raw_candidate_head_budget_has_target20"] for r in never),
        "no_enter_correct_location_in_nms_top5":len(top5_never),
        "warning":"Ceilings under the 20m center-proximity proxy, not achievable SR estimates; instance identity and ranking correctness are unverified.",
    }
    for row in episodes:
        row["tags"] = [tag for tag in row.get("tags",[]) if tag != "multiple_polygons_same_name"]
        if any(int(n)>1 for n in row.get("query_anchor_name_polygon_fragment_counts",{}).values()):
            row["tags"].append("repeated_anchor_polygon_fragments")
        if row.get("catalog_homonymous_anchor_names"):
            row["tags"].append("catalog_same_name_multiple_object_ids")
    summary["ambiguity_tag_definitions"]={
        "repeated_anchor_polygon_fragments":"One query contains multiple polygon fragments for one displayed anchor name; these may be fragments of one multipart entity and are not counted as homonymous entities.",
        "catalog_same_name_multiple_object_ids":"The map catalog contains distinct object IDs with the same exact case-folded name. This flags possible ambiguity; it does not prove that the parser bound the wrong object.",
        "scope_or_binding_error":"Not independently adjudicated because no human gold relation/binding graph exists for all 2,470 instructions.",
    }
    summary["multilabel_counts"] = {
        tag: sum(tag in row.get("tags",[]) for row in episodes)
        for tag in sorted({tag for row in episodes for tag in row.get("tags",[])})
    }
    summary["landmark_binding_ambiguity_audit"] = {
        "tasks_without_public_anchor": sum(int(row.get("public_anchor_count",0)) == 0 for row in episodes),
        "tasks_with_repeated_polygon_fragments_for_name": sum("repeated_anchor_polygon_fragments" in row.get("tags",[]) for row in episodes),
        "tasks_with_same_exact_name_on_distinct_catalog_object_ids": sum("catalog_same_name_multiple_object_ids" in row.get("tags",[]) for row in episodes),
        "tasks_with_multiple_public_anchor_names": sum(int(row.get("public_anchor_count",0)) > 1 for row in episodes),
        "tasks_without_anchor_and_failed": sum(int(row.get("public_anchor_count",0)) == 0 and not row["sr"] for row in episodes),
        "warning":"No independent gold binding/scope annotations exist; these counts describe public query/catalog structure only, not parser error counts.",
    }
    def distribution(values):
        values=np.asarray([float(v) for v in values if v is not None],dtype=float)
        if not len(values): return {"n":0,"median":None,"p75":None,"p90":None,"mean":None}
        return {"n":int(len(values)),"median":float(np.median(values)),"p75":float(np.percentile(values,75)),"p90":float(np.percentile(values,90)),"mean":float(np.mean(values))}
    summary["top1_to_best_target20_score_gap_logit"]={
        "all_target20_candidates":distribution([r.get("score_gap_top1_to_best_gt20_logit") for r in episodes]),
        "never_entered_with_nms_top5":distribution([r.get("score_gap_top1_to_best_gt20_logit") for r in top5_never]),
        "never_entered_with_raw_head_budget_target":distribution([r.get("score_gap_top1_to_best_gt20_logit") for r in never if r["raw_candidate_head_budget_has_target20"]]),
    }
    summary["relation_proxy"] = {
        name:{"n":len([r for r in episodes if r["relation_complexity_proxy"]==name]),
             "sr":float(np.mean([r["sr"] for r in episodes if r["relation_complexity_proxy"]==name])),
             "nms_top5":float(np.mean([r["initial_nms_recall_at_5"] for r in episodes if r["relation_complexity_proxy"]==name])),
             "never_entered":sum(not r["osr"] for r in episodes if r["relation_complexity_proxy"]==name),
             "never_entered_nms_top5":sum(not r["osr"] and r["initial_nms_recall_at_5"] for r in episodes if r["relation_complexity_proxy"]==name)}
        for name in sorted({r["relation_complexity_proxy"] for r in episodes})}
    summary["target_category_summary"]={
        name:{"n":len([r for r in episodes if r["target_category_proxy"]==name]),
             "sr":float(np.mean([r["sr"] for r in episodes if r["target_category_proxy"]==name])),
             "nms_r5":float(np.mean([r["initial_nms_recall_at_5"] for r in episodes if r["target_category_proxy"]==name])),
             "nms_r20":float(np.mean([r["initial_nms_recall_at_20"] for r in episodes if r["target_category_proxy"]==name])),
             "never_entered":sum(not r["osr"] for r in episodes if r["target_category_proxy"]==name),
             "never_entered_top5":sum(not r["osr"] and r["initial_nms_recall_at_5"] for r in episodes if r["target_category_proxy"]==name)}
        for name in sorted({r["target_category_proxy"] for r in episodes})}
    (out/"diagnostic_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")

    # Refresh parser outcome CSV values and include target/object/anchor polygons in the pictures.
    case_path=out/"public_binding_corrected_damaged_74.csv"
    case_rows=load_csv(case_path)
    case_by_idx={int(x["index"]):x for x in case_rows}
    all_gallery_ids=sorted(case_by_idx)
    from sbfnav.planning_state import PlanningStateBuilder
    from graphnav.closedloop import PublicCatalog
    policy=FrozenV2GeometryPolicy(a.source_v2,a.trained_v2,a.stage1,out/"scratch_gallery_inputs","trained_v2")
    renderer=CityNavObservedViewRenderer(a.data/"rgbd",out/"unused_render_dir",image_size=384)
    for j,idx in enumerate(all_gallery_ids):
        oldq,newq=legacy_q[idx],expanded_q[idx]
        fields={}
        for label,q in (("old",oldq),("new",newq)):
            policy.record=q;policy.visited_goal_cells=set()
            if label=="new" and (out/"expanded_field_cache"/f"{idx:05d}.npz").is_file():
                cached=np.load(out/"expanded_field_cache"/f"{idx:05d}.npz")
                fields[label]={"logits":cached["logits"],"world_xy":cached["world_xy"]}
            else:
                field,inputs=policy.frozen_field();bounds=np.asarray(q["world_bounds"],float);grid=inputs["grid_xy"][0].float().cpu().numpy()
                fields[label]={"logits":field.belief_logits[0].float().cpu().numpy(),"world_xy":bounds[:2]+grid*(bounds[2:]-bounds[:2])}
        record=ds[idx];oldrow=legacy_rows[idx];newrow=new_rows[idx]
        proxy=observations[idx];frame_step=proxy.get("best_proxy_frame_step")
        frame=None
        if frame_step is not None:
            pose=np.asarray(newrow["poses"][frame_step],float)
            obs=renderer.current_rgbd(record.map_name,pose,GROUND_LEVEL[record.map_name])
            saved_metric=proxy["frames"][frame_step]
            frame={"step":frame_step,"bgr":obs["bgr"],"metric":saved_metric}
        obj=target_object(catalog,record.map_name,record.object_id)
        img=out/"cases"/f"case_{idx:04d}.jpg"
        draw_case(idx,oldq,newq,fields["old"],fields["new"],oldrow,newrow,obj,record.target_position[:2],frame,img)
        if idx in case_by_idx: case_by_idx[idx]["case_image"]=f"cases/{img.name}"
        if (j+1)%20==0: print(f"rendered corrected/damaged diagnostic cases {j+1}/{len(all_gallery_ids)}",flush=True)
    del policy
    for png in (out/"cases").glob("*.png"): png.unlink()
    for x in case_rows:
        idx=int(x["index"])
        if idx in case_by_idx: x.update(case_by_idx[idx])
    write_csv(case_path,case_rows)
    changed_rows=[{"index":i,"case_image":case_by_idx[i]["case_image"],"outcome":"corrected" if i in {int(x["index"]) for x in corrected} else "damaged"} for i in sorted(case_by_idx)]
    delta_rows=[]
    for idx in sorted(set(int(x["index"]) for x in corrected+damaged)):
        r=episodes[idx]
        delta_rows.append({k:r[k] for k in ("index","episode_id","map_name","instruction","target_category_proxy","legacy_anchor_names","expanded_anchor_names","legacy_to_expanded_added_names","legacy_to_expanded_removed_names","landmark_query_geometry_changed","initial_nms_gt_rank20m","initial_top1_ne_m","final_ne_m","sr","osr","primary_failure_class","tags")})
        delta_rows[-1]["legacy_sr"]=legacy_rows[idx]["metrics"]["SR"]
        delta_rows[-1]["expanded_sr"]=new_rows[idx]["metrics"]["SR"]
        delta_rows[-1]["case_image"]=case_by_idx[idx]["case_image"]
    write_dashboard(out,summary,episodes,delta_rows)
    write_csv(out/"per_episode_attribution.csv",episodes)
    make_plots(out,summary,episodes,observations)
    print(json.dumps({"raw_candidates":summary["raw_candidate_proposal_budget"],
                      "top5_ceiling":summary["top5_failure_upper_bound"],
                      "observation_subset":summary["historical_observation_by_failure_subset"],
                      "binding_pair":summary["binding_pair_audit"]},ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__": main()
