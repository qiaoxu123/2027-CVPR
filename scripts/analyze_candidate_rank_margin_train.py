"""Select a candidate-switch margin on TRAIN development, confirm once on Birmingham.
GT is loaded only after candidate-score inference and used for offline metrics.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.trajectory_policy import CandidateTrajectoryStop
from scripts.train_v2_candidate_controller import ControllerStates, collate_states, candidate_world


def evaluate(model, features, labels, split, batch_size, device):
    indices=[r["index"] for r in labels if r["split"]==split]
    states=[]
    for i in indices:
        states.append({"query_index":i,"pose_world":[0.,0.,0.,0.],
            "target_world":labels[i]["target_world"],"teacher_world":[[0.,0.,0.]],
            "valid_teacher_z":[[0.]],"ground_level":0.,"is_start":True})
    ds=ControllerStates(features,states)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,
                      collate_fn=collate_states,pin_memory=True)
    records=[];model.eval()
    with torch.no_grad():
        for x,y,ids,_ in loader:
            x={k:v.to(device,non_blocking=True) for k,v in x.items()}
            out=model(**x)
            logits=out.candidate_logits.float()
            xy=candidate_world(x)
            target=y["target_world"].to(device)[:,:2]
            dist=torch.linalg.vector_norm(xy-target[:,None,:],dim=-1)
            margins=logits[:,1:].max(-1).values-logits[:,0]
            alt_slots=logits[:,1:].argmax(-1)+1
            for j,i in enumerate(ids.tolist()):
                records.append({"index":i,"margin":float(margins[j]),
                    "alt_slot":int(alt_slots[j]),"base_NE_m":float(dist[j,0]),
                    "alt_NE_m":float(dist[j,alt_slots[j]]),
                    "base_R20":int(dist[j,0]<=20),"alt_R20":int(dist[j,alt_slots[j]]<=20)})
    return records


def summarize(records, threshold):
    selected=[r["alt_NE_m"] if r["margin"]>threshold else r["base_NE_m"] for r in records]
    hits=[x<=20 for x in selected]
    base=[r["base_R20"] for r in records]
    corrected=sum(h and not b for h,b in zip(hits,base))
    damaged=sum(b and not h for h,b in zip(hits,base))
    return {"n":len(records),"R20":float(np.mean(hits)),"NE_m":float(np.mean(selected)),
        "switches":sum(r["margin"]>threshold for r in records),"corrected":corrected,
        "damaged":damaged,"net":corrected-damaged}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared",type=Path,required=True)
    p.add_argument("--feature-cache",type=Path,required=True)
    p.add_argument("--ranker",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=128)
    a=p.parse_args();device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    features=torch.load(a.feature_cache/"features.pt",map_location="cpu",weights_only=True)
    fm=json.load(open(a.feature_cache/"manifest.json"))
    state=torch.load(a.ranker,map_location="cpu",weights_only=True)
    if state.get("geometry_sha256")!=fm.get("trained_v2_sha256"):
        raise ValueError("Ranker and candidate-feature Geometry checkpoint mismatch")
    labels=json.load(open(a.prepared/"labels_private.json"))
    splits={s:[r for r in labels if r["split"]==s] for s in ("development","confirmation")}
    assert len(splits["development"])==3126 and len(splits["confirmation"])==7619
    model=CandidateTrajectoryStop(hidden=state["hidden"]).to(device).eval()
    model.load_state_dict(state["model"],strict=True)
    dev=evaluate(model,features,labels,"development",a.batch_size,device)
    base_dev=summarize(dev,float("inf"));full_dev=summarize(dev,float("-inf"))
    margins=sorted(set(r["margin"] for r in dev))
    candidates=[float("inf")]+margins
    # Choose by R@20, then NE, then fewer switches. Only TRAIN development is used.
    ranked=[]
    for t in candidates:
        m=summarize(dev,t);ranked.append((m["R20"],-m["NE_m"],-m["switches"],t,m))
    _,_,_,threshold,dev_selected=max(ranked,key=lambda q:(q[0],q[1],q[2]))
    confirmation=evaluate(model,features,labels,"confirmation",a.batch_size,device)
    base_confirm=summarize(confirmation,float("inf"));full_confirm=summarize(confirmation,float("-inf"))
    selected_confirm=summarize(confirmation,threshold)
    a.output.mkdir(parents=True,exist_ok=True)
    atomic_json({"development_rows":dev,"confirmation_rows":confirmation},a.output/"rows.json")
    atomic_json({"seed":42,"trainable_model":"frozen preserving-ranker; no re-training",
        "ranker_sha256":file_sha256(a.ranker),"feature_manifest_sha256":file_sha256(a.feature_cache/"manifest.json"),
        "prepared_manifest_sha256":file_sha256(a.prepared/"manifest.json"),"threshold_rule":"strictly switch from slot0 to best alternate if best_alternate_logit - slot0_logit > threshold; threshold selected on Cambridge TRAIN development maximizing R20, then minimizing NE and switch count",
        "threshold":threshold if np.isfinite(threshold) else None,
        "development":{"slot0":base_dev,"always_rank":full_dev,"selected_gate":dev_selected},
        "birmingham_confirmation":{"slot0":base_confirm,"always_rank":full_confirm,"selected_gate":selected_confirm},
        "caveat":"Geometry features are public at inference; labels used only for offline scoring. Base V2 has seen all official TRAIN during prior pretraining; this is adaptation diagnostic, not unseen-pretraining generalization. Confirmation evaluated once after threshold freeze."},a.output/"summary.json")
    print(json.dumps({"threshold":threshold if np.isfinite(threshold) else None,
        "dev_base":base_dev,"dev_rank":full_dev,"dev_gated":dev_selected,
        "confirmation_base":base_confirm,"confirmation_rank":full_confirm,
        "confirmation_gated":selected_confirm},indent=2),flush=True)

if __name__=="__main__":main()
