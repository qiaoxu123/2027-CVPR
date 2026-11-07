"""Evaluate candidate-score vs joint candidate/mode selection on TRAIN Dev only."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.trajectory_policy import CandidateTrajectoryStop, remaining_waypoints
from scripts.train_v2_candidate_controller import ControllerStates, collate_states, evaluate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared-data", "data-root", "sbf-source", "stage1", "feature-cache", "training-run", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=128)
    a = p.parse_args()
    sys.path.insert(0, str(a.sbf_source.resolve()))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = torch.load(a.feature_cache / "features.pt", map_location="cpu", weights_only=True)
    private = json.loads((a.prepared_data / "labels_private.json").read_text())
    ids = [r["index"] for r in private if r["split"] == "development"]
    if len(ids) != 3126 or any(private[i]["city"] != "cambridge" for i in ids):
        raise ValueError("Expected fixed Cambridge TRAIN development group")
    records = CityNavDataset(a.data_root, "train_seen", version="processed_citynav")
    states = []
    for i in ids:
        record = records[i]; path = np.asarray(record.trajectory, dtype=np.float32); label = private[i]
        for cut in sorted(set((0, len(path)//3, 2*len(path)//3, len(path)-1))):
            teacher = remaining_waypoints(path, cut); ground = GROUND_LEVEL[record.map_name]
            states.append({"query_index": i, "pose_world": path[cut].tolist(),
                "target_world": label["target_world"], "teacher_world": teacher.tolist(),
                "valid_teacher_z": ((teacher[:, 2] >= ground+10) & (teacher[:, 2] <= ground+100)).astype(np.float32).tolist(),
                "ground_level": ground, "is_start": cut == 0})
    dataset = ControllerStates(features, states)
    loader = DataLoader(dataset, batch_size=a.batch_size, shuffle=False, num_workers=0,
                        pin_memory=device.type == "cuda", collate_fn=collate_states)
    old = torch.load(a.stage1, map_location="cpu", weights_only=True)
    hidden = int(features["candidate_context"].shape[-1])
    old_state = {k[len("trajectory."):]: v for k, v in old["model"].items() if k.startswith("trajectory.")}
    models = {"old_stage1": old_state}
    for arm in ("warmstart", "scratch"):
        checkpoint = a.training_run / arm / "last.pt"
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        models[arm] = payload["model"]
    result = {"n_episodes": len(ids), "n_states": len(states),
        "train_data_manifest_sha256": file_sha256(a.prepared_data / "manifest.json"),
        "feature_manifest_sha256": file_sha256(a.feature_cache / "manifest.json"),
        "checkpoints": {}}
    for name, state in models.items():
        model = CandidateTrajectoryStop(hidden=hidden).to(device).eval()
        model.load_state_dict(state, strict=True)
        metrics, rows = evaluate(model, loader, device)
        result["checkpoints"][name] = {"metrics": metrics, "rows": rows}
        del model
    atomic_json(result, a.output)
    print(json.dumps({k: v["metrics"] for k, v in result["checkpoints"].items()}, indent=2), flush=True)


if __name__ == "__main__":
    main()
