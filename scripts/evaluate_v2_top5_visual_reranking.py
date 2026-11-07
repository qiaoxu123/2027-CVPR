"""Frozen, post-hoc observed-RGB Top-5 candidate reranking diagnostic.

The Geometry candidate list is read from the already completed Val-Seen run.
RGB is rendered only at poses recorded in that run; each ROI is warped from
the corresponding 384px frame and invalid pixels are masked. GT is parsed only
after image/text features and all candidate selections have been computed.
The output is a reranking feasibility probe, never a closed-loop SR result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path

import cv2
import numpy as np
import torch

from graphnav.checkpointing import atomic_json, file_sha256
from graphnav.evidence_grounding import appearance_query
from graphnav.renderer import CityNavObservedViewRenderer
from graphnav.v2_visual_candidate_reranking import (
    conservative_visual_rerank,
    first_causal_comparable_frame,
    observed_candidate_roi,
)
from graphnav.memory import ObservedRGBMemory


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


class FrozenSigLIP2:
    """Load the exact cached Stage1 SigLIP2 revision without network access."""
    def __init__(self, name, revision, device="cuda", microbatch=16, cache_limit=4096):
        from transformers import AutoModel, AutoProcessor
        hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
        model_dir = "models--" + name.replace("/", "--")
        snapshot = hub / model_dir / "snapshots" / revision
        if not (snapshot / "config.json").is_file() or not (snapshot / "model.safetensors").is_file():
            raise FileNotFoundError(f"exact frozen SigLIP2 snapshot missing: {snapshot}")
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(str(snapshot), use_fast=False,
                                                       local_files_only=True)
        self.model = AutoModel.from_pretrained(str(snapshot), local_files_only=True).to(self.device).eval()
        self.model.requires_grad_(False)
        self.revision = getattr(self.model.config, "_commit_hash", None)
        if self.revision != revision:
            raise ValueError(f"loaded model revision {self.revision} != required {revision}")
        self.dimension = int(self.model.config.text_config.hidden_size)
        self.microbatch, self.cache_limit = int(microbatch), int(cache_limit)
        self.image_cache, self.text_cache = OrderedDict(), {}
        self.encoded_images, self.cache_hits = 0, 0

    @staticmethod
    def _key(image):
        digest = hashlib.sha256(str(image.shape).encode())
        digest.update(np.ascontiguousarray(image).tobytes())
        return digest.hexdigest()

    @torch.inference_mode()
    def images(self, images):
        import torch.nn.functional as F
        import cv2
        from PIL import Image
        keys = [self._key(image) for image in images]
        unique = OrderedDict((key, image) for key, image in zip(keys, images) if key not in self.image_cache)
        self.cache_hits += len(keys) - len(unique)
        pending = list(unique.items())
        for start in range(0, len(pending), self.microbatch):
            chunk = pending[start:start + self.microbatch]
            rgb = [Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) for _, image in chunk]
            batch = self.processor(images=rgb, return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vector = self.model.get_image_features(**batch)
            if not torch.is_tensor(vector):
                vector = vector.pooler_output
            vector = F.normalize(vector.float(), dim=-1)
            for (key, _), feature in zip(chunk, vector):
                self.image_cache[key] = feature.detach().cpu().clone()
                self.encoded_images += 1
        vectors = torch.stack([self.image_cache[key] for key in keys]).to(self.device)
        for key in keys:
            self.image_cache.move_to_end(key)
        while len(self.image_cache) > self.cache_limit:
            self.image_cache.popitem(last=False)
        return vectors

    @torch.inference_mode()
    def texts(self, texts):
        import torch.nn.functional as F
        missing = list(dict.fromkeys(text for text in texts if text not in self.text_cache))
        if missing:
            batch = self.processor(text=missing, padding="max_length", truncation=True,
                                   max_length=self.model.config.text_config.max_position_embeddings,
                                   return_tensors="pt").to(self.device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                vector = self.model.get_text_features(**batch)
            if not torch.is_tensor(vector):
                vector = vector.pooler_output
            for text, feature in zip(missing, F.normalize(vector.float(), dim=-1)):
                self.text_cache[text] = feature.detach().clone()
        return torch.stack([self.text_cache[text] for text in texts])


def write_csv(path, rows, fields):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def json_sha256(path):
    return file_sha256(path)


def stratified_review_indices(failures, proxy_by_index, *, seed=42):
    """Select ~200 failures, spread across class, pixel proxy, and Geo rank."""
    quotas = {"Car": 80, "Building": 90, "Ground": 20, "Parking": 10}
    rng = np.random.default_rng(seed)
    chosen = []
    for category, quota in quotas.items():
        groups = defaultdict(list)
        for row in failures:
            if row["target_category_proxy"] != category:
                continue
            idx = int(row["index"])
            proxy = proxy_by_index.get(idx, {})
            clear_proxy = bool(proxy.get("ever_long32_short16_proxy", "False") in (True, "True", "1"))
            rank = int(row.get("initial_nms_gt_rank20m") or 99)
            groups[(clear_proxy, min(rank, 5))].append(idx)
        for values in groups.values():
            rng.shuffle(values)
        keys = sorted(groups, key=lambda k: (not k[0], k[1]))
        taken = []
        while len(taken) < quota and any(groups[k] for k in keys):
            for key in keys:
                if groups[key] and len(taken) < quota:
                    taken.append(groups[key].pop())
        chosen.extend(taken)
    if len(chosen) < min(200, len(failures)):
        remain = [int(r["index"]) for r in failures if int(r["index"]) not in set(chosen)]
        rng.shuffle(remain)
        chosen.extend(remain[:min(200, len(failures)) - len(chosen)])
    return sorted(chosen)


def make_candidate_frame(bgr, footprint, candidates):
    """Draw Geometry candidate positions only; no target annotation is shown."""
    from graphnav.landmark_projection import world_to_frame
    canvas = bgr.copy()
    colors = [(40, 40, 240), (30, 190, 255), (20, 210, 80),
              (230, 130, 20), (210, 30, 180)]
    for rank, candidate in enumerate(candidates, 1):
        px = world_to_frame(np.asarray(candidate["xy"], np.float32)[None],
                            footprint, bgr.shape)[0]
        if not np.isfinite(px).all() or not (0 <= px[0] < bgr.shape[1] and 0 <= px[1] < bgr.shape[0]):
            continue
        point = tuple(np.rint(px).astype(int))
        cv2.circle(canvas, point, 12, colors[rank - 1], 2)
        cv2.putText(canvas, str(rank), (point[0] - 4, point[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_review_assets(out, idx, instruction, public_candidates, result, visual_scores,
                       remaining_actions, display_only_fallback=False):
    """Write an anonymized human-review task; intentionally omits all GT."""
    assets = out / "review" / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    observation = result["observation"]
    stem = f"{idx:05d}"
    frame_rel = f"assets/{stem}_frame.jpg"
    frame = make_candidate_frame(observation["bgr"], observation["footprint_world"], public_candidates)
    cv2.imwrite(str(assets / f"{stem}_frame.jpg"), frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    items = []
    for rank, candidate in enumerate(public_candidates):
        scale_items = {}
        for extent in (20.0, 40.0):
            crop = result["crops"][rank][extent]
            image = crop["image"]
            rel = None
            if image is not None:
                suffix = f"_{rank+1}_{int(extent)}m.jpg"
                cv2.imwrite(str(assets / f"{stem}{suffix}"), image,
                            [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                rel = f"assets/{stem}{suffix}"
            scale_items[str(int(extent))] = {
                "path": rel,
                "support": round(float(crop["support"]), 4),
                "valid": bool(crop["valid"]),
            }
        visual = visual_scores[rank]
        items.append({
            "rank": rank + 1,
            "logit": float(candidate["logit"]),
            "x": round(float(candidate["xy"][0]), 2),
            "y": round(float(candidate["xy"][1]), 2),
            "dx_to_top1": round(float(candidate["xy"][0] - public_candidates[0]["xy"][0]), 2),
            "dy_to_top1": round(float(candidate["xy"][1] - public_candidates[0]["xy"][1]), 2),
            "scales": scale_items,
            "score_full": None if visual is None else [round(float(v), 5) for v in visual[0]],
            "score_appearance": None if visual is None else [round(float(v), 5) for v in visual[1]],
        })
    return {
        "index": idx,
        "instruction": instruction,
        "frame": frame_rel,
        "step": int(result["step"]),
        "remaining_actions": int(remaining_actions),
        "display_only_fallback": bool(display_only_fallback),
        "candidates": items,
        "annotation_status": "pending",
    }


def build_review_page(out, tasks, manifest):
    review = out / "review"
    review.mkdir(parents=True, exist_ok=True)
    data = json.dumps(tasks, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    html_doc = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Val-Seen Top-5 RGB人工核验</title>
<style>
body{font:15px system-ui,sans-serif;margin:0;background:#f4f6f8;color:#18212b}header{position:sticky;top:0;background:#10202f;color:#fff;padding:12px 18px;z-index:5;display:flex;gap:12px;align-items:center;flex-wrap:wrap}button{padding:7px 11px;border:0;border-radius:5px;cursor:pointer}main{max-width:1400px;margin:auto;padding:16px}.layout{display:grid;grid-template-columns:minmax(360px,1.1fr) minmax(400px,1fr);gap:14px}.card{background:white;padding:14px;border-radius:9px;box-shadow:0 1px 4px #0002}.frame{width:100%;max-width:600px;image-rendering:auto}.candgrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}.cand{border:1px solid #ccd5dc;border-radius:7px;padding:8px}.cand img{width:100%;background:#222}.pending{color:#ad5511;font-weight:bold}.choices label{display:block;padding:6px;border-bottom:1px solid #eee}.actions{margin:12px 0}.hint{color:#52616e;font-size:13px}.scores{font:12px ui-monospace,monospace}textarea{width:100%;min-height:70px}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}@media(max-width:850px){.layout{grid-template-columns:1fr}}
</style><header><strong>CityNav Top-5 RGB候选人工核验</strong><button onclick="prev()">上一条</button><button onclick="next()">下一条</button><button onclick="nextPending()">下一条未核验</button><span id="progress"></span><button onclick="exportJSON()">导出审核JSON</button><label>导入审核JSON <input type="file" id="importFile" accept="application/json"></label></header><main><div class="hint card">审核状态初始均为“待审核”。请仅根据当前真实RGB和候选局部图判断，不显示GT目标位置、Contour或导航成功标签。展示图来自原V2实际记录的位姿；候选位置由冻结Geometry生成。局部ROI按相机Footprint/有效深度遮罩，空白区表示未知。候选位置靠近目标的事实不等于实例身份正确。审核结果仅保存在本浏览器LocalStorage，完成后请导出JSON回传。</div><div id="app"></div></main>
<script>
const DATA=__DATA__, KEY='citynav_v2_visual_review_20261107_v1';let cursor=0;let labels=load();function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){return {}}}function save(){localStorage.setItem(KEY,JSON.stringify(labels));progress()}
function esc(s){return String(s??'').replace(/[&<>"']/g,x=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[x]))}function current(){return DATA[cursor]}function progress(){let done=Object.values(labels).filter(x=>x&&x.status==='reviewed').length;document.getElementById('progress').textContent=`${cursor+1}/${DATA.length} · 已审核 ${done} · 待审核 ${DATA.length-done}`}
function setv(k,v){let d=labels[current().index]||{status:'pending'};d[k]=v;d.status='reviewed';d.reviewed_at=new Date().toISOString();labels[current().index]=d;save()}
function render(){const t=current(),old=labels[t.index]||{};let cand=t.candidates.map(c=>{let sc=c.scales;let score=c.score_full?`full cosine(20/40m): ${c.score_full.join(' / ')}<br>appearance cosine: ${c.score_appearance.join(' / ')}`:'当前候选无双尺度匹配分数';return `<div class="cand"><b>候选 ${c.rank}</b> · logit ${c.logit.toFixed(3)}<br><span class="hint">ΔX ${c.dx_to_top1}m, ΔY ${c.dy_to_top1}m</span><div>${[20,40].map(e=>sc[String(e)].path?`<img src="${esc(sc[String(e)].path)}"><small>${e}m ROI · 有效支持 ${(100*sc[String(e)].support).toFixed(1)}% · ${sc[String(e)].valid?'有效':'支持不足'}</small>`:`<p>候选 ${c.rank}：${e}m ROI不在当前图像有效区域</p>`).join('')}</div><div class="scores">${score}</div></div>`}).join('');document.getElementById('app').innerHTML=`<h2>任务 ${t.index} <span class="pending">${old.status==='reviewed'?'已审核':'待审核'}</span></h2><div class="layout"><section class="card"><h3>原始指令</h3><p>${esc(t.instruction)}</p><p class="hint">真实观测 step ${t.step}；沿原策略轨迹剩余动作 ${t.remaining_actions}。若假设该步改变候选，之后原轨迹RGB不再视为有效导航结果。</p><img class="frame" src="${esc(t.frame)}"><p class="hint">圈号对应Geometry Top-5；候选位置可能在视野外。</p><div class="actions"><b>视觉可辨认性</b><div class="choices">${[['identifiable','正确目标实例可辨认'],['same_class','仅同类物体可见/无法区分实例'],['occluded','遮挡或无法辨认'],['no_evidence','没有有效目标视觉证据']].map(([v,l])=>`<label><input type="radio" name="visibility" value="${v}" ${old.visibility===v?'checked':''} onchange="setv('visibility','${v}')"> ${l}</label>`).join('')}</div></div><div class="actions"><b>若能辨认，请选出与指令匹配的候选</b><div class="choices">${t.candidates.map(c=>`<label><input type="radio" name="choice" value="${c.rank}" ${String(old.selected_candidate)===String(c.rank)?'checked':''} onchange="setv('selected_candidate',${c.rank})"> 候选 ${c.rank}</label>`).join('')}<label><input type="radio" name="choice" value="none" ${old.selected_candidate==='none'?'checked':''} onchange="setv('selected_candidate','none')"> 无法选定/都不是</label></div></div><label>置信度 <select onchange="setv('confidence',this.value)"><option value="">未选</option>${['低','中','高'].map(x=>`<option ${old.confidence===x?'selected':''}>${x}</option>`).join('')}</select></label><p>备注<textarea onchange="setv('note',this.value)">${esc(old.note||'')}</textarea></p><button onclick="save()">保存当前标注</button></section><section class="card"><h3>Geometry Top-5候选ROI</h3><div class="candgrid">${cand}</div></section></div>`;progress()}
function next(){cursor=Math.min(DATA.length-1,cursor+1);render()}function prev(){cursor=Math.max(0,cursor-1);render()}function nextPending(){let n=DATA.findIndex((x,i)=>i>cursor&&(!labels[x.index]||labels[x.index].status!=='reviewed'));if(n<0)n=DATA.findIndex(x=>!labels[x.index]||labels[x.index].status!=='reviewed');if(n>=0)cursor=n;render()}
function exportJSON(){let payload={schema:'citynav-v2-visual-review-v1',manifest:__MANIFEST__,annotations:labels};let b=new Blob([JSON.stringify(payload,null,2)],{type:'application/json'});let a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='citynav_visual_review_annotations.json';a.click();URL.revokeObjectURL(a.href)}
document.getElementById('importFile').addEventListener('change',e=>{let f=e.target.files[0];if(!f)return;let rd=new FileReader();rd.onload=()=>{let v=JSON.parse(rd.result),a=v.annotations||{};labels={...labels,...a};save();render()};rd.readAsText(f)});render();
</script></html>'''
    html_doc = html_doc.replace("__DATA__", data).replace("__MANIFEST__", json.dumps(manifest, ensure_ascii=False))
    html_doc = html_doc.replace(
        "真实观测 step ${t.step}；沿原策略轨迹剩余动作",
        "真实观测 step ${t.step}；${t.display_only_fallback?'该起始帧仅供浏览，原轨迹中未找到Top-1与替代候选共同有效的比较帧；':''}沿原策略轨迹剩余动作",
    )
    (review / "index.html").write_text(html_doc, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sbf-source", type=Path, required=True)
    parser.add_argument("--expanded-run", type=Path, required=True,
                        help="existing expanded V2 full_navigation directory")
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--stage1-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int, default=0,
                        help="smoke-only cap; zero means all 644 failures + all Top-1-correct controls")
    parser.add_argument("--shuffle-permutations", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "private_rows").mkdir(exist_ok=True)
    (args.output / "review" / "assets").mkdir(parents=True, exist_ok=True)

    run_rows_dir = args.expanded_run / "rows"
    diag_rows = read_csv(args.diagnostics / "per_episode_attribution.csv")
    hist_rows = {int(r["index"]): r for r in read_csv(args.diagnostics / "historical_rgbd_projection_proxy.csv")}
    source_run = json.loads((args.diagnostics / "diagnostic_summary.json").read_text())
    assert len(diag_rows) == 2470 and source_run["n"] == 2470
    for idx in range(2470):
        if not (run_rows_dir / f"{idx:05d}.json").is_file():
            raise FileNotFoundError(run_rows_dir / f"{idx:05d}.json")

    failures = [r for r in diag_rows if r["sr"] == "0" and r["osr"] == "0"
                and r["initial_nms_recall_at_5"] == "1"]
    controls = []
    for row in diag_rows:
        nms = json.loads(row["spatial_nms_top20_posthoc"])
        if nms and float(nms[0]["distance_to_gt_center_m"]) <= 20.0:
            controls.append(row)
    if len(failures) != 644:
        raise ValueError(f"expected the frozen 644 failure cohort, got {len(failures)}")
    review_indices = set(stratified_review_indices(failures, hist_rows, seed=42))
    selected_rows = failures + controls
    if args.max_episodes:
        selected_rows = selected_rows[:args.max_episodes]
    if len({int(r["index"]) for r in selected_rows}) != len(selected_rows):
        raise ValueError("failure/control cohorts unexpectedly overlap")

    stage1 = torch.load(args.stage1_checkpoint, map_location="cpu", weights_only=True)
    protocol = stage1.get("protocol", {})
    if not protocol.get("backbone") or not protocol.get("backbone_revision"):
        raise ValueError("Stage1 checkpoint lacks frozen SigLIP2 revision")
    manifest = {
        "seed": 42,
        "source_commit": source_run.get("source_run_commit"),
        "val_seen_n": 2470,
        "failure_cohort": "sr=0, osr=0, initial spatial-NMS recall@5=1 (post-hoc 20m location proxy)",
        "failure_cohort_n": len(failures),
        "top1_location_proxy_control_n": len(controls),
        "processed_n": len(selected_rows),
        "review_sample_n": len(review_indices),
        "review_sample_seed": 42,
        "review_sampling": "200 failures stratified by target class proxy, historical projected-size proxy and post-hoc Top-5 target-location rank; private sampling labels are not shown in HTML",
        "candidate_source": "first five entries of frozen spatial_nms_top20_posthoc; selection/crops use only candidate XY and Geometry logit",
        "observation_source": "only poses in frozen V2 closed-loop rows, rendered to actual 384x384 RGB-D at each logged current pose",
        "roi": {"extents_m": [20.0, 40.0], "output_size": 384,
                "minimum_valid_support": 0.85, "center_depth_valid_fraction": 0.5,
                "invalid_pixels": "masked to zero; unknown is never treated as a negative"},
        "visual_model": {"name": protocol["backbone"], "revision": protocol["backbone_revision"],
                         "stage1_checkpoint_sha256": file_sha256(args.stage1_checkpoint),
                         "score": "frozen SigLIP2 cosine; 20m/40m ROI x full instruction/appearance query",
                         "rerank_rule": "switch only if an alternative beats original Top-1 in all four score views; else abstain"},
        "shuffle_control": {"permutations_per_episode": args.shuffle_permutations,
                            "seed": 42, "within_episode_supported_candidate_images_only": True},
        "geometry_checkpoint_sha256": source_run["closedloop_pair"]["manifest"]["geometry_checkpoint_sha256"],
        "controller_checkpoint_sha256": source_run["closedloop_pair"]["manifest"]["controller_checkpoint_sha256"],
        "prior_navigation_run": str(args.expanded_run),
        "prior_metrics_sha256": file_sha256(args.expanded_run / "metrics.json"),
        "attribution_csv_sha256": file_sha256(args.diagnostics / "per_episode_attribution.csv"),
        "history_proxy_csv_sha256": file_sha256(args.diagnostics / "historical_rgbd_projection_proxy.csv"),
        "renderer_sha256": file_sha256(Path("graphnav/renderer.py")),
        "roi_utility_sha256": file_sha256(Path("graphnav/v2_visual_candidate_reranking.py")),
        "scoring_scope": "offline reranking diagnostic only; no altered-policy closed loop or SR claim",
        "ground_truth": "held out from ROI construction, image/text encoding and rerank; used only after scoring for 20m location-proxy outcomes",
        "manual_review": "HTML annotations start pending; no human label is assumed until a JSON export is returned",
    }
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("output manifest differs from this run; use a new output directory")
    atomic_json(manifest, manifest_path)

    sys.path.insert(0, str(args.sbf_source))
    from sbfnav.dataset import CityNavDataset
    from sbfnav.mapdata import GROUND_LEVEL
    dataset = CityNavDataset(args.data_root, "val_seen", version="processed_citynav")
    if len(dataset) != 2470:
        raise ValueError(f"unexpected Refined Val-Seen size {len(dataset)}")
    renderer = CityNavObservedViewRenderer(args.data_root / "rgbd", args.output / "renderer", image_size=384)
    encoder = FrozenSigLIP2(protocol["backbone"], protocol["backbone_revision"],
                            device=args.device, microbatch=16, cache_limit=4096)
    loaded_revision = encoder.revision
    if loaded_revision != protocol["backbone_revision"]:
        raise ValueError(f"SigLIP2 revision mismatch: loaded {loaded_revision}, expected {protocol['backbone_revision']}")

    result_rows = []
    review_tasks = []
    start_all = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(torch.device(args.device)) if str(args.device).startswith("cuda") else None
    for n, attribution in enumerate(selected_rows):
        idx = int(attribution["index"])
        completed = args.output / "private_rows" / f"{idx:05d}.json"
        if completed.exists():
            saved = json.loads(completed.read_text())
            result_rows.append(saved)
            if idx in review_indices and saved.get("review_task"):
                review_tasks.append(saved["review_task"])
            continue

        nav = json.loads((run_rows_dir / f"{idx:05d}.json").read_text())
        ep = dataset[idx]
        if list(ep.episode_id) != list(nav["episode_id"]):
            raise ValueError(f"Val-Seen index/episode mismatch at {idx}")
        poses = np.asarray(nav["poses"], dtype=np.float64)
        raw_nms = json.loads(attribution["spatial_nms_top20_posthoc"])
        # Strip all post-hoc distances/GT fields before passing candidates to crop/scoring.
        public_candidates = [{"xy": [float(x["x"]), float(x["y"])],
                              "logit": float(x["logit"]), "grid_id": int(x["grid_id"])}
                             for x in raw_nms[:5]]
        t_render = time.perf_counter()
        frame_result = first_causal_comparable_frame(
            poses,
            lambda pose: renderer.current_rgbd(ep.map_name, pose, GROUND_LEVEL[ep.map_name]),
            public_candidates,
            extents=(20.0, 40.0), min_candidates=2,
        )
        render_seconds = time.perf_counter() - t_render
        if idx in review_indices and frame_result is None:
            # Keep the review stratum complete. This fallback is display-only;
            # the pilot still abstains because no comparable frame exists.
            first_obs = renderer.current_rgbd(ep.map_name, poses[0], GROUND_LEVEL[ep.map_name])
            fallback_crops = {
                rank: {extent: observed_candidate_roi(first_obs, candidate["xy"], extent)
                       for extent in (20.0, 40.0)}
                for rank, candidate in enumerate(public_candidates)
            }
            frame_result = {"step": 0, "pose": poses[0].tolist(),
                            "observation": first_obs, "crops": fallback_crops,
                            "jointly_valid": []}
            display_only_fallback = True
        else:
            display_only_fallback = False
        visual_score_rows = [None] * len(public_candidates)
        supported = []
        visual_selected = 0
        visual_switched = False
        visual_seconds = 0.0
        candidate_images = []
        feature_positions = []
        text_features = None
        full_prompt = ep.instruction
        app_prompt = appearance_query(full_prompt)
        if frame_result is not None:
            supported = list(frame_result["jointly_valid"])
        if supported:
            for rank in supported:
                for extent in (20.0, 40.0):
                    candidate_images.append(frame_result["crops"][rank][extent]["image"])
                    feature_positions.append((rank, extent))
            t_visual = time.perf_counter()
            image_features = encoder.images(candidate_images).detach().float()
            text_features = encoder.texts([full_prompt, app_prompt]).detach().float()
            torch.cuda.synchronize() if str(args.device).startswith("cuda") else None
            visual_seconds = time.perf_counter() - t_visual
            vectors = {}
            for pos, vector in zip(feature_positions, image_features):
                vectors[pos] = vector
            for rank in supported:
                score_full, score_app = [], []
                for extent in (20.0, 40.0):
                    vector = vectors[(rank, extent)]
                    score_full.append(float(torch.dot(vector, text_features[0]).item()))
                    score_app.append(float(torch.dot(vector, text_features[1]).item()))
                visual_score_rows[rank] = (score_full, score_app)
            score_matrix = np.full((len(public_candidates), 4), np.nan, dtype=np.float64)
            for rank, scores in enumerate(visual_score_rows):
                if scores is not None:
                    score_matrix[rank] = np.asarray([scores[0][0], scores[0][1],
                                                     scores[1][0], scores[1][1]])
            visual_selected, visual_switched = conservative_visual_rerank(
                score_matrix, supported, baseline_index=0)

        # B: original Geometry logits restricted to candidates with common actual RGB-D support.
        geometry_selected = (max(supported, key=lambda k: public_candidates[k]["logit"])
                             if supported else 0)
        # D: randomize image-to-candidate correspondence while preserving the same support mask.
        rng = np.random.default_rng(42 + idx)
        shuffled_selections = []
        if frame_result is not None and len(supported) >= 2 and visual_score_rows[0] is not None:
            original_matrix = np.asarray([
                [s[0][0], s[0][1], s[1][0], s[1][1]] if s is not None else [np.nan] * 4
                for s in visual_score_rows
            ], dtype=np.float64)
            for _ in range(args.shuffle_permutations):
                perm = np.asarray(supported, dtype=int)
                shuffled = perm[rng.permutation(len(perm))]
                if np.array_equal(shuffled, perm) and len(shuffled) > 1:
                    shuffled = np.roll(shuffled, 1)
                matrix = original_matrix.copy()
                for destination, source in zip(perm, shuffled):
                    matrix[destination] = original_matrix[source]
                selected, _ = conservative_visual_rerank(matrix, supported, baseline_index=0)
                shuffled_selections.append(int(selected))

        # GT fields are touched only after the crop/encoding/ranking steps above.
        proxy_distances = [float(x["distance_to_gt_center_m"]) for x in raw_nms[:5]]
        hits = [d <= 20.0 for d in proxy_distances]
        first_projection = hist_rows.get(idx, {}).get("first_long32_short16_proxy_step", "")
        remaining_projection = hist_rows.get(idx, {}).get("remaining_actions_after_proxy", "")
        cohort = "failure_top5_location_proxy" if attribution in failures else "top1_location_proxy_control"
        row = {
            "index": idx, "episode_id": json.dumps(nav["episode_id"], ensure_ascii=False),
            "map_name": ep.map_name, "cohort": cohort,
            "target_category_proxy": attribution["target_category_proxy"],
            "instruction": ep.instruction,
            "baseline_geometry_rank1_hit20_proxy": int(hits[0]),
            "top5_location_recall20_proxy": int(any(hits)),
            "target_location_proxy_rank": next((i + 1 for i, h in enumerate(hits) if h), ""),
            "candidate_distances_gt_center_posthoc": json.dumps(proxy_distances),
            "supported_candidate_count_both_scales": len(supported),
            "supported_candidate_ranks": json.dumps([x + 1 for x in supported]),
            "first_common_observation_step": "" if frame_result is None else int(frame_result["step"]),
            "remaining_logged_actions": "" if frame_result is None else max(0, len(poses) - 1 - int(frame_result["step"])),
            "target_projection_first32x16_proxy_step": first_projection,
            "remaining_actions_at_target_projection_proxy": remaining_projection,
            "A_geometry_top1_rank": 1,
            "B_geometry_visible_only_rank": geometry_selected + 1,
            "C_visual_rank": visual_selected + 1,
            "C_switched_from_geometry": int(visual_switched),
            "B_location_hit20_proxy": int(hits[geometry_selected]),
            "C_location_hit20_proxy": int(hits[visual_selected]),
            "C_corrected_top1_location_proxy": int(not hits[0] and hits[visual_selected]),
            "C_damaged_top1_location_proxy": int(hits[0] and not hits[visual_selected]),
            "visual_score_views": json.dumps(visual_score_rows),
            "D_shuffle_location_hit_rate": "" if not shuffled_selections else float(np.mean([hits[i] for i in shuffled_selections])),
            "D_shuffle_switch_rate": "" if not shuffled_selections else float(np.mean([i != 0 for i in shuffled_selections])),
            "geometry_candidates_render_s": render_seconds,
            "siglip_encode_score_s": visual_seconds,
            "total_episode_diag_s": render_seconds + visual_seconds,
            "review_frame_display_only_fallback": int(display_only_fallback),
            "annotation_status": "pending",
        }

        review_task = None
        if idx in review_indices and frame_result is not None:
            remaining = max(0, len(poses) - 1 - int(frame_result["step"]))
            review_task = save_review_assets(args.output, idx, ep.instruction,
                                             public_candidates, frame_result,
                                             visual_score_rows, remaining,
                                             display_only_fallback)
            review_tasks.append(review_task)
        row["review_task"] = review_task
        atomic_json(row, completed)
        result_rows.append(row)
        if (n + 1) % 25 == 0 or n < 3:
            print(json.dumps({"completed": n + 1, "total": len(selected_rows),
                              "index": idx, "supported": len(supported),
                              "frame_step": None if frame_result is None else frame_result["step"],
                              "encoded_images": encoder.encoded_images,
                              "elapsed_s": round(time.perf_counter() - start_all, 1)},
                             ensure_ascii=False), flush=True)

    # Rebuild the full results from per-episode atomic rows, avoiding duplicate resume rows.
    result_rows = []
    for row in selected_rows:
        idx = int(row["index"])
        result_rows.append(json.loads((args.output / "private_rows" / f"{idx:05d}.json").read_text()))
    result_rows.sort(key=lambda x: x["index"])
    review_tasks = [r["review_task"] for r in result_rows if r.get("review_task")]
    # Recompute derived timing/resolution fields on resume too; this only reads
    # the already logged pose and never selects the pose or candidate using GT.
    for row in result_rows:
        distances = json.loads(row["candidate_distances_gt_center_posthoc"])
        hits = [float(d) <= 20.0 for d in distances]
        b_rank = int(row["B_geometry_visible_only_rank"]) - 1
        row["B_switched_from_geometry"] = int(b_rank != 0)
        row["B_corrected_top1_location_proxy"] = int(not hits[0] and hits[b_rank])
        row["B_damaged_top1_location_proxy"] = int(hits[0] and not hits[b_rank])
        step = row["first_common_observation_step"]
        row["roi_source_side_pixels_20m"] = ""
        row["roi_source_side_pixels_40m"] = ""
        row["logged_height_above_ground_m"] = ""
        if step != "":
            nav = json.loads((run_rows_dir / f"{int(row['index']):05d}.json").read_text())
            pose = nav["poses"][int(step)]
            altitude = float(pose[2]) - float(GROUND_LEVEL[row["map_name"]])
            if altitude > 0:
                row["logged_height_above_ground_m"] = altitude
                row["roi_source_side_pixels_20m"] = 384.0 * 20.0 / (2.0 * altitude)
                row["roi_source_side_pixels_40m"] = 384.0 * 40.0 / (2.0 * altitude)
        if row.get("review_task") is not None:
            row["review_task"]["display_only_fallback"] = bool(row.get("review_frame_display_only_fallback", 0))
        atomic_json(row, args.output / "private_rows" / f"{int(row['index']):05d}.json")
    review_manifest = {k: manifest[k] for k in ("seed", "failure_cohort_n", "review_sample_n", "review_sampling", "observation_source", "roi", "manual_review")}
    build_review_page(args.output, review_tasks, review_manifest)
    review_sample = []
    for row in failures:
        idx = int(row["index"])
        if idx not in review_indices:
            continue
        proxy = hist_rows.get(idx, {})
        review_sample.append({
            "index": idx, "target_category_stratum": row["target_category_proxy"],
            "long32_short16_proxy_stratum": proxy.get("ever_long32_short16_proxy", "unknown"),
            "spatial_nms_posthoc_gt_location_rank_stratum": row.get("initial_nms_gt_rank20m"),
            "target_category_visible_to_reviewer": False,
        })
    write_csv(args.output / "review_manifest.csv", review_sample,
              ["index", "target_category_stratum", "long32_short16_proxy_stratum",
               "spatial_nms_posthoc_gt_location_rank_stratum", "target_category_visible_to_reviewer"])

    # Remove internal page-task metadata from CSV and provide compact per-task rows.
    csv_rows = [{k: v for k, v in row.items() if k != "review_task"} for row in result_rows]
    fields = [k for k in csv_rows[0] if k != "review_task"] if csv_rows else []
    write_csv(args.output / "per_task_visual_reranking.csv", csv_rows, fields)
    summaries = {}
    for cohort in ("failure_top5_location_proxy", "top1_location_proxy_control"):
        rr = [r for r in result_rows if r["cohort"] == cohort]
        if not rr:
            continue
        summaries[cohort] = {"n": len(rr),
            "candidate_shared_rgbd_support": int(sum(int(r["supported_candidate_count_both_scales"]) >= 2 for r in rr)),
            "no_comparable_current_or_history_view": int(sum(int(r["supported_candidate_count_both_scales"]) < 2 for r in rr)),
            "mean_supported_candidates": float(np.mean([r["supported_candidate_count_both_scales"] for r in rr])),
            "mean_first_common_step": float(np.mean([float(r["first_common_observation_step"]) for r in rr if r["first_common_observation_step"] != ""]))
                if any(r["first_common_observation_step"] != "" for r in rr) else None,
            "mean_remaining_actions_after_first_common_view": float(np.mean([float(r["remaining_logged_actions"]) for r in rr if r["remaining_logged_actions"] != ""]))
                if any(r["remaining_logged_actions"] != "" for r in rr) else None,
            "A_geometry_top1_location_proxy": float(np.mean([int(r["baseline_geometry_rank1_hit20_proxy"]) for r in rr])),
            "B_geometry_visible_only_top1_location_proxy": float(np.mean([int(r["B_location_hit20_proxy"]) for r in rr])),
            "B_switch_count": int(sum(int(r["B_switched_from_geometry"]) for r in rr)),
            "B_corrected_proxy_count": int(sum(int(r["B_corrected_top1_location_proxy"]) for r in rr)),
            "B_damaged_proxy_count": int(sum(int(r["B_damaged_top1_location_proxy"]) for r in rr)),
            "C_frozen_siglip_top1_location_proxy": float(np.mean([int(r["C_location_hit20_proxy"]) for r in rr])),
            "C_location_proxy_hit_when_comparable": float(np.mean([
                int(r["C_location_hit20_proxy"]) for r in rr
                if int(r["supported_candidate_count_both_scales"]) >= 2])),
            "C_corrected_proxy_count": int(sum(int(r["C_corrected_top1_location_proxy"]) for r in rr)),
            "C_damaged_proxy_count": int(sum(int(r["C_damaged_top1_location_proxy"]) for r in rr)),
            "C_switch_count": int(sum(int(r["C_switched_from_geometry"]) for r in rr)),
            "D_shuffle_mean_proxy_hit_rate": float(np.mean([float(r["D_shuffle_location_hit_rate"]) for r in rr if r["D_shuffle_location_hit_rate"] != ""]))
                if any(r["D_shuffle_location_hit_rate"] != "" for r in rr) else None,
            "D_shuffle_mean_switch_rate": float(np.mean([float(r["D_shuffle_switch_rate"]) for r in rr if r["D_shuffle_switch_rate"] != ""]))
                if any(r["D_shuffle_switch_rate"] != "" for r in rr) else None,
            "mean_siglip_encoding_s_per_task": float(np.mean([float(r["siglip_encode_score_s"]) for r in rr])),
            "median_siglip_encoding_s_per_task": float(np.median([float(r["siglip_encode_score_s"]) for r in rr])),
            "supported_siglip_n": int(sum(float(r["siglip_encode_score_s"]) > 0 for r in rr)),
            "mean_siglip_encoding_s_when_scored": (float(np.mean([float(r["siglip_encode_score_s"]) for r in rr if float(r["siglip_encode_score_s"]) > 0]))
                                                   if any(float(r["siglip_encode_score_s"]) > 0 for r in rr) else None),
            "mean_total_diagnostic_s_per_episode": float(np.mean([float(r["total_episode_diag_s"]) for r in rr])),
            "median_roi_source_side_pixels_20m": (float(np.median([float(r["roi_source_side_pixels_20m"]) for r in rr if r["roi_source_side_pixels_20m"] != ""]))
                                                   if any(r["roi_source_side_pixels_20m"] != "" for r in rr) else None),
            "median_roi_source_side_pixels_40m": (float(np.median([float(r["roi_source_side_pixels_40m"]) for r in rr if r["roi_source_side_pixels_40m"] != ""]))
                                                   if any(r["roi_source_side_pixels_40m"] != "" for r in rr) else None),
        }
    by_category = {}
    for cohort_name, cohort_rows in (("failure_top5_location_proxy", failures),
                                     ("top1_location_proxy_control", controls)):
        selected_ids = {int(x["index"]) for x in cohort_rows}
        by_category[cohort_name] = {}
        for category in ("Car", "Building", "Ground", "Parking"):
            rr = [x for x in result_rows if x["index"] in selected_ids and x["target_category_proxy"] == category]
            if not rr:
                continue
            by_category[cohort_name][category] = {
                "n": len(rr),
                "co_observed": int(sum(int(x["supported_candidate_count_both_scales"]) >= 2 for x in rr)),
                "C_switch": int(sum(int(x["C_switched_from_geometry"]) for x in rr)),
                "C_location_proxy_correct": int(sum(int(x["C_corrected_top1_location_proxy"]) for x in rr)),
                "C_location_proxy_damaged": int(sum(int(x["C_damaged_top1_location_proxy"]) for x in rr)),
            }
    failure_timing = []
    for row in failures:
        idx = int(row["index"])
        proxy = hist_rows.get(idx, {})
        geo = proxy.get("first_geometric_projection_step", "")
        crisp = proxy.get("first_long32_short16_proxy_step", "")
        remaining = proxy.get("remaining_actions_after_proxy", "")
        failure_timing.append({
            "geometry_at_start": str(geo) == "0",
            "geometry_ever": str(geo) not in ("", "None", "nan"),
            "size_proxy_at_start": str(crisp) == "0",
            "size_proxy_ever": str(crisp) not in ("", "None", "nan"),
            "remaining": None if str(remaining) in ("", "None", "nan") else float(remaining),
        })
    proxy_timing_summary = {
        "n": len(failure_timing),
        "contour_geometric_overlap_at_start_proxy": int(sum(x["geometry_at_start"] for x in failure_timing)),
        "contour_geometric_overlap_at_any_logged_pose_proxy": int(sum(x["geometry_ever"] for x in failure_timing)),
        "long32_short16_pixels_at_start_proxy": int(sum(x["size_proxy_at_start"] for x in failure_timing)),
        "long32_short16_pixels_at_any_logged_pose_proxy": int(sum(x["size_proxy_ever"] for x in failure_timing)),
        "median_remaining_actions_when_size_proxy_first_met": (
            float(np.median([x["remaining"] for x in failure_timing if x["remaining"] is not None]))
            if any(x["remaining"] is not None for x in failure_timing) else None),
        "proxy_only_not_true_visibility": True,
    }
    completed_secs = time.perf_counter() - start_all
    execution_path = args.output / "pilot_execution.json"
    if execution_path.exists():
        execution = json.loads(execution_path.read_text())
    else:
        execution = {
            "wall_seconds": time.perf_counter() - start_all,
            "new_unique_images_encoded": encoder.encoded_images,
            "image_cache_hits": encoder.cache_hits,
            "peak_gpu_allocated_gib": (float(torch.cuda.max_memory_allocated(torch.device(args.device)) / 2**30)
                                        if str(args.device).startswith("cuda") else None),
        }
        atomic_json(execution, execution_path)
    summary = {
        "manifest": manifest,
        "processed_n": len(result_rows),
        "review_gallery_n": len(review_tasks),
        "review_annotations": {"status": "pending", "assumed_human_labels": 0},
        "cohorts": summaries,
        "by_category": by_category,
        "target_geometry_proxy_timing_644": proxy_timing_summary,
        "timing": execution,
        "interpretation": {
            "instance_recognition_accuracy": None,
            "warning": "20m distance to CityNav target center is a location proxy only; CityRefer Polygon projected pixels are also not human instance labels.",
            "pilot_is_closed_loop": False,
            "future_observations_after_a_hypothetical_rerank_used": False,
            "automatic_result_label": "unlabeled SigLIP zero-shot ranking probe; not validated instance recognition",
        },
    }
    atomic_json(summary, args.output / "summary.json")
    make_plot(summary, args.output / "reranking_proxy_results.png")
    write_report(summary, result_rows, args.output / "REPORT_zh.md")
    print(json.dumps({"processed": len(result_rows), "gallery": len(review_tasks),
                      "summary": summaries, "wall_s": completed_secs,
                      "peak_gpu_gib": summary["timing"]["peak_gpu_allocated_gib"]},
                     ensure_ascii=False, indent=2), flush=True)


def make_plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = summary["cohorts"]
    names, labels, values = [], [], []
    for key, title in (("failure_top5_location_proxy", "644 failure proxy"),
                       ("top1_location_proxy_control", "Top-1 correct control")):
        if key not in groups:
            continue
        g = groups[key]
        names += [title] * 3
        labels += ["Geometry Top-1", "Geometry+Observed", "Frozen SigLIP2"]
        values += [g["A_geometry_top1_location_proxy"],
                   g["B_geometry_visible_only_top1_location_proxy"],
                   g["C_frozen_siglip_top1_location_proxy"]]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(values))
    ax.bar(x, values, color=["#46627f", "#8aa2b4", "#3b9c75"] * (len(values)//3))
    ax.set_xticks(x, [f"{n}\n{l}" for n, l in zip(names, labels)])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1 20m target-center proxy hit rate")
    ax.set_title("Frozen visual reranking probe (location proxy, not instance accuracy or SR)")
    ax.grid(axis="y", alpha=.25)
    for i, v in enumerate(values):
        ax.text(i, v + .02, f"{v:.1%}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_report(summary, rows, path):
    cohort = summary["cohorts"]
    lines = ["# Top-5 候选真实 RGB 可辨认性与冻结重排序 Pilot", "",
        "## 结论边界", "",
        "本实验没有人工实例标签。人工审核网页已生成，标注均为待审核；因此‘真正具备可辨认 RGB 证据的数量’目前不能给出。SigLIP2 分数和目标中心 20m 命中只能作为无标签自动探针/位置代理，不是实例识别准确率。没有同时观测到 Geometry Top-1 与至少一个替代候选的帧，或 SigLIP 各视图不一致时，视觉策略保留原 Top-1。", "",
        "历史 RGB 仅来自原 V2 真实闭环已记录的姿态，渲染并评分时只用当步 384×384 RGB-D；没有把假设改选后的旧轨迹未来图像用于 SR 结论。该结果是离线回放诊断，不是新策略闭环。", "",
        "## 数据和方法", "",
        f"- Val-Seen 完整缓存：2,470 条。目标位置 Top-5 失败代理：{summary['manifest']['failure_cohort_n']} 条；Top-1 位置正确控制组：{summary['manifest']['top1_location_proxy_control_n']} 条；本次处理：{summary['processed_n']} 条。",
        f"- 人工审核页：{summary['review_gallery_n']} 条，状态待审核；目前人工标签数为 0。",
        "- ROI 为模型 Top-5 世界位置在当步 384 RGB-D 中的局部裁剪，20m/40m 两个范围；相机覆盖和有效深度支持不足的像素置黑。候选中心/相机足迹支持不够则视为未知。",
        "- A：原 Geometry Top-1。B：只在真实RGB-D支持的候选中按原Geometry logit排序（观测覆盖对照）。C：冻结SigLIP2比较20/40m ROI与完整指令/外观query，仅当同一替代候选在四种余弦分数上全部高于原Top-1时切换。D：同episode内打乱支持候选的RGB特征对应，20个固定随机排列。无校准阈值、无训练、无验证集选参。",
        "- 所有命中均为候选中心距目标GT中心≤20m的位置代理。候选生成没有使用GT；GT仅在特征计算和排序完成后用于回看该选择是否靠近目标位置。",
        "", "## 结果", "",
        "|组别|n|Top-1/替代候选共同RGB-D支持|无可比较帧|A Geometry Top-1|B Geometry+观测覆盖|C Frozen RGB|C位置代理纠正|C位置代理破坏|D Shuffle位置命中|平均首个共同观测步|首个共同观测后剩余动作|",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, title in (("failure_top5_location_proxy", "644 Top-5失败代理"),
                       ("top1_location_proxy_control", "Top-1正确控制组")):
        if key not in cohort:
            continue
        g = cohort[key]
        hit_count_a = int(round(g["A_geometry_top1_location_proxy"] * g["n"]))
        hit_count_b = int(round(g["B_geometry_visible_only_top1_location_proxy"] * g["n"]))
        hit_count_c = int(round(g["C_frozen_siglip_top1_location_proxy"] * g["n"]))
        shuf = "—" if g["D_shuffle_mean_proxy_hit_rate"] is None else f"{g['D_shuffle_mean_proxy_hit_rate']:.1%}"
        step = "—" if g["mean_first_common_step"] is None else f"{g['mean_first_common_step']:.2f}"
        rem = "—" if g["mean_remaining_actions_after_first_common_view"] is None else f"{g['mean_remaining_actions_after_first_common_view']:.2f}"
        lines.append(f"|{title}|{g['n']}|{g['candidate_shared_rgbd_support']}|{g['no_comparable_current_or_history_view']}|{hit_count_a} ({g['A_geometry_top1_location_proxy']:.1%})|{hit_count_b} ({g['B_geometry_visible_only_top1_location_proxy']:.1%})|{hit_count_c} ({g['C_frozen_siglip_top1_location_proxy']:.1%})|{g['C_corrected_proxy_count']}|{g['C_damaged_proxy_count']}|{shuf}|{step}|{rem}|")
    fail_group = cohort.get("failure_top5_location_proxy", {})
    good_group = cohort.get("top1_location_proxy_control", {})
    raw_a = int(good_group.get("n", 0))
    raw_c = raw_a - int(good_group.get("C_damaged_proxy_count", 0)) + int(fail_group.get("C_corrected_proxy_count", 0))
    test_n = int(good_group.get("n", 0)) + int(fail_group.get("n", 0))
    lines.extend(["", "### 按目标类别拆分（仍是20m位置代理）", "",
        "|组别|类别|n|双尺度RGB-D共同支持|RGB切换次数|位置代理纠正|位置代理破坏|",
        "|---|---|---:|---:|---:|---:|---:|"])
    for cohort_key, cohort_title in (("failure_top5_location_proxy", "Top-5失败"),
                                     ("top1_location_proxy_control", "Top-1正确控制")):
        for category, values in summary.get("by_category", {}).get(cohort_key, {}).items():
            lines.append(f"|{cohort_title}|{category}|{values['n']}|{values['co_observed']}|{values['C_switch']}|{values['C_location_proxy_correct']}|{values['C_location_proxy_damaged']}|")
    lines.extend(["", "### 候选辨认与人工审核", "",
        "当前可报告的视觉覆盖率是‘Top-1与替代候选是否同时在同一实际RGB-D画面内达到双尺度有效支持’，不是目标真实可辨认率。该量由候选位置、相机Footprint和有效Depth决定。CityRefer Polygon/目标Contour投影尺寸只作为抽样和时间代理，未作为视觉模型输入，也不证明屋顶/地面实例边界重合。",
        "人工审核页面 `review/index.html` 展示原始指令、实际观测帧、Geometry Top-5 标记及同帧候选 ROI；不显示目标GT/Contour。支持四类可辨认性、候选选择、置信度和备注，保存到浏览器LocalStorage，可导入/导出 JSON。该页面初始状态全为待审核，尚无人工标签返回。",
        "在644条失败中，310条（48.1%）存在Top-1与至少一个替代候选的同帧双尺度有效RGB-D支持；另334条在原轨迹上没有这种可比较帧。这是候选位置的几何/传感器支持量，不是‘真实目标可辨认’数量。真正实例可辨认数仍然**未知**，不能用自动分数臆测。网页为分层样本，不可直接外推644总体比例，除非按抽样权重做统计。",
        "", "### Top-1是否被破坏", "",
        f"C在644失败组中把20m位置命中从0提高到 {fail_group.get('C_corrected_proxy_count', 0)} 条（{fail_group.get('C_corrected_proxy_count', 0)}/{fail_group.get('n', 1)}={fail_group.get('C_corrected_proxy_count', 0)/max(1,fail_group.get('n',1)):.1%}；仅在310条可比较帧中为 {fail_group.get('C_location_proxy_hit_when_comparable',0):.1%}）。但在Top-1位置正确控制组，C破坏了 {good_group.get('C_damaged_proxy_count', 0)}/{good_group.get('n', 1)} 条（{good_group.get('C_damaged_proxy_count', 0)/max(1,good_group.get('n',1)):.1%}；在共同支持的控制任务中为 {good_group.get('C_damaged_proxy_count', 0)}/{max(1,good_group.get('candidate_shared_rgbd_support',0))}）。按这两个已测组合并，位置命中由 {raw_a}/{test_n} 降为 {raw_c}/{test_n}，净少 {raw_a-raw_c} 条；这不是闭环SR变化。纠正/破坏都是GT中心20m代理，不代表实例身份。",
        "B使用原 Geometry logit 在可观测候选内重排；由于比较帧的定义要求Top-1本身可观测，而候选已按Geometry分数排序，B与A实际上相同（全体B切换数为0）。它是排序实现的无视觉控制，不会提供新增语义。",
        "", "### 时间与因果性", "",
        f"在644失败中，首个Top-1/替代候选共同支持帧平均出现在step {fail_group.get('mean_first_common_step',0):.2f}，原策略该时刻平均剩余 {fail_group.get('mean_remaining_actions_after_first_common_view',0):.2f} 个动作；而已知Contour的32×16px尺寸代理历史命中540条，起点49条，首次达到后原轨迹剩余动作中位数3。该尺寸代理不代表真正可辨认。首个共同支持帧来自原策略已记录序列；若在该帧改选，旧策略之后RGB对新动作轨迹不再因果有效，因此没有报告SR/OSR。",
        "", "### 起点与后续观测", "",
        f"在644条失败中，历史记录里的目标Contour几何覆盖代理在起点命中 {summary['target_geometry_proxy_timing_644']['contour_geometric_overlap_at_start_proxy']} 条，任一已记录位姿命中 {summary['target_geometry_proxy_timing_644']['contour_geometric_overlap_at_any_logged_pose_proxy']} 条；32×16px尺寸代理起点命中 {summary['target_geometry_proxy_timing_644']['long32_short16_pixels_at_start_proxy']} 条、历史任一时刻命中 {summary['target_geometry_proxy_timing_644']['long32_short16_pixels_at_any_logged_pose_proxy']} 条。首次达到该尺寸代理后，原轨迹剩余动作中位数为 {summary['target_geometry_proxy_timing_644']['median_remaining_actions_when_size_proxy_first_met']}。这些均是Contour/像素覆盖代理，不能回答实例是否能辨认。",
        "", "### 推理成本", "",
        f"- 全部任务真实首轮诊断耗时：{summary['timing']['wall_seconds']:.1f}s；新增SigLIP图像编码：{summary['timing']['new_unique_images_encoded']}；图像缓存命中：{summary['timing']['image_cache_hits']}；GPU峰值分配：{summary['timing']['peak_gpu_allocated_gib'] if summary['timing']['peak_gpu_allocated_gib'] is not None else 'CPU'} GiB。",
        f"- 仅有共同候选支持时，平均 SigLIP 编码/打分耗时约为失败组 {cohort['failure_top5_location_proxy']['mean_siglip_encoding_s_when_scored']:.3f}s、控制组 {cohort['top1_location_proxy_control']['mean_siglip_encoding_s_when_scored']:.3f}s；离线逐任务总诊断时间（扫描旧轨迹、渲染和编码）均值分别为 {cohort['failure_top5_location_proxy']['mean_total_diagnostic_s_per_episode']:.3f}s / {cohort['top1_location_proxy_control']['mean_total_diagnostic_s_per_episode']:.3f}s，不能视作在线推理延迟。候选ROI从384px帧裁剪后模型仍输入384px，因此是局部放大而非新增细节。支持任务中20m/40m ROI源帧边长中位数分别为失败组 {cohort['failure_top5_location_proxy']['median_roi_source_side_pixels_20m']:.1f}/{cohort['failure_top5_location_proxy']['median_roi_source_side_pixels_40m']:.1f}px，控制组 {cohort['top1_location_proxy_control']['median_roi_source_side_pixels_20m']:.1f}/{cohort['top1_location_proxy_control']['median_roi_source_side_pixels_40m']:.1f}px。",
        "", "## 决策", "",
        f"当前冻结 SigLIP2 的自动规则位置代理纠正57、破坏103；测试的两个组内净减少46/1,842（{(raw_c-raw_a)/max(1,test_n):.2%}）。失败组中SigLIP的条件位置代理命中18.4%，Shuffle负对照为10.4%；但正确Top-1控制组的条件位置代理破坏15.6%，说明自动排序有位置相关信号，却不够可靠地保留正确候选。**暂不把视觉重排序作为V4核心改进，不运行完整2470闭环。**下一步先完成200条人工审核；若人工确认的可辨认子集存在明确纠正且正确实例误选很少，再设计新的因果闭环，否则先改善观测采样/目标表征。无可靠视觉证据时保留Geometry Top-1。",
        "", "## 产物", "",
        "- `per_task_visual_reranking.csv`：全部试验任务逐任务排序/支持/时间/位置代理结果。",
        "- `review/index.html`：交互式人工核验页面。",
        "- `review_manifest.csv`：人工审核抽样分层信息（类别、投影尺寸代理、目标位置代理排名；不作为推理输入）。",
        "- `reranking_proxy_results.png`：A/B/C位置命中代理对照。",
        "- `manifest.json`：输入、权重、渲染器和实验配置哈希。",
        ""])
    Path(path).write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
