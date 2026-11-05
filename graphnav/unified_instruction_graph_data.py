"""Weak public-text graph preprocessing and source-aligned Polygon binding.

Parser proposals are NOT reviewed direct/indirect ground truth. GT coords,
teacher poses and future images never enter graph compilation or forward.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .instruction_relation_graph import tensorize_relation_graph

MAX_NODES=32
MAX_EDGES=64


def empty_graph():
    return dict(nodes=[],edges=[],paths=[],target_id=None,
                target_GT_used=False,world_north_verified=False)


def row_signature(row):
    return hashlib.sha256(json.dumps(
        [row["map_name"],row["instruction"]],ensure_ascii=False).encode()).hexdigest()


def read_graphs(path,records):
    lines=[json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines()
           if s.strip()]
    if len(lines)!=len(records):
        raise ValueError("Graph/source query count mismatch")
    graphs=[]
    for i,(line,row) in enumerate(zip(lines,records)):
        if (line["index"]!=i or line["row_signature"]!=row_signature(row)
                or line["map_name"]!=row["map_name"]):
            raise ValueError(f"Graph/source alignment failure at {i}")
        graph=line.get("graph")
        if not isinstance(graph,dict) or graph.get("target_GT_used") is not False:
            raise ValueError("Every graph must explicitly forbid GT")
        graphs.append(graph)
    return graphs


def _node_phrase(item):
    phrase=str(item.get("category") or item.get("phrase") or "")
    attrs=[]
    for attr in item.get("attributes",[]):
        value=attr.get("value",attr.get("text",""))
        if value:attrs.append(str(value))
    if attrs:phrase+="; "+", ".join(sorted(set(attrs),key=str.lower))
    if item.get("role")=="public_landmark" and item.get("public_names"):
        phrase="; ".join(sorted(set(map(str,item["public_names"])),key=str.lower))
    return phrase or "[UNK]"


def _encode_groups(groups,bert,device,capacity):
    b=len(groups)
    data=torch.zeros((b,capacity,768),device=device)
    entries=[(i,j,s) for i,group in enumerate(groups) for j,s in enumerate(group)]
    if entries:
        emb=bert.encode([s for _,_,s in entries]).float()
        if emb.shape!=(len(entries),768):
            raise ValueError("Unexpected frozen BERT feature shape")
        for k,(i,j,_) in enumerate(entries):data[i,j]=emb[k]
    return data


def graph_input_batch(graphs,source_records,bert,device,*,
                      max_nodes=MAX_NODES,max_edges=MAX_EDGES,
                      entity_capacity=36):
    """Public path-endpoint ID binding to existing static Polygon slots.

    Polygon order is the original FullCandidateDataset record landmarks order.
    The graph branch does not newly select, rank, crop or correct those
    pre-existing source references using a target label. The original
    processed CityNav source may itself contain annotation-selected refs;
    that inherited benchmark limitation must be disclosed in reports.
    """
    if not graphs or len(graphs)!=len(source_records):
        raise ValueError("Invalid graph/source pairing")
    packs=[];node_phrases=[];edge_phrases=[]
    assignment=np.zeros((len(graphs),max_nodes,entity_capacity),np.float32)
    for i,(graph,row) in enumerate(zip(graphs,source_records)):
        if graph.get("target_GT_used") is not False:
            raise ValueError("GT-free graph provenance required")
        pack=tensorize_relation_graph(graph,max_nodes=max_nodes,max_edges=max_edges)
        packs.append(pack)
        node_phrases.append([_node_phrase(x) for x in graph["nodes"]])
        edge_phrases.append([
            str(x.get("source_predicate") or x.get("predicate") or "[UNK]")
            for x in graph["edges"]
        ])
        terminals={p["nodes"][-1] for p in graph.get("paths",[])
                   if p.get("hops",0)>=1 and p.get("nodes")}
        refs=[str(x.get("id","")) for x in row.get("landmarks",[])]
        if len(refs)>entity_capacity:
            raise ValueError("Public references exceed entity capacity")
        for j,node in enumerate(graph["nodes"]):
            if node.get("role")!="public_landmark" or node.get("id") not in terminals:
                continue
            ids={str(k) for k in node.get("public_ids",[])}
            for k,oid in enumerate(refs):
                if oid and oid in ids:assignment[i,j,k]=1.
    packed={
        key:(torch.from_numpy(np.stack([x[key] for x in packs])).to(device)
             if isinstance(packs[0][key],np.ndarray) else
             torch.tensor([x[key] for x in packs],device=device))
        for key in packs[0] if key not in ("node_phrases","edge_phrases")
    }
    return dict(
        packed=packed,
        node_text=_encode_groups(node_phrases,bert,device,max_nodes),
        edge_text=_encode_groups(edge_phrases,bert,device,max_edges),
        node_anchor=torch.tensor(assignment,device=device),
    )


def prepare_graphs(source_path,objects_path,output_path,*,
                   expected_records=21878,spacy_model="en_core_web_sm"):
    import spacy
    from scripts.analyze_instruction_landmark_association import PublicNames
    from graphnav.structured_navigation import parse_instruction
    from graphnav.instruction_relation_graph import compile_relation_graph

    out=Path(output_path)
    if out.exists():raise FileExistsError("Refuse overwrite")
    source=Path(source_path)
    rows=[json.loads(s) for s in source.read_text().splitlines() if s.strip()]
    if len(rows)!=expected_records:
        raise ValueError(f"Require full {expected_records} TRAIN rows, got {len(rows)}")
    public=json.loads(Path(objects_path).read_text())
    names=PublicNames(public)
    nlp=spacy.load(spacy_model,disable=["ner"])
    out.parent.mkdir(parents=True,exist_ok=True)
    counts=Counter()
    with out.open("w",encoding="utf-8") as stream:
        for i,row in enumerate(rows):
            if row["map_name"] not in public:
                raise ValueError("Missing map in public CityRefer catalog")
            try:
                parsed=parse_instruction(row["instruction"],row["map_name"],nlp,names)
                graph=compile_relation_graph(parsed,public[row["map_name"]],
                                             row["world_bounds"],
                                             world_north_verified=False)
                if len(graph["nodes"])>MAX_NODES or len(graph["edges"])>MAX_EDGES:
                    raise ValueError("Graph exceeds capacity")
                summary=graph["summaries"]
                status=("direct_and_indirect" if summary["direct_paths"] and summary["indirect_paths"]
                        else "direct" if summary["direct_paths"] else
                        "indirect" if summary["indirect_paths"] else
                        "no_target_landmark_path")
            except (KeyError,ValueError,TypeError,IndexError) as exc:
                graph=empty_graph()
                status="unresolved"
                counts["error_"+type(exc).__name__]+=1
            counts[status]+=1
            stream.write(json.dumps(dict(index=i,map_name=row["map_name"],
                row_signature=row_signature(row),status=status,graph=graph),
                ensure_ascii=False)+"\n")
    manifest=dict(protocol="citynav-weak-graph-v1",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        objects_sha256=hashlib.sha256(Path(objects_path).read_bytes()).hexdigest(),
        graph_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        n=len(rows),status=dict(counts),parser=spacy_model,
        target_GT_used=False,world_north_verified=False,
        parser_proposals_not_verified=True)
    out.with_suffix(".manifest.json").write_text(
        json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    return manifest
