"""FP32-only geometry-positive-control -> TRAIN synchronized counterfactuals.

No labels/private target files, RGB, Val or Test are consumed. No training.
"""
import argparse
from collections import Counter
import hashlib
import json
import re
from pathlib import Path
import time
import numpy as np
import rasterio
import shapely
import torch
from graphnav.checkpointing import atomic_json,file_sha256
from graphnav.direct_target_data import load_model,relation_geometry
from graphnav.direct_target_language import PREDICATES
from graphnav.direct_target_runtime import bind_public_anchors
from graphnav.landmark_projection import world_to_frame,frame_to_world
from graphnav.memory import ObservedRGBMemory
from graphnav.landmark_region_supervision import named_union
from graphnav.semantic_geometry import prior_and_rule_fields,geometry_distance
from graphnav.semantic_graph import prepare_semantic_graph
from graphnav.candidate_geometry import FullCandidateDataset,collate_full
from graphnav.full_experiment import _batch_inputs
from graphnav.token_language import FrozenBERTTokens
from scripts.analyze_instruction_landmark_association import PublicNames
from scripts.relation_counterfactual_metrics import PAIRS,PHRASES,AXES,replacement_span,replace_relation,relation_regions,score_pair,effective_logits

WORDS=('north','south','east','west','inside','outside')


def coordinate_audit(tasks,data_root,official):
    maps=[];max_error=0.
    for name in sorted({t['map_name'] for t in tasks}):
        with rasterio.open(data_root/'rgbd'/f'{name}.tif') as ds:
            transform=ds.transform
            maps.append(dict(map_name=name,CRS=ds.crs.to_string() if ds.crs else None,
                             affine=list(transform)[:6],column_plus_X=transform.a>0 and transform.d==0,
                             row_minus_Y=transform.e<0 and transform.b==0))
            center=np.array([(ds.bounds.left+ds.bounds.right)/2,(ds.bounds.bottom+ds.bounds.top)/2])
        for yaw in (0.,np.pi/2,np.pi,-np.pi/2):
            footprint=ObservedRGBMemory.view_corners(center,yaw,50.)
            xy=center+np.array([[0,0],[10,0],[0,10],[-10,-10]])
            pixels=world_to_frame(xy,footprint,(384,384,3))
            restored=frame_to_world(pixels,footprint,(384,384,3))
            max_error=max(max_error,float(np.abs(restored-xy).max()))
    assert max_error<.001
    return dict(TRAIN_maps=len(maps),maps=maps,max_world_pixel_roundtrip_error_m=max_error,
        numeric_engineering_convention='grid columns +X, grid rows +Y; raster columns +X, raster rows -Y; yaw0 +X',
        geographic_mapping_verified=False,geographic_direction_accuracy=None,
        reason='Local TIF has no CRS or documented local-to-geographic rotation/origin. Raster image-up is not independent geographic-north proof.',
        official_sources={str(p):file_sha256(p) for p in [official/'rasterize.py',official/'gsamllavanav/space.py',official/'gsamllavanav/observation/cropclient.py']},
        primary_links=['https://github.com/water-cookie/citynav/blob/main/rasterize.py',
            'https://github.com/water-cookie/citynav/blob/main/gsamllavanav/space.py',
            'https://github.com/ATR-DBI/CityRefer/blob/main/README.md'],
        GT_used=False,axis_flip_applied=False)


def synthetic_control(output):
    axis=(np.arange(28)+.5)/28*200-100;yy,xx=np.meshgrid(axis,axis,indexing='ij');world=np.stack((xx.ravel(),yy.ravel()),-1)
    contour=np.array([[-20,-15],[20,-15],[20,15],[-20,15]],float);polygon=shapely.Polygon(contour)
    regions=relation_regions(world,polygon);fields={};compiler={};support={};checks=[]
    for word in WORDS:
        # Diagnostic geometry reference, NOT a new deployed network or method.
        fields[word]=np.where(regions[word],0.,-30.)
        record=[dict(name='Anchor Hall',object_type='Building',contour=((contour+100)/200).tolist())]
        graph=prepare_semantic_graph(record,[],f'A building {PHRASES[word]} Anchor Hall.',[-100,-100,100,100],max_landmarks=16,max_roads=0,max_relations=8)
        prior,rules,_=prior_and_rule_fields(world,graph['entities'],graph['relations'],sigma_m=25.)
        compiler[word]=prior+rules;support[word]=[r.kind for r in graph['relations']]
    for pair,(a,b) in PAIRS.items():
        result=score_pair(fields[a],fields[b],world,regions,pair)
        assert result['joint_mass_correct']
        if pair!='IO':assert result['centroid_direction_correct']
        checks.append(dict(pair=pair,analytic_reference=result,
            existing_rule_baseline=score_pair(compiler[a],compiler[b],world,regions,pair)))
    # Exact rectangle distances, independent of language/model and GT targets.
    probes=np.array([[0,0],[20,0],[30,0],[0,25],[30,25]],float)
    expected=np.array([0,0,10,10,np.sqrt(200)])
    distances=geometry_distance(probes,contour,is_road=False)
    assert np.allclose(distances,expected,atol=1e-5)
    np.savez_compressed(output/'synthetic_fields.npz',world=world,polygon=contour,
                        **{'analytic_'+w:fields[w] for w in WORDS},**{'compiler_'+w:compiler[w] for w in WORDS})
    result=dict(candidate_count=784,coordinate_frame='synthetic +Y defined north, +X east; NOT proof of real city geography',
        polygon=contour.tolist(),probability_controls=checks,compiler_relations=support,
        distance_probes=probes.tolist(),measured_distances_m=distances.tolist(),expected_distances_m=expected.tolist(),
        synthetic_geometry_passed=True,inside_outside_operator_missing=not support['inside'] and not support['outside'],
        reference_prior_is_diagnostic_only=True)
    atomic_json(result,output/'synthetic_control.json')


def visible_relation(tokenizer,text,span):
    tok=tokenizer(text,max_length=96,truncation=True,return_offsets_mapping=True)
    offsets=[(s,e) for s,e in tok['offset_mapping'] if e>s and s<span[1] and e>span[0]]
    return bool(offsets and min(s for s,e in offsets)<=span[0] and max(e for s,e in offsets)>=span[1]),tok['input_ids']


def main():
    p=argparse.ArgumentParser()
    for key in ('source','data-root','official-source','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Immutable experiment; use a fresh output')
    a.output.mkdir(parents=True);torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False
    tasks=json.loads((a.source/'inventory_validated/public_tasks.json').read_text())
    records=[json.loads(l) for l in (a.source/'cache_aligned/public_static.jsonl').read_text().splitlines()]
    assert all(r['target_xy']==[0.,0.] for r in records), 'Only dummy public scaffolding allowed'
    catalog=json.loads((a.data_root/'cityrefer/objects.json').read_text());public_names=PublicNames(catalog)
    coordinate=coordinate_audit(tasks,a.data_root,a.official_source);atomic_json(coordinate,a.output/'coordinate_audit.json')
    synthetic_control(a.output)
    checkpoint=a.source/'training_aligned/best.pt';weight_sha=file_sha256(checkpoint)
    bundle=torch.load(checkpoint,map_location='cpu',weights_only=True)
    model=load_model(bundle).eval().requires_grad_(False)
    assert all(p.dtype==torch.float32 for p in model.parameters())
    args=bundle['config']['geometry_args']
    bert=FrozenBERTTokens(args['text_backbone'],device='cuda',max_length=args['max_text_length'])
    assert all(p.dtype==torch.float32 for p in bert.model.parameters())
    summaries=[];skipped=[];prediction_data={};start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode(),torch.autocast('cuda',enabled=False):
        for task in tasks:
            if len(task['anchors'])!=1:
                skipped.append(dict(index=task['index'],reason='multi-reference scope not uniquely replaceable'));continue
            mentions=public_names.match(task['map_name'],task['instruction'])
            try:span=replacement_span(task,mentions)
            except ValueError as err:
                skipped.append(dict(index=task['index'],reason=str(err)));continue
            bound,_=bind_public_anchors(task['anchors'],catalog[task['map_name']]);poly=named_union(bound[0]['contours'])
            texts={};kept={};token_ids={};changed_records=[]
            for word in WORDS:
                text,new_span=replace_relation(task['instruction'],span,word);texts[word]=text
                kept[word],token_ids[word]=visible_relation(bert.tokenizer,text,new_span)
                assert text[:span[0]]==task['instruction'][:span[0]]
                assert text[new_span[1]:]==task['instruction'][span[1]:]
                changed_records.append(dict(records[task['index']],instruction=text))
            path=a.output/'current_public_queries.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in changed_records))
            ds=FullCandidateDataset(path,grid_size=28,max_landmarks=args['max_landmarks'],max_roads=args['max_roads'],
                max_relations=args['max_relations'],prior_sigma_m=args['prior_sigma_m'],rule_sigma_m=args['rule_sigma_m'])
            assert len(ds)==6
            batch=collate_full([ds[i] for i in range(6)])
            static,tokens,mask,nodes=_batch_inputs(batch,bert,'cuda','geometry')
            geometry=relation_geometry(dict(task,anchors=bound),static['grid_xy'][0].cpu().numpy())
            ids=torch.tensor([[PREDICATES.index(w)] for w in WORDS],device='cuda')
            model.reset_episode()
            result=model.forward_direct_target(static_inputs=static,tokens=tokens,token_mask=mask,node_emb=nodes,
                relation_geometry=geometry[None].expand(6,-1,-1,-1).cuda(),relation_ids=ids,
                relation_valid=torch.ones_like(ids,dtype=torch.bool),observed=None,target_text=None)
            geom=result['geometry_field'].belief_logits;relation=result['relation_field'].belief_logits
            assert geom.dtype==relation.dtype==tokens.dtype==torch.float32
            g=geom.cpu().numpy();r=relation.cpu().numpy();bounds=np.array(task['world_bounds'])
            world=bounds[:2]+static['grid_xy'][0].cpu().numpy()*(bounds[2:]-bounds[:2]);regions=relation_regions(world,poly)
            branches={w:dict(Geometry=g[i],Relation=r[i],Effective=effective_logits(g[i],r[i],w)) for i,w in enumerate(WORDS)}
            compiled={w:[q.kind for q in ds._prepared(i)[0]['relations']] for i,w in enumerate(WORDS)}
            # Leave the rest of each instruction intact, including any extra
            # conditions. Mark them rather than pretending all rewrites feasible.
            rest=task['instruction'][:span[0]]+' '+task['instruction'][span[1]:]
            extra=bool(re.search(r'\b(north|south|east|west|inside|outside|behind|front|left|right|between)\b',rest,re.I))
            group=task['map_name']+'::'+','.join(bound[0]['landmark']['public_ids'])
            for pair,(wa,wb) in PAIRS.items():
                text_seen=kept[wa] and kept[wb];token_changed=token_ids[wa]!=token_ids[wb]
                row=dict(index=task['index'],map_name=task['map_name'],subset=task['subset'],group=group,pair=pair,
                    original_predicate=bound[0]['original_predicate'],target_category=task['structured']['target']['category'],
                    single_ID=len(bound[0]['landmark']['public_ids'])==1,relation_in_token_window=text_seen,
                    token_IDs_changed=token_changed,other_spatial_terms=extra,
                    original_instruction=task['instruction'],texts={wa:texts[wa],wb:texts[wb]},edited_span=list(span),
                    predicates={wa:PREDICATES.index(wa),wb:PREDICATES.index(wb)},
                    token_SHA256={w:hashlib.sha256(json.dumps(token_ids[w]).encode()).hexdigest() for w in (wa,wb)},
                    compiler_relations={w:compiled[w] for w in (wa,wb)},
                    compiler_rule_difference=float((static['rules'][WORDS.index(wa)]-static['rules'][WORDS.index(wb)]).abs().max()),
                    runtime_fallback=pair!='IO',counterfactual_GT_available=False,geographic_direction_correctness=None,
                    metrics={m:score_pair(branches[wa][m],branches[wb][m],world,regions,pair)
                             for m in ('Geometry','Relation','Effective')})
                summaries.append(row)
            prediction_data[f'{task["index"]:04d}_geometry']=g
            prediction_data[f'{task["index"]:04d}_relation']=r
            if task['index']%100==0:print(json.dumps(dict(query=task['index'],pairs=len(summaries),seconds=time.perf_counter()-start)),flush=True)
    assert file_sha256(checkpoint)==weight_sha
    np.savez_compressed(a.output/'all_logits.npz',**prediction_data)
    atomic_json(summaries,a.output/'pairs.json');atomic_json(skipped,a.output/'skipped.json')
    atomic_json(dict(source='Revised TRAIN public inventory only',queries=len({r['index'] for r in summaries}),
        paired_conditions=len(summaries),forward_batches=len({r['index'] for r in summaries}),
        model_conditions=6*len({r['index'] for r in summaries}),
        checkpoint_sha256=weight_sha,weights_unchanged=True,training=False,RGB=False,Val_Test_accessed=False,
        actual_GT_files_read=False,dtype='FP32 BERT/Geometry/Relation, TF32 disabled',
        BERT_name=bert.name,BERT_revision=bert.revision,max_text_length=bert.max_length,
        geographic_mapping_verified=coordinate['geographic_mapping_verified'],seconds=time.perf_counter()-start,
        peak_GPU_GiB=torch.cuda.max_memory_allocated()/2**30,
        predicate_training_counts=json.loads((a.source/'inventory_validated/audit.json').read_text())['relations'],
        code_sha256={str(p):file_sha256(p) for p in [Path(__file__),Path('scripts/relation_counterfactual_metrics.py'),
             Path('graphnav/semantic_geometry.py'),Path('graphnav/unified_pgvb.py'),Path('graphnav/direct_target_runtime.py')]},
        public_inventory_sha256=file_sha256(a.source/'inventory_validated/public_tasks.json')),
        a.output/'manifest.json')
    print('Finished',len(summaries),'paired interventions',flush=True)


if __name__=='__main__':main()
