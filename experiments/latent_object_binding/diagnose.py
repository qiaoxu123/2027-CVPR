"""Label-separated coverage, public-geometry, and symmetry diagnostics."""
import collections, gzip, json, time
import numpy as np
import shapely, torch
from experiments.semantic_polygon_grounding.audit import save
from .resources import OUT, baseline, primary_features, regions
from .language import active_variables


def read_rows(path):
    with gzip.open(path, 'rt') as f:
        return [json.loads(line) for line in f]


def public_diagnostics():
    start=time.time();a3,source,graphs,proof=baseline();summary={};rows=[]
    for split in ('train','dev'):
        selected=[q for q in source['queries'] if q['split']==split]
        audit=[]
        for q in selected:
            g=graphs[q['map_name']]
            for label in q['labels']:
                i=g.byid[str(label['target_object_id'])];point=np.asarray(label['target_world'])[:2];shape=g.shapes[i]
                bd=float(shape.boundary.distance(shapely.Point(point)))
                audit.append(dict(query_index=q['query_index'],map_name=q['map_name'],object_id=g.ids[i],type=g.types[i],shape_kind=shape.geom_type,inside=bool(shape.covers(shapely.Point(point))),boundary_distance=bd if np.isfinite(bd) else None,outside_distance=float(shape.distance(shapely.Point(point))),public_position_error=float(np.linalg.norm(point-np.asarray(g.objects[i]['position'][:2]))),centroid_error=float(np.linalg.norm(point-g.centers[i]))))
        summary[split]=dict(queries=len(selected),episodes=sum(len(q['family_ids']) for q in selected),maps=len({q['map_name'] for q in selected}),unique_objects=len({(r['map_name'],r['object_id']) for r in audit}),label_records=len(audit),multiple_positive_queries=sum(len(q['labels'])>1 for q in selected),public_position_exact=sum(r['public_position_error']<1e-6 for r in audit),inside_or_boundary=sum(r['inside'] for r in audit),outside=sum(not r['inside'] for r in audit),point_fallback_labels=sum(r['shape_kind']=='Point' for r in audit),outside_distance_quantiles=np.quantile([r['outside_distance'] for r in audit],[0,.5,.9,1]).tolist(),centroid_error_quantiles=np.quantile([r['centroid_error'] for r in audit],[0,.5,.9,1]).tolist())
        rows+=audit
    save(OUT/'geometry_label_audit.json',dict(summary=summary,rows=rows,baseline_proof=proof,public_protocol='all object positions/polygons/types; labels audited after inference only',seconds=time.time()-start))
    return summary


def support_diagnostics():
    manifest=json.loads((OUT/'full_geometry/complete.json').read_text());labels={l['query_index']:l for l in torch.load(OUT/'full_geometry/labels.pt',weights_only=True)};rows=[]
    for file in manifest['chunks']:
        for d in torch.load(OUT/'full_geometry'/file,weights_only=True,mmap=True):
            lang=d['language'];active,factors=active_variables(lang);v=lang.get('variables',[]);references=[i for i in active if i!=0];pairs=[]
            # Exchanging the same two Between anchors does not change the
            # geometric field. The raw colour-to-instance identity need not
            # be resolved when the unordered pair itself is fixed.
            f=factors[0] if len(factors)==1 else None
            if f and f['kind']=='between' and f['subject']==0 and len(references)==2:
                group=collections.defaultdict(list)
                for j,a in enumerate(d['assignments']):
                    if len(a)==2:group[tuple(sorted(a.values()))].append(j)
                for ids,indices in group.items():
                    if len(indices)>1:
                        delta=float((d['log_phi'][indices[0]].float()-d['log_phi'][indices[1]].float()).abs().max())
                        pairs.append(dict(object_indices=list(ids),field_max_delta=delta))
            rows.append(dict(query_index=d['query_index'],sample_id=str(d['query_index']),map_name=d['map_name'],split=d['split'],family_ids=d['family_ids'],instruction=d['instruction'],variables=len(v),active_variables=len(active),factors=len(factors),relative_unknown_factors=sum(f.get('frame_unknown',False) for f in lang.get('factors',[])),named_references=sum(bool(v[i]['named_ids']) for i in references),anonymous_references=sum(not bool(v[i]['named_ids']) for i in references),target_colors=v[0].get('colors',[]) if v else [],known_mass=d['known_mass'],support_mass=d['support_mass'],gate=.25*d['known_mass'],unknown=d['unknown'],assignments=len(d['assignments']),symmetric_pairs=pairs,target_candidate_count=d['target_features'].shape[0],target_positive_count=len(labels[d['query_index']]['positive_objects'])))
    summary={}
    for split in ('train','dev'):
        rs=[r for r in rows if r['split']==split];summary[split]=dict(queries=len(rs),with_connected_references=sum(r['active_variables']>1 for r in rs),with_named_references=sum(r['named_references']>0 for r in rs),with_anonymous_references=sum(r['anonymous_references']>0 for r in rs),with_target_color=sum(bool(r['target_colors']) for r in rs),relative_frame_unknown=sum(r['relative_unknown_factors']>0 for r in rs),zero_known=sum(r['known_mass']==0 for r in rs),known_mass_quantiles=np.quantile([r['known_mass'] for r in rs],[0,.25,.5,.75,.9,1]).tolist(),with_symmetric_exchange_pairs=sum(bool(r['symmetric_pairs']) for r in rs),symmetric_exchange_pairs=sum(len(r['symmetric_pairs']) for r in rs),max_symmetric_field_delta=max([p['field_max_delta'] for r in rs for p in r['symmetric_pairs']]+[0]),maximum_target_candidates=max(r['target_candidate_count'] for r in rs),minimum_target_candidates=min(r['target_candidate_count'] for r in rs))
    save(OUT/'binding_support_audit.json',dict(summary=summary,rows=rows,reference_binding_recall=None,reference_truth='No independent anonymous reference identities; exact public names are supplied bindings, not learnt accuracy.',target_candidate_recall=1.,scope_alternatives='Primary hypothesis only; omitted alternatives contribute Unknown, not falsely claimed full inference.'))
    return summary


def observed_diagnostics():
    summaries={};out=[];maps=torch.load(OUT/'full_geometry/maps.pt',weights_only=True)
    for split in ('train','dev'):
        f,states,manifest=primary_features(split);lookup={s['metadata']['sample_id']:s for s in states};data=torch.load(OUT/'observed'/f'{split}_public.pt',weights_only=True,mmap=True);labels={l['sample_id']:l for l in torch.load(OUT/'observed'/f'{split}_labels.pt',weights_only=True)};base={r['sample_id']:r for r in read_rows(OUT/f'observed_{split}_A0.jsonl.gz')};rs=[]
        for d in data:
            meta=d['metadata'];sid=meta['sample_id'];state=lookup[sid];visible=regions(state);lab=labels[sid];tid=lab['target_object_id'];r=visible.get(tid);p=d['versions']['real'];geom=d['versions']['geometry'];g=maps[meta['map_name']];i=g['ids'].index(tid)
            rs.append(dict(sample_id=sid,family_id=meta['family_id'],map_name=meta['map_name'],target_id=tid,static_wrong=not bool(base[sid]['metrics']['goal_r1']),target_visible=r is not None,target_native_side=r['native_min_side'] if r else None,target_depth_support=r['depth_support'] if r else None,observed_objects=len(visible),candidate_objects=len(g['ids']),pixel_scored=d['coverage']['real'].get(0,d['coverage']['real'].get('0',{})).get('pixel_scored',0),target_rgb_scores=p['target_features'][i,-2:].float().tolist(),reference_assignments_changed=p['assignments']!=geom['assignments'],known_mass=p['known_mass'],known_mass_geometry=geom['known_mass'],target_colors=p['language']['variables'][0].get('colors',[]) if p['language'].get('variables') else [],type=f['plan']['labels'][sid].get('object_type',state['metadata'].get('object_type')),cutoff_step=meta['cutoff_step']))
        summaries[split]=dict(families=len(rs),maps=len({r['map_name'] for r in rs}),objects=len({(r['map_name'],r['target_id']) for r in rs}),static_wrong=sum(r['static_wrong'] for r in rs),target_visible=sum(r['target_visible'] for r in rs),static_wrong_and_target_visible=sum(r['static_wrong'] and r['target_visible'] for r in rs),independent_wrong_visible_objects=len({(r['map_name'],r['target_id']) for r in rs if r['static_wrong'] and r['target_visible']}),rgb_changes_reference_proposals=sum(r['reference_assignments_changed'] for r in rs),with_pixel_evidence=sum(r['pixel_scored']>0 for r in rs),total_observed_object_occurrences=sum(r['observed_objects'] for r in rs),total_pixel_object_occurrences=sum(r['pixel_scored'] for r in rs),known_mass_quantiles=np.quantile([r['known_mass'] for r in rs],[0,.25,.5,.75,.9,1]).tolist())
        out+=rs;del f
    save(OUT/'observation_audit.json',dict(summary=summaries,rows=out,one_prefix_per_family=True,quality_is_geometric_weak_proxy=True,attribute_oracle_verified_labels=0))
    return summaries


if __name__=='__main__':
    torch.set_num_threads(1)
    print('geometry',public_diagnostics(),flush=True);print('support',support_diagnostics(),flush=True);print('observation',observed_diagnostics(),flush=True)
