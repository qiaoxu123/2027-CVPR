#!/usr/bin/env python3
"""TRAIN-only, conservative scope extraction and offline geometry audit.
GT is read ONLY after parsing/name resolution. No model inputs or weights.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from scipy.stats import circmean

WORDS = 'left right front behind back rear above below top bottom north south east west'.split()
TOKEN = re.compile(r'\b('+'|'.join(WORDS)+r')\b', re.I)
AXES = np.array([0.,90.,180.,270.])
# Only egocentric hypotheses have predeclared offsets; world axes are inferred.
OFFSETS = {'left':90.,'right':270.,'front':0.,'behind':180.,'back':180.,'rear':180.}
SEED=20261106

def norm(s):
    return re.sub(r'[^a-z0-9]+',' ',s.lower().replace('’',"'")).strip()

def angle(x,y): return math.degrees(math.atan2(y,x))%360

def distance(a,b): return np.abs((np.asarray(a)-b+180)%360-180)

def wilson(k,n):
    if not n:return [None,None]
    p=k/n; z=1.959963984540054; den=1+z*z/n
    mid=(p+z*z/(2*n))/den; rad=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return [float(mid-rad),float(mid+rad)]

def polygon(obj):
    c=obj.get('contour',[])
    if len(c)<3:return None
    p=Polygon(c)
    # Invalid contours are excluded, never silently fixed.
    return p if p.is_valid and p.area>0 else None

_SPAN_CACHE={}
def name_spans(text, objects):
    """Exact normalized full names; optional 'the' removed, punctuation ignored.
    Word offsets refer to original text, aliases/fuzzy matching intentionally absent.
    """
    key=(id(objects),text)
    if key in _SPAN_CACHE:return _SPAN_CACHE[key]
    toks=list(re.finditer(r'[a-z0-9]+',text.lower().replace('’',"'")))
    vals=[t[0] for t in toks]; hits=[]
    idx=collections.defaultdict(list)
    for oid,o in objects.items():
        n=norm(o.get('name','')); n=re.sub(r'^the ','',n)
        if n:idx[n].append(oid)
    for n,ids in idx.items():
        nt=n.split(); L=len(nt)
        for j in range(len(vals)-L+1):
            if vals[j:j+L]==nt:
                hits.append({'start':toks[j].start(),'end':toks[j+L-1].end(), 'ids':ids,'name':n})
    # Prefer longest full name if nested (School names etc).
    result=sorted([h for h in hits if not any(g['start']<=h['start'] and g['end']>=h['end'] and g['end']-g['start']>h['end']-h['start'] for g in hits)],key=lambda h:h['start'])
    _SPAN_CACHE[key]=result
    return result

def sentence_start(text,pos):
    # St. and Ltd. are not sentence boundaries.
    t=re.sub(r'\b(St|Rd|Ltd)\.',lambda m:m[0][:-1]+' ',text,flags=re.I)
    ends=[m.end() for m in re.finditer(r'[.!?]\s+',t[:pos])]
    return ends[-1] if ends else 0

def parse(text,match,objects,target_type):
    s,e=match.span(); w=match[0].lower(); hits=name_spans(text,objects)
    r={'word':w,'span_start':s,'span_end':e,'scope':'unknown','status':'excluded','reason':'no_explicit_relation','ref_id':None,'inverse':False,'binding_rule':None}
    def reject(why,scope='unknown'):r.update(reason=why,scope=scope);return r
    if any(h['start']<=s and h['end']>=e for h in hits):return reject('direction_in_proper_name','nonrelational')
    aft=text[e:]; bef=text[:s]; tail=text[s:]
    if re.match(r'\s+(?:next|beside|near|along|across|on|by|at|outside|under|over|before|after|in\s+front|behind)\b',aft,re.I) and w=='right':return reject('right_emphasis','nonrelational')
    if re.search(r'\b(?:turn|turning|take|taking|go|going|heading|traveling|travelling|facing)\s+(?:a\s+|to\s+the\s+)?$',bef,re.I):return reject('motion_or_heading','nonrelational')
    if re.match(r'\s+(?:yard|door|bumper|roof|license|plate|seat|seats|section|wheel|windows?|end)\b',aft,re.I) and w in ['front','back','rear','top','bottom']:return reject('part_attribute_or_extremum','nonrelational')
    # Relation marker must start at this token, not nearest-landmark heuristics.
    pat={
      'left':r'left\s+(?:hand\s+)?(?:side\s+)?of\s+',
      'right':r'right\s+(?:hand\s+)?(?:side\s+)?of\s+',
      'front':r'front\s+of\s+', 'back':r'back\s+of\s+', 'rear':r'rear\s+of\s+',
      'behind':r'behind\s+', 'above':r'above\s+', 'below':r'below\s+',
      'top':r'top\s+of\s+', 'bottom':r'bottom\s+of\s+',
      'north':r'north\s+(?:side\s+)?of\s+', 'south':r'south\s+(?:side\s+)?of\s+',
      'east':r'east\s+(?:side\s+)?of\s+', 'west':r'west\s+(?:side\s+)?of\s+'}[w]
    rel=re.match(pat,tail,re.I)
    if not rel:
        # Explicit inverse: "Behind it is NAME" only with target-headed previous sentence.
        inv=re.match(r'(?:behind|above|below)\s+(?:it|this\s+(?:car|building|house))\s+(?:is|lies|stands)\s+(?:the\s+)?',tail,re.I)
        if inv and sentence_start(text,s)>0:
            first=text[:sentence_start(text,s)]
            if not re.match(r'(?:this\s+is\s+|it\s+is\s+|the\s+|a\s+|an\s+)',first,re.I):return reject('inverse_antecedent_uncertain')
            if re.search(r'\b(?:there\s+(?:is|are)|it\s+has|which\s+has)\b',first,re.I):return reject('inverse_antecedent_uncertain')
            nh=[h for h in hits if h['start']>=s+inv.end() and h['start']<=s+inv.end()+4]
            if len(nh)==1:
                h=nh[0]
                if len(h['ids'])!=1:return reject('duplicate_landmark_name')
                r.update(ref_id=h['ids'][0],inverse=True,scope='direct',status='bound',reason='',binding_rule='explicit_inverse_pronoun',ref_name=h['name']); return r
        if re.search(r'\b(?:from|counting|counted|row|lane|column)\s+(?:the\s+)?$',bef,re.I) or w in ['top','bottom']:return reject('ordinal_extremum_or_part','nonrelational')
        if w in ['front','back','rear']:return reject('part_or_unresolved_relation')
        return reject('no_explicit_relation')
    r['relation_phrase']=rel[0]
    after=s+rel.end()
    # Compound directions are not treated as a standalone cardinal label.
    if re.search(r'\b(?:upper|lower|top|bottom|north|south|diagonally)\s+(?:and\s+)?$',bef,re.I) or re.match(r'\s+(?:east|west)\b',aft,re.I):return reject('compound_direction_not_single_axis')
    # Reference NP is either exact full named landmark or article + named landmark.
    nh=[h for h in hits if h['start']>=after and re.fullmatch(r'\s*(?:the\s+)?',text[after:h['start']],re.I)]
    if len(nh)!=1:
        later=[h for h in hits if after<h['start']<after+100]
        lead=text[after:later[0]['start']] if later else ''
        if later and re.search(r'\b(?:in|on|of|behind|beside|near|next|left|right|front|above|below|at)\b',lead,re.I):return reject('indirect_reference_np','indirect')
        return reject('reference_not_uniquely_named')
    h=nh[0];r['ref_candidate_ids']=h['ids'];r['ref_name']=h['name']
    if len(h['ids'])!=1:return reject('duplicate_landmark_name')
    # A named landmark modifying a different reference head is not a direct pair.
    if re.match(r'\s+(?:road\s+car\s+park|car\s+park|parking\s+lot|yard|entrance|wall|side|corner|driveway)\b',text[h['end']:],re.I):return reject('reference_is_landmark_subregion','indirect')
    # Reject coordination where additional referents are unbound.
    if re.match(r'\s+(?:and|or)\s+(?!to\b)',text[h['end']:],re.I):return reject('coordinated_reference_uncertain')
    start=sentence_start(text,s); prefix=text[start:s]
    # Strip the relation's function words, leaving target NP/predicate.
    prefix=re.sub(r'(?:\b(?:to|on|at|in)\s+(?:the\s+)?|\b(?:directly|just|immediately|far)\s+)+$','',prefix,flags=re.I).strip()
    # A relation at sentence start qualifies only when followed by explicit target.
    if not prefix:
        rest=text[h['end']:]
        if not re.match(r'(?:\s+(?:building|road|church|hall))?\s*,\s*(?:this|the|it\s+is|a|an)\b',rest,re.I):return reject('fronted_relation_subject_uncertain')
        r['binding_rule']='fronted_target_clause'
    else:
        if start>0 and not re.match(r'(?:it|this\s+(?:car|building|house|vehicle)|the\s+(?:car|building|house|vehicle))\b',prefix,re.I):return reject('later_sentence_subject_uncertain')
        # First sentence NP must be a target description, not a context subject.
        if not re.match(r'(?:this\s+is\s+|it\s+is\s+|this\s+|the\s+|a\s+|an\s+)',prefix,re.I):return reject('subject_uncertain')
        if re.search(r'\b(?:left|right|above|below|behind|front|north|south|east|west)\b',prefix,re.I):return reject('preceding_relation_scope_uncertain','indirect')
        # Embedded head nouns / preposition chains change the relation's subject.
        if re.search(r'\b(?:with|has|have|having|there|whose|between|surrounded|facing|off|along|beside|near|across|under|over|on|in|at|of|from|next|by|outside|nearest|closest)\b',prefix,re.I):return reject('embedded_or_indirect_subject','indirect')
        # The head noun must match target category (not GT spatial location).
        cat=target_type.lower()
        allowed={'building':r'building|house|home|shed|warehouse|structure|roof|church', 'car':r'car|vehicle|van|truck|bus', 'ground':r'ground|area|land|field|lawn|garden|tree', 'parking':r'parking|park|lot'}
        expr=next((v for k,v in allowed.items() if k in cat),r'building|car|ground|parking')
        if not re.search(r'\b(?:'+expr+r')\b',prefix,re.I) and not re.match(r'it\s+(?:is|sits|lies|stands)\b',prefix,re.I):return reject('target_head_not_established')
        r['binding_rule']='simple_target_head'
    if w in ['top','bottom']:return reject('part_or_region_not_external_pair','nonrelational')
    if w=='front' and not re.search(r'\bin\s+(?:the\s+)?$',bef,re.I):return reject('front_part_not_in_front')
    if w in ['back','rear'] and not re.search(r'\b(?:at|to)\s+(?:the\s+)?$',bef,re.I):return reject('back_rear_part_or_unknown')
    r.update(ref_id=h['ids'][0],scope='direct',status='bound',reason='');return r

def geometry(tgt,ref,gt,tol=1.):
    a=np.array(gt[:2],float);b=np.array(ref['position'][:2],float);d=a-b
    p,q=polygon(tgt),polygon(ref)
    if p is None or q is None:return {'geometry_status':'missing_or_invalid_polygon','dx':float(d[0]),'dy':float(d[1]),'angle_world':angle(*d),'target_xy':a.tolist(),'ref_xy':b.tolist(),'target_polygon':tgt.get('contour',[]),'ref_polygon':ref.get('contour',[])}
    r={'dx':d[0],'dy':d[1],'dz':gt[2]-ref['position'][2], 'angle_world':angle(*d), 'distance_m':float(np.linalg.norm(d)),
       'target_xy':a.tolist(),'ref_xy':b.tolist(),'target_polygon':tgt['contour'],'ref_polygon':ref['contour'],
       'target_area':p.area,'ref_area':q.area,'polygon_distance':p.distance(q), 'intersection_area':p.intersection(q).area,
       'target_gt_vs_object_m':float(np.linalg.norm(a-np.array(tgt['position'][:2]))),
       'target_gt_in_polygon':p.buffer(tol).covers(__import__('shapely').geometry.Point(*a)),
       'geometry_status':'valid'}
    if np.linalg.norm(d)<=tol or p.intersects(q) or p.distance(q)<=tol:r['geometry_status']='ambiguous_overlap_or_near_boundary'
    for ax in AXES:
        u=np.array([np.cos(np.deg2rad(ax)),np.sin(np.deg2rad(ax))])
        # Entire target contour must lie beyond entire reference contour.
        r[f'gap_{int(ax)}']=float(np.min(np.array(tgt['contour'])@u)-np.max(np.array(ref['contour'])@u))
    rect=np.array(q.minimum_rotated_rectangle.exterior.coords)[:4];edges=np.roll(rect,-1,axis=0)-rect; lens=np.linalg.norm(edges,axis=1);v=edges[np.argmax(lens)]
    r['ref_unsigned_axis_deg']=angle(*v)%180;r['ref_elongation']=float(max(lens)/min(lens))
    r['ref_heading_status']='unavailable_no_directed_heading_annotation'
    return r

def cluster_ci(df,correct,seed=SEED,B=1000):
    if len(df)==0:return [None,None]
    # Resample target objects; descriptions of the same object stay together.
    g=pd.DataFrame({'cluster':df['cluster'].values,'correct':np.asarray(correct,dtype=float)}).groupby('cluster')['correct'].agg(['sum','count'])
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(g),(B,len(g)));num=g['sum'].values[idx].sum(1);den=g['count'].values[idx].sum(1)
    return np.quantile(num/den,[.025,.975]).tolist()

def summarize(df,angle_col,expected=None):
    n=len(df)
    if not n:return {'n':0}
    vals=df[angle_col].to_numpy();counts=[int(np.sum(distance(vals,a)<=45)) for a in AXES]
    ax=float(AXES[np.argmax(counts)]) if expected is None else float(expected)
    err=distance(vals,ax);correct=err<=45
    z=np.mean(np.exp(1j*np.deg2rad(vals)))
    return {'n':n,'clusters':df.cluster.nunique(),'axis_deg':ax,'axis_fit':'best_on_same_TRAIN_descriptive' if expected is None else 'predeclared',
       'correct':int(correct.sum()),'consistency':float(np.mean(correct)), 'wilson95':wilson(int(correct.sum()),n),
       'cluster_bootstrap95':cluster_ci(df,correct),'mean_abs_error_deg':float(err.mean()),'median_abs_error_deg':float(np.median(err)),
       'resultant_R':float(abs(z)),'mean_angle_deg':float(np.angle(z,deg=True)%360),
       'plus_X':counts[0],'plus_Y':counts[1],'minus_X':counts[2],'minus_Y':counts[3]}

def dump(path,obj):path.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data-root',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path(__file__).parent/'results');args=ap.parse_args();root=args.data_root;out=args.out;out.mkdir(parents=True,exist_ok=True)
    files=[root/'original_citynav/citynav_train_seen.json',root/'cityrefer_original/objects.json',root/'cityrefer_original/processed_descriptions.json']
    assert all('val' not in f.name and 'test' not in f.name for f in files)
    episodes,objects,processed=[json.loads(f.read_text()) for f in files]
    maps=sorted({f"{e['area']}_block_{e['block']}" for e in episodes});rows=[];mismatch=[]
    for i,ep in enumerate(episodes):
        mapname=f"{ep['area']}_block_{ep['block']}";oid=str(ep['object_ids'][0]);ann=ep['ann_ids'][0];text=ep['descriptions'][0];tgt=objects[mapname][oid]
        desc=tgt['descriptions']; aligned=ann<len(desc) and text==desc[ann]
        if not aligned:mismatch.append({'episode':i,'map':mapname,'target_id':oid,'ann_id':ann,'text':text})
        for m in TOKEN.finditer(text):
            r=parse(text,m,objects[mapname],tgt['object_type'])
            r.update(episode=i,map=mapname,target_id=oid,ann_id=ann,instruction=text,target_category=tgt['object_type'],cluster=f'{mapname}:{oid}',description_aligned=aligned)
            r['occurrence_id']=f'{mapname}:{oid}:{ann}:{m.start()}'
            if r['status']=='bound':
                if not aligned:r.update(status='excluded',reason='description_annotation_mismatch')
                elif r['ref_id']==oid:r.update(status='excluded',reason='self_reference')
                else:
                    ref=objects[mapname][r['ref_id']];r['ref_category']=ref['object_type']
                    r.update(geometry(tgt,ref,ep['target_positions'][0]));r['gt_xyz']=ep['target_positions'][0];r['ref_xyz']=ref['position']
                    pose=ep['trajectory'][0];r['initial_pose']=pose
                    yaw=angle(pose[3],pose[4]) if len(pose)==6 and np.hypot(pose[3],pose[4])>1e-6 else (np.degrees(pose[3])%360 if len(pose)==5 else None)
                    r['initial_yaw_deg']=yaw
                    r['angle_yaw']=(r.get('angle_world',0)-yaw)%360 if yaw is not None else None
                    rb=np.array(ref['position'][:2])-np.array(pose[:2]);r['los_reference_deg']=angle(*rb) if np.linalg.norm(rb)>1 else None
                    r['angle_los']=(r.get('angle_world',0)-r['los_reference_deg'])%360 if r['los_reference_deg'] is not None else None
                    # Inverse relations are normalized to target REL reference by reversing WORD,
                    # not geometric coordinates; word remains original for occurrence accounting.
                    r['semantic_word']={'behind':'front','above':'below','below':'above'}.get(r['word'],r['word']) if r['inverse'] else r['word']
                    r['status']='valid' if r['geometry_status']=='valid' else 'ambiguous'
                    r['reason']='' if r['status']=='valid' else r['geometry_status']
            rows.append(r)
    extended=[]
    for i,ep in enumerate(episodes):
        for m in re.finditer(r'\b(?:leftmost|rightmost|northern|southern|eastern|western|northwest|northeast|southwest|southeast|upper|lower|backyard)\b',ep['descriptions'][0],re.I):
            extended.append({'episode':i,'lexeme':m[0].lower(),'span_start':m.start(),'instruction':ep['descriptions'][0],'analysis':'supplemental_lexical_sweep_not_single_word_primary_set'})
    dump(out/'extended_lexemes.json',extended)
    dump(out/'occurrences.json',rows)
    df=pd.DataFrame(rows);df.drop(columns=[c for c in ['target_polygon','ref_polygon'] if c in df]).to_csv(out/'occurrences.csv',index=False)
    geo=[r for r in rows if 'angle_world' in r];dump(out/'pairs_geometry.json',geo)
    valid=df[df.status=='valid'].copy();valid['semantic_word']=valid.semantic_word.fillna(valid.word)
    # One original occurrence appears exactly once, never duplicate weighting by repeated text.
    assert df.occurrence_id.nunique()==len(df)
    totals=[]
    for w in WORDS:
        sub=df[df.word==w]; reasons=sub[sub.status!='valid'].reason.value_counts().to_dict()
        totals.append({'word':w,'occurrences':len(sub),'episodes':sub.episode.nunique(),'unique_instruction_texts':sub.instruction.nunique(),'bound_pairs':int(sub.status.isin(['valid','ambiguous']).sum()),'valid':int((sub.status=='valid').sum()),'excluded_or_ambiguous':int((sub.status!='valid').sum()),'reasons':reasons})
    dump(out/'word_counts.json',totals);pd.DataFrame(totals).to_csv(out/'word_counts.csv',index=False)
    stats=[]
    for w in WORDS:
        sub=valid[valid.semantic_word==w]
        for hyp,col in [('A_world','angle_world'),('B_initial_yaw','angle_yaw'),('C_start_landmark_LOS','angle_los'),('D_landmark_heading',None)]:
            supp=sub if hyp=='A_world' else sub[sub[col].notna()] if col and w in OFFSETS else sub.iloc[:0]
            st=summarize(supp,col,OFFSETS[w] if hyp!='A_world' and w in OFFSETS else None) if len(supp) else {'n':0}
            st.update(word=w,hypothesis=hyp,coverage_of_valid=len(supp)/len(sub) if len(sub) else 0,coverage_of_mentions=len(supp)/len(df[df.word==w]) if len(df[df.word==w]) else 0,
                      unavailable_reason='no_reliable_directed_heading' if hyp=='D_landmark_heading' else 'no_predeclared_planar_egocentric_semantics' if hyp!='A_world' and w not in OFFSETS else '')
            stats.append(st)
    dump(out/'hypotheses.json',stats);pd.DataFrame(stats).to_csv(out/'hypotheses.csv',index=False)
    world={s['word']:s for s in stats if s['hypothesis']=='A_world' and s['n']}
    groups=[]
    for key in ['map','target_category','ref_category']:
        for (w,v),sub in valid.groupby(['semantic_word',key]):
            st=summarize(sub,'angle_world',world[w]['axis_deg']);st.update(word=w,group_type=key,group=v)
            local=summarize(sub,'angle_world');st['local_best_axis_deg']=local['axis_deg'];st['local_best_consistency']=local['consistency'];groups.append(st)
    dump(out/'groups.json',groups);pd.DataFrame(groups).to_csv(out/'groups.csv',index=False)
    # Train-only cross-object held-out estimate for A (no network training).
    hold=[]
    for w,sub in valid.groupby('semantic_word'):
        split=sub.cluster.map(lambda s:int(hashlib.sha256(s.encode()).hexdigest()[:8],16)%5)
        for fold in range(5):
            fit=sub[split!=fold];ev=sub[split==fold]
            if len(fit) and len(ev):
                a=summarize(fit,'angle_world')['axis_deg'];z=summarize(ev,'angle_world',a);z.update(word=w,fold=fold,fit_n=len(fit));hold.append(z)
    dump(out/'world_crossfit.json',hold)
    # Polygon projections in each hypothesis frame; shape and centroid tests separated.
    boundary=[]
    for r in geo:
        if r['status']!='valid':continue
        w=r['semantic_word']
        for hyp,base,off in [('A_world',0,world[w]['axis_deg']),('B_initial_yaw',r['initial_yaw_deg'],OFFSETS.get(w)),('C_start_landmark_LOS',r['los_reference_deg'],OFFSETS.get(w))]:
            if base is None or off is None:continue
            direction=(base+off)%360;u=np.array([np.cos(np.deg2rad(direction)),np.sin(np.deg2rad(direction))]);p=np.array(r['target_polygon']);q=np.array(r['ref_polygon']);gap=float(min(p@u)-max(q@u)); reversegap=float(max(p@u)-min(q@u))
            for tol in [.5,1.,2.]:
                boundary.append({'occurrence_id':r['occurrence_id'],'word':w,'hypothesis':hyp,'tolerance_m':tol,'projection_gap_m':gap,'opposite_projection_gap_m':reversegap,'boundary_class':'consistent' if gap>tol else 'opposite' if reversegap< -tol else 'ambiguous','centroid_sector_consistent':bool(distance(r['angle_world'],direction)<=45),'target_area':r['target_area'],'ref_area':r['ref_area']})
    bdf=pd.DataFrame(boundary);bdf.to_csv(out/'polygon_checks.csv',index=False)
    bs=bdf.groupby(['word','hypothesis','tolerance_m','boundary_class']).size().reset_index(name='n');bs.to_csv(out/'polygon_summary.csv',index=False)
    # Actual reverse statements on same pair; include within/in between target objects.
    direct=valid[~valid.inverse].copy();reverse=[];opposites={'left':'right','right':'left','front':'behind','behind':'front','above':'below','below':'above','north':'south','south':'north','east':'west','west':'east'}
    bypair=collections.defaultdict(list)
    for r in direct.to_dict('records'):bypair[(r['map'],r['target_id'],r['ref_id'])].append(r)
    for r in direct.to_dict('records'):
        for q in bypair.get((r['map'],r['ref_id'],r['target_id']),[]):
            if r['occurrence_id']>=q['occurrence_id']:continue
            if r['semantic_word'] not in opposites:continue
            reverse.append({'first':r['occurrence_id'],'second':q['occurrence_id'],'word1':r['semantic_word'],'word2':q['semantic_word'],'opposite_lexical_pair':opposites[r['semantic_word']]==q['semantic_word'],'angle_separation_deg':float(distance(r['angle_world'],q['angle_world']))})
    dump(out/'reverse_statements.json',reverse)
    # Circular antipodality of lexical distributions is separate from paired inverse text.
    anti=[]
    for a,b in [('left','right'),('front','behind'),('above','below'),('north','south'),('east','west')]:
        if a in world and b in world:anti.append({'word1':a,'word2':b,'n1':world[a]['n'],'n2':world[b]['n'],'mean_angle_separation':float(distance(world[a]['mean_angle_deg'],world[b]['mean_angle_deg'])),'deviation_from_180':float(180-distance(world[a]['mean_angle_deg'],world[b]['mean_angle_deg'])),'R1':world[a]['resultant_R'],'R2':world[b]['resultant_R']})
    dump(out/'antipodality.json',anti)
    # Coordinate calibration is independent of direction text and target displacements.
    import rasterio
    raster=[]
    for m in maps:
        with rasterio.open(root/'rgbd'/f'{m}.tif') as src:
            tr=src.transform
            raster.append({'map':m,'crs':str(src.crs),'transform':list(tr)[:6],'bounds':list(src.bounds),'width':src.width,'height':src.height,'image_right_is_plus_X':tr.a>0 and tr.b==0 and tr.d==0,'image_up_is_plus_Y':tr.e<0 and tr.b==0 and tr.d==0,'true_geographic_north':'unverified_missing_CRS_and_control_points'})
    dump(out/'coordinate_audit.json',raster)
    corpus={'episodes':len(episodes),'maps':len(maps),'direction_occurrences':len(df),'episodes_with_direction':df.episode.nunique(),'unique_direction_texts':df.instruction.nunique(),'valid_direct_pairs':len(valid),'valid_episodes':valid.episode.nunique(),'valid_clusters':valid.cluster.nunique(),'unbound_occurrences':int((df.status=='excluded').sum()),'ambiguous_occurrences':int((df.status=='ambiguous').sum()),'description_mismatches':len(mismatch),'unknown_frame_fraction_occurrences':None,'missing_annotator_camera_pose_fraction':1.0,'unknown_frame_definition':'Unidentifiable population rate: lexical occurrences include nonrelations; missing annotation camera pose is not equivalent to unknown frame type.'}
    dump(out/'corpus.json',corpus);dump(out/'annotation_mismatches.json',mismatch)
    manifest={'input_files':[{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in files], 'split':'original_citynav TRAIN ONLY','excluded_splits_opened':False,'seed':SEED,'boundary_tolerance_m':1,'sector_halfwidth_deg':45,'heading_source':'first trajectory record only','D':'unavailable: schema has no heading; polygon axis not directed heading','software':{'numpy':np.__version__,'pandas':pd.__version__},'scope_policy':'Conservative exact full-name singleton + target-headed/main or explicit fronted/inverse clause. No GT name resolution.'}
    dump(out/'manifest.json',manifest)
    print(json.dumps(corpus,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
