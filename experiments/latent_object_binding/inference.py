"""Target marginal over latent reference bindings with explicit Unknown mass."""
import itertools
import numpy as np
import torch
from experiments.language_object_graph.reasoning import probability
from .language import parse,active_variables
from .factors import Geometry,unary_features

DEFAULT_WEIGHTS=np.array([1.,5.,12.,.5,.5,.5,1.,1.,1.],np.float32)


def support(e,g,scores,limit):
    if e['named_ids']:
        pool=np.array(e['named_ids'],int);prob=probability(scores[pool]);return pool,prob,1.
    typ=e['object_type'];pool=np.flatnonzero(np.array(g.types)==typ) if typ else np.arange(len(g.ids))
    if not len(pool):return np.empty(0,int),np.empty(0),0.
    p=probability(scores[pool]);order=np.argsort(-p,kind='stable')[:limit];mass=float(p[order].sum());return pool[order],p[order],mass


def prepare(graph,instruction,public,*,visual=None,language=None,reference_limit=8,beam=16,geometry=None):
    if set(public)!={'static_logits','grid_xy','bounds','pose_world'}:raise ValueError('private/unsupported inference inputs')
    language=language or parse(instruction,graph);variables=language.get('variables',language.get('entities',[]));bounds=np.asarray(public['bounds']);world=bounds[:2]+np.asarray(public['grid_xy'])*(bounds[2:]-bounds[:2]);geometry=geometry or Geometry(graph,world);base=probability(public['static_logits']);prior=np.maximum(geometry.kernels@base,1e-30)
    if not variables: # Output the same A3 belief for unresolved text.
        variables=[dict(object_type=None,named_ids=[],size=None,shape=None,colors=[],role='target')]
    visual=visual or {};features=[];unaries=[]
    for i,e in enumerate(variables):
        x=unary_features(e,graph,prior if i==0 else np.ones(len(graph.ids)))
        if i in visual:
            v=np.asarray(visual[i],np.float32)
            if v.shape!=(len(graph.ids),2):raise ValueError('same public candidates for visual evidence')
            x[:,-2:]=v
        features.append(x);unaries.append(x@DEFAULT_WEIGHTS[:8])
    active,factors=active_variables(language)
    references=sorted(active-{0})[:7];pools={};probs={};unknown=[]
    for i in references:
        ids,p,mass=support(variables[i],graph,unaries[i],reference_limit);pools[i]=ids;probs[i]=dict(zip(ids.tolist(),p.tolist()))
        if not len(ids):unknown.append(f'variable {i} has no public support')
    configurations=[({},1.)]
    symmetric_pair=len(references)==2 and len(factors)==1 and factors[0]['kind']=='between' and factors[0]['subject']==0 and set(factors[0]['anchors'])==set(references)
    for i in references:
        options=[]
        for assignment,w in configurations:
            for j in pools[i]:options.append(({**assignment,i:int(j)},w*probs[i][int(j)]))
        # Beam truncation keeps absolute mass; omitted hypotheses are Unknown.
        if symmetric_pair and i==references[-1]:
            groups={}
            for assignment,w in options:
                if len(set(assignment.values()))<2:continue
                key=tuple(sorted(assignment.values()));groups.setdefault(key,[]).append((assignment,w))
            keys=sorted(groups,key=lambda k:(-sum(w for _,w in groups[k]),k))[:max(1,beam//2)]
            configurations=[item for key in keys for item in groups[key]]
        else:configurations=sorted(options,key=lambda x:(-x[1],tuple(sorted(x[0].items()))))[:beam]
    if not references:configurations=[({},1.)]
    n=len(graph.ids);potentials=[];ref_features=[];joint=[];weights=[];assignments=[];used=[]
    proposal_mass=0.
    for assignment,w in configurations:
        # Distinct nominal variables are not collapsed merely to improve scores.
        # Explicit co-reference is represented by reusing a variable index.
        illegal=False
        for i,j in itertools.combinations(assignment,2):
            if assignment[i]==assignment[j] and not (variables[i]['named_ids'] and set(variables[i]['named_ids'])==set(variables[j]['named_ids'])):illegal=True
        if illegal:continue
        proposal_mass+=w
        field=np.ones(len(geometry.points),np.float32);valid=True;factor_used=[]
        for f in factors:
            s=f['subject'];anchors=f['anchors'];kind=f['kind']
            if f.get('frame_unknown') or kind in ('left','right','front','behind'):continue
            if 0 in anchors:
                if len(anchors)==1 and s in assignment:
                    inverse={'east':'west','west':'east','north':'south','south':'north','near':'near','along':'along'}.get(kind)
                    if inverse:
                        field*=np.maximum(geometry.field(inverse,[assignment[s]]),1e-6);factor_used.append(f)
                    else:unknown.append('target-as-Between-anchor factor not certified')
                continue
            if any(a not in assignment for a in anchors):continue
            ids=[assignment[a] for a in anchors]
            if s==0:
                field*=np.maximum(geometry.field(kind,ids),1e-6)
                for a in ids:field[a]=1e-6 # target is not its independent reference
                factor_used.append(f)
            elif s in assignment:
                value=geometry.object_relation(kind,assignment[s],ids);w*=max(value,1e-6);factor_used.append(f)
        # Store unclamped target likelihood; scoring is differentiable in weights.
        potentials.append(np.log(np.maximum(field,1e-6)));weights.append(w);assignments.append(assignment);used.extend(factor_used)
        ref_features.append(np.sum([features[i][obj] for i,obj in assignment.items()],axis=0) if assignment else np.zeros(8,np.float32))
    if not potentials:
        potentials=[np.zeros(len(geometry.points),np.float32)];weights=[0.];ref_features=[np.zeros(8,np.float32)];assignments=[{}];unknown.append('all reference support Unknown')
    mass=float(min(1.,proposal_mass));confidence=float(language.get('confidence',0)) if language.get('entities') else 0.
    role_mass=language.get('hypotheses',[{'probability':1.}])[0]['probability'] if language.get('hypotheses') else 1.
    scope_mass=float(np.prod([f.get('subject_hypotheses',[{'probability':1.}])[0]['probability'] for f in factors if not f.get('frame_unknown')]))
    confidence*=role_mass*scope_mass
    if role_mass<1 or scope_mass<1:unknown.append('omitted role/scope alternatives remain Unknown')
    if len(active)>8:unknown.append('connected variables beyond budget retained as Unknown');confidence*=8/len(active)
    known=mass*confidence;logweights=np.log(np.maximum(weights,1e-30))
    return dict(target_features=features[0],reference_features=np.array(ref_features,np.float32),log_phi=np.array(potentials,np.float32),log_weights=logweights.astype(np.float32),support_mass=mass,confidence=confidence,known_mass=known,assignments=assignments,active_variables=sorted(active),language=language,unknown=unknown,kernels=geometry.kernels,base=base,world=world,reference_posteriors={str(i):dict(indices=pools[i].tolist(),probabilities=[probs[i][int(j)] for j in pools[i]],omitted_mass=1-sum(probs[i].values())) for i in references},factors=used)


def posterior(prepared,weights=None,*,hard=False,relations=True,blend=.25):
    """Torch sum-product permits supervision through latent reference hypotheses."""
    w=torch.as_tensor(DEFAULT_WEIGHTS if weights is None else weights,dtype=torch.float32)
    device=w.device;x=torch.as_tensor(prepared['target_features'],device=device,dtype=torch.float32);ref=torch.as_tensor(prepared['reference_features'],device=device,dtype=torch.float32);phi=torch.as_tensor(prepared['log_phi'],device=device,dtype=torch.float32);lw=torch.as_tensor(prepared['log_weights'],device=device,dtype=torch.float32)
    # Proposal base probabilities already encode fixed reference unaries. Learned
    # deltas reweight assignments, rather than assuming reference-instance labels.
    delta=ref@(w[:8]-torch.tensor(DEFAULT_WEIGHTS[:8],device=device));rw=lw+delta
    if hard:
        idx=rw.argmax();rw=rw[idx:idx+1];phi=phi[idx:idx+1]
    target=x@w[:8];n=len(target)
    joint=rw[:,None]+target[None,:]
    if relations:joint=joint+torch.clamp(w[8],0,4)*phi[:,:n]
    logits=torch.logsumexp(joint,dim=0);belief=torch.softmax(logits,0)
    # Continuous output also marginalizes reference hypotheses; keep A3 prior.
    base=torch.as_tensor(prepared['base'],dtype=torch.float32,device=device);kernels=torch.as_tensor(prepared['kernels'],device=device)
    spatial=torch.logsumexp(rw[:,None]+(torch.clamp(w[8],0,4)*phi[:,n:] if relations else torch.zeros_like(phi[:,n:])),dim=0)
    continuous=torch.softmax(torch.log(torch.clamp(base,min=1e-30))+spatial,0)
    object_heat=belief@kernels;known=prepared['known_mass'] if relations else prepared['confidence'];gate=float(blend)*known
    heat=(1-gate)*base+gate*(.5*object_heat+.5*continuous)
    # Reporting the true discrete joint reference posterior requires summing
    # over target hypotheses, not merely normalizing reference unary weights.
    assignment_posterior=torch.softmax(torch.logsumexp(joint,dim=1),0)
    assignments=prepared['assignments']
    if hard:assignments=[assignments[int(idx)]]
    marginals={}
    for j,assignment in enumerate(assignments):
        for variable,obj in assignment.items():
            key=str(variable);marginals.setdefault(key,{})
            marginals[key][str(obj)]=marginals[key].get(str(obj),0)+assignment_posterior[j]
    return dict(object_belief=belief,heatmap=heat,logits=torch.log(torch.clamp(heat,min=1e-30)),known_mass=known,gate=gate,assignment_posterior=assignment_posterior,reference_prior_assignment_weights=torch.softmax(rw,0),reference_binding_marginals=marginals)
