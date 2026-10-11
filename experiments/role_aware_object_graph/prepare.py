"""Public text export and sealed offline weak annotations. No GT-coordinate selection."""
import json
from pathlib import Path
from experiments.language_object_graph.run import load
from experiments.semantic_polygon_grounding.audit import DATA,save
OUT=Path('runs/role_aware_object_graph')
def main():
    processed=json.loads((DATA/'cityrefer/processed_descriptions.json').read_text());counts={}
    for split in ('train','dev'):
        states,labels,digest=load(split);public=[];weak=[]
        for s in states:
            m=s['metadata'];sid=m['sample_id'];label=labels[sid];parts=m['family_id'].split(':');index=int(parts[3]);annotations=processed.get(m['map_name'],{}).get(str(label['target_object_id']),[]);a=annotations[index] if index<len(annotations) else {}
            public.append(dict(sample_id=sid,family_id=m['family_id'],map_name=m['map_name'],instruction=m['instruction']))
            weak.append(dict(sample_id=sid,target=a.get('target'),reference_names=a.get('landmarks',[]),status='automatic processed description, not manual truth; offline only'))
        save(OUT/f'{split}_public_texts.json',public);save(OUT/f'{split}_weak_roles.json',weak);counts[split]=dict(families=len(states),plan_sha=digest)
    save(OUT/'preparation.json',counts)
if __name__=='__main__':main()
