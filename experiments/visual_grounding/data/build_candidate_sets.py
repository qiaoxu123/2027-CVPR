"""Same episode causal candidate sets; unknown attributes are never semantic negatives."""
import argparse,json,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from PIL import Image,ImageDraw

def partition(row):
    if row['split']!='train_seen':return row['split']
    xy=np.asarray(row['object_center_world']) if 'object_center_world' in row else np.asarray(row['object_geometry']).mean(0);key=f"{row['map_name']}:{int(xy[0]//200)}:{int(xy[1]//200)}"
    return 'internal_dev' if int(hashlib.sha256(key.encode()).hexdigest()[:8],16)%10==0 else 'train'

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default='runs/visual_grounding_benchmark/data_audit');a=p.parse_args();root=Path(a.data)
    regions=[json.loads(l) for l in (root/'regions.jsonl').open()];queries=[json.loads(l) for l in (root/'manifest.jsonl').open()];byepisode=defaultdict(list)
    for r in regions:
        r['internal_partition']=partition(r);byepisode[r['episode_id']].append(r)
    counts=Counter();sizes=[]
    with (root/'candidates.jsonl').open('w') as handle:
      for q in queries:
        if not q['roi_quality']['attribute_resolution_eligible']:counts['query_below24px']+=1;continue
        if not q.get('roi_path'):counts['query_unmaterialized']+=1;continue
        latest={}
        for r in byepisode[q['episode_id']]:
            if not r.get('roi_path') or r['observed_step']>q['observation_cutoff_step'] or r['object_type']!=q['object_type'] or not r['roi_quality']['attribute_resolution_eligible'] or partition(r)!=partition(q):continue
            if r['object_id'] not in latest or latest[r['object_id']]['observed_step']<r['observed_step']:latest[r['object_id']]=r
        for kind,label in q.get('query_attributes',{}).items():
            tagged=[r for r in latest.values() if kind in r['attributes']];positives=[r for r in tagged if r['attributes'][kind]['label']==label];negatives=[r for r in tagged if r['attributes'][kind]['label']!=label]
            if not any(r['object_id']==q['object_id'] for r in positives):counts['target_attribute_unconfirmed_or_conflicting']+=1;continue
            if not negatives or len(tagged)<2:counts['no_supported_different_attribute_candidate']+=1;continue
            row=dict(query_sample_id=q['sample_id'],map_name=q['map_name'],split=q['split'],internal_partition=partition(q),episode_id=q['episode_id'],observation_cutoff_step=q['observation_cutoff_step'],instruction=q['instruction'],attribute_phrase=q['attribute_phrase'],attribute_type=kind,attribute_label=label,object_type=q['object_type'],candidate_sample_ids=[r['sample_id'] for r in tagged],positive_sample_ids=[r['sample_id'] for r in positives],negative_sample_ids=[r['sample_id'] for r in negatives],unknown_candidates_excluded=len(latest)-len(tagged),label_status='language_weak_not_visual_verified',candidate_object_ids=[r['object_id'] for r in tagged])
            handle.write(json.dumps(row)+'\n');counts[q['split']+'_weak_candidate_queries']+=1;sizes.append(len(tagged))
    # 150 fixed Train/Seen cases, targets prioritized, all shown with actual source provenance.
    eligible=[r for r in regions if r.get('roi_path') and r['split']!='val_unseen' and r['object_type'] in ('Building','Car') and r['attributes'] and r['roi_quality']['attribute_resolution_eligible']]
    rng=np.random.default_rng(20261010);targets=[r for r in queries if r.get('roi_path') and r['split']!='val_unseen' and r['object_type'] in ('Building','Car') and r.get('query_attributes') and r['roi_quality']['attribute_resolution_eligible']];selected=[targets[i] for i in rng.choice(len(targets),min(100,len(targets)),replace=False)];seen={r['sample_id'] for r in selected};other=[r for r in eligible if r['sample_id'] not in seen];selected += [other[i] for i in rng.choice(len(other),min(150-len(selected),len(other)),replace=False)]
    (root/'cases').mkdir(exist_ok=True)
    for si in range(0,len(selected),12):
        sheet=Image.new('RGB',(1200,1050),'white');draw=ImageDraw.Draw(sheet)
        for j,r in enumerate(selected[si:si+12]):
            x=j%4*300;y=j//4*350;im=Image.open(r['roi_path']).convert('RGB');im.thumbnail((294,245));sheet.paste(im,(x,y))
            phrase=r.get('attribute_phrase','') or str({k:v['label'] for k,v in r['attributes'].items()});text=f"{si+j} {r['split']} {r['map_name']}\nobj{r['object_id']} {r['object_type']} target={r['is_episode_target']}\nep{r['source_episode_index']} step{r['observed_step']} px{r['roi_pixel_size']}\n{phrase[:42]}\n{phrase[42:84]}"
            draw.text((x+2,y+248),text,fill='black')
        sheet.save(root/'cases'/f'contact_{si//12:02}.jpg',quality=92)
    (root/'cases.json').write_text(json.dumps(selected,indent=2));(root/'candidate_stats.json').write_text(json.dumps({'counts':dict(counts),'candidate_sizes':dict(Counter(sizes)),'verified_candidate_queries':0,'multi_positive_policy':'all same supported attribute labels positive; unknown excluded; object IDs linkage only','internal_partition':'map 200m spatial cell hash; all views of identity same subset'},indent=2));print(json.dumps(dict(counts)))
if __name__=='__main__':main()
