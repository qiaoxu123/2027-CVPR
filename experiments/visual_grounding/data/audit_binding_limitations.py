"""Post-freeze language ambiguity audit; never changes labels or reruns a model."""
import argparse,json,re
from pathlib import Path
from collections import Counter
COLORS='white|gray|grey|red|blue|brown|black|green|yellow|orange|beige'
COORD=re.compile(r'\b('+COLORS+r')\b[\s,\-]*(?:and|or|/)[\s,\-]*\b('+COLORS+r')\b',re.I)
MULTI=re.compile(r'\b(?:multi[ -]?(?:color(?:ed)?|colour(?:ed)?)|two[ -]?(?:toned|color(?:ed)?|colour(?:ed)?))\b',re.I)

def issues(phrase):
 result=[]
 if MULTI.search(phrase):result.append('explicit_multicolor_or_two_tone')
 for c1,c2 in COORD.findall(phrase):
  if c1.lower().replace('grey','gray')!=c2.lower().replace('grey','gray'):result.append('coordinated_distinct_colors');break
 return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-root',default='runs/visual_grounding_benchmark/data_audit');a=p.parse_args();root=Path(a.audit_root);out={'status':'language-only post-freeze limitation audit; main frozen labels/images/protocol unchanged','limitations':['adjacency parser can collapse gray and brown roof to brown because gray is not directly adjacent to roof','a consistent adjective across descriptions is not independent image verification','multiple target colors may genuinely coexist; they cannot safely define a unique color class'],'splits':{}}
 for split,suffix in [('train_seen','train'),('val_seen','seen'),('val_unseen','unseen')]:
  frozen=root/f'frozen_{suffix}'
  if not (frozen/'frozen.json').exists():continue
  index={r['sample_id']:r for r in map(json.loads,(frozen/'index.jsonl').open())};qs=[];ats=[];counts=Counter();unique=set()
  for q in map(json.loads,(frozen/'queries.jsonl').open()):
   reason=issues(q['attribute_phrase']) if q['attribute_type'].endswith('color') else []
   if reason:
    qs.append({'query_sample_id':q['query_sample_id'],'attribute_type':q['attribute_type'],'bound_label':q['attribute_label'],'attribute_phrase':q['attribute_phrase'],'reasons':reason,'map_name':q['map_name'],'episode_id':q['episode_id']});counts.update(reason);unique.add(q['query_sample_id'])
  for attr in map(json.loads,(frozen/'attributes.jsonl').open()):
   if not attr['attribute_type'].endswith('color'):continue
   r=index[attr['sample_id']];support=r['attributes'].get(attr['attribute_type'],{}).get('support_phrases',[]);bad=[{'phrase':phrase,'reasons':issues(phrase)} for phrase in support if issues(phrase)]
   if bad:ats.append(dict(attr,suspicious_support=bad))
  out['splits'][split]={'flagged_candidate_queries':len(qs),'flagged_candidate_query_images':len(unique),'reason_counts':dict(counts),'flagged_object_balanced_attribute_rows':len(ats),'flagged_attribute_instances':len({(x['map_name'],x['object_id'],x['attribute_type']) for x in ats}),'query_mask':qs,'attribute_mask':ats}
 (root/'binding_limitations.json').write_text(json.dumps(out,indent=2));print(json.dumps({s:{k:v for k,v in d.items() if k not in ('query_mask','attribute_mask')} for s,d in out['splits'].items()}))
if __name__=='__main__':main()
