"""Reproduce TRAIN-only token counts and posthoc exact-input label conflicts."""
import argparse,json,collections
from pathlib import Path
import numpy as np
from transformers import AutoTokenizer
from graphnav.checkpointing import atomic_json

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--prepared-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);tokenizer=AutoTokenizer.from_pretrained('bert-base-uncased',local_files_only=True)
    tokens={};ambiguity={}
    for mode in ('full_text','landmark_relation'):
        rows=[json.loads(s) for s in (a.prepared_dir/f'{mode}.jsonl').read_text().splitlines()];assert len(rows)==21878
        length=np.array([len(x) for x in tokenizer([q['instruction'] for q in rows],truncation=False,add_special_tokens=True)['input_ids']])
        tokens[mode]=dict(all_queries=len(rows),mean=float(length.mean()),median=float(np.median(length)),p95=float(np.quantile(length,.95)),max=int(length.max()),over_96=int((length>96).sum()),birmingham_mean=float(length[[q['map_name'].startswith('birmingham_') for q in rows]].mean()),uses_full_token_sequence=True)
        groups=collections.defaultdict(list)
        for i,x in enumerate(rows):
            if not x['map_name'].startswith('birmingham_'):continue
            # No label/target/episode fields in the signature.
            key=json.dumps([x['map_name'],x['instruction'].casefold(),x['world_bounds'],x['landmarks']],sort_keys=True);groups[key].append(i)
        conflicts=[]
        for ids in groups.values():
            # GT enters only this posthoc diagnostic, never model inference.
            xy=np.array([np.array(rows[i]['target_xy'])*rows[i]['physical_map_sizes'] for i in ids]);diam=float(np.linalg.norm(xy[:,None]-xy[None],axis=-1).max())
            if diam>40:conflicts.append(dict(indices=ids,diameter_m=diam,model_instruction=rows[ids[0]]['instruction'],map_name=rows[ids[0]]['map_name']))
        ambiguity[mode]=dict(queries=7619,distinct_input_signatures=len(groups),duplicate_input_groups=sum(len(v)>1 for v in groups.values()),GT_diameter_over40m_groups=len(conflicts),queries_in_incompatible_target_groups=sum(len(c['indices']) for c in conflicts),examples=conflicts[:8],GT_only_posthoc=True,counts_are_conservative='Exact public reference metadata and casefolded BERT-uncased text; no target fields in signature. This is an information-conflict diagnosis, not an oracle prediction or accuracy bound.')
    atomic_json(tokens,a.output/'token_counts.json');atomic_json(ambiguity,a.output/'input_ambiguity.json')
    audit=[json.loads(s) for s in (a.prepared_dir/'extraction_audit.jsonl').read_text().splitlines()];phrases=collections.Counter()
    for row in audit:
        for name,word in {(x['name'].casefold(),x['geometry_words']) for x in row['kept_pairs']}:
            if word:phrases[word]+=1
    atomic_json(dict(phrase_units=dict(phrases.most_common()),counts_are_query_name_phrase_units_not_polygon_parts=True),a.output/'lexical_phrases.json')

if __name__=='__main__':main()
