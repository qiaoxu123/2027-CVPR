"""Public language-only evaluation strata, never GT-corrected relationship labels."""


def anchor_strata(parsed,reference):
 paths=[p for p in parsed['paths'] if str(reference['id']) in p['public_ids']]
 trusted=[p for p in paths if p.get('semantic_confident',False) and all(s.get('original_relation') not in {'associated region','possessive region association'} for s in p['steps'])]
 n=min((p['hops'] for p in trusted),default=0)
 return dict(group='direct' if n==1 else 'indirect' if n>1 else 'unknown',hops=n,has_direct=any(p['hops']==1 for p in trusted),has_indirect=any(p['hops']>1 for p in trusted),raw_paths=len(paths),unresolved_relative_frame=any(s['reference_frame']=='unresolved_relative' for p in trusted for s in p['steps']))
