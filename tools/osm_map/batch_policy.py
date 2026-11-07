"""Public structural choice separated from independent validation."""
import numpy as np

def select_public_transform(initial,local,bounds,cost_initial,cost_final,edge_samples,building_proxies):
 initial=np.asarray(initial);local=np.asarray(local);x0,y0,x1,y1=bounds;c=np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]]);shift=float(np.linalg.norm(c@local[:,:2].T+local[:,2]-(c@initial[:,:2].T+initial[:,2]),axis=1).max());improvement=(cost_initial-cost_final)/max(cost_initial,1e-6);use=improvement>=.1 and shift<=15
 risk=[]
 if edge_samples<100:risk.append('insufficient_edge_samples')
 if building_proxies<3:risk.append('insufficient_building_proxies')
 if cost_final>10:risk.append('high_structural_cost')
 if shift>30:risk.append('unstable_local_correction')
 return dict(matrix=(local if use else initial).tolist(),selected='scene_similarity' if use else 'city_uniform',local_max_corner_shift_m=shift,relative_improvement=improvement,risk_flags=risk,structure_cost_selected=cost_final if use else cost_initial)

def grade_independent(validation,risk_flags,fit_n):
 if validation and validation['n']>=5 and fit_n>=10 and validation['max_m']<=5:return 'verified'
 return 'unreliable' if risk_flags or (validation and validation['n']>=5 and validation['max_m']>5) else 'pending_validation'
