import numpy as np
from scripts.diagnose_aerialvg_kernel_feasibility import necessary_constraints,solve_goal


def test_known_correcting_posterior_remains_in_necessary_relaxation():
    base=np.array([0.,1.,-2.]);observed=np.array([True,True,False]);legal=np.array([True,False,False]);prior=np.exp(base)*observed;prior/=prior.sum();support=np.array([[1.,0.],[0.,1.],[0.,0.]])
    constraints,reason=necessary_constraints(base,prior,support,observed,legal,0,.7)
    assert reason is None
    A,b=constraints;assert np.max(A@np.array([1.,0.])-b)<=1e-8
    solutions,status=solve_goal(base,prior,support,observed,legal,0,.7)
    assert solutions and status['status']=='feasible_relaxation'


def test_identical_kernels_cannot_express_a_needed_density_ratio():
    base=np.array([0.,1.]);prior=np.exp(base);prior/=prior.sum();support=np.full((2,2),.5);observed=np.ones(2,bool);legal=np.array([True,False])
    solutions,status=solve_goal(base,prior,support,observed,legal,0,.7)
    assert not solutions and status['status']=='numerically_infeasible'


def test_necessary_conditions_retain_known_random_actual_update_witnesses():
    rng=np.random.default_rng(4);checked=0
    for i in range(150):
        base=rng.normal(size=6);observed=rng.random(6)>.3;observed[0]=True
        kernel=rng.random((6,3))*observed[:,None];kernel[:,0]=0
        support=kernel/np.maximum(kernel.sum(0),1e-9)
        prior=np.exp(base-base.max())*observed;prior/=prior.sum()
        weights=np.array([1.,0.,0.]) if i%5==0 else rng.dirichlet(np.ones(3))
        density=support@weights;updated=base+.7*observed*np.clip(np.log(np.maximum(density,1e-9))-np.log(np.maximum(prior,1e-9)),-2,2)
        u=int(updated.argmax())
        if base.argmax()==u:continue
        legal=np.arange(6)==u;constraints,reason=necessary_constraints(base,prior,support,observed,legal,u,.7)
        assert reason is None
        A,b=constraints;assert np.max(A@weights-b)<=1e-7
        checked+=1
    assert checked>=10


def test_actual_oracle_witness_artifact_is_not_mistaken_for_real_model_gain():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]/'reports/aerialvg_diagnostics_20261101'
    x=json.loads((root/'kernel_feasibility.json').read_text())
    assert x['checkpoint_unchanged'] and x['summary']['dev']['states']==9
    for part in x['summary'].values():
        assert sum(part['counts'].values())==part['states']
        for row in part['rows']:
            if row['outcome']=='verified_mixture_witness':
                assert row['witness']['updated_NE']<=20
                assert abs(sum(row['witness']['oracle_weights'])-1)<1e-7
    assert 'not a formal exact-arithmetic certificate' in x['protocol']
