import itertools,unittest
import numpy as np
from controlswitch.boundary_had import adaptive_decode,boundary_observations,targets
from controlswitch.model import decode

class AdaptiveBoundaryTests(unittest.TestCase):
    def test_zero_strength_exactly_matches_frozen_decoder(self):
        rng=np.random.default_rng(20);p=rng.random((80,3));p/=p.sum(1,keepdims=True)
        np.testing.assert_array_equal(adaptive_decode(p,rng.random(80),0),decode(p,12,stop_penalty=3))
    def test_matches_exhaustive_path_search_and_preserves_stop_cost(self):
        p=np.array([[.8,.15,.05],[.2,.7,.1],[.1,.25,.65],[.15,.75,.1]]);q=np.array([.1,.9,.1,.8]);strength=.5
        for mode in ('balanced','recovery'):
            factor=np.clip(1+strength*(1-2*q),.25,2) if mode=='balanced' else np.maximum(1-strength*q,.25)
            def objective(path):
                value=sum(np.log(p[i,k]) for i,k in enumerate(path))
                for i in range(1,len(path)):
                    a,b=path[i-1],path[i]
                    if a!=b:value-=3 if 2 in (a,b) else 12*factor[i]
                return value
            best=max(itertools.product(range(3),repeat=4),key=objective)
            np.testing.assert_array_equal(adaptive_decode(p,q,strength,mode),best)
    def test_scores_and_geometry_are_team_invariant(self):
        rng=np.random.default_rng(2);n=100
        s={'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
        p=rng.random((n,3));p/=p.sum(1,keepdims=True)
        a=boundary_observations(s,np.array([105,68]),p);b=boundary_observations({**s,'teams':1-s['teams'],'y':np.full(n,2)},np.array([105,68]),p[:,[1,0,2]])
        np.testing.assert_allclose(a,b)
    def test_targets_exclude_stoppages(self):
        y=np.array([0]*50+[2]*50+[1]*50);t,valid=targets(y)
        self.assertFalse(t.any());self.assertFalse(valid[45:106].any())
        direct=np.array([0]*50+[1]*50);t,valid=targets(direct);self.assertTrue(t[49:52].all());self.assertTrue(valid.all())
    def test_invalid_emission_and_boundary_rejected(self):
        for p,q in [(np.empty((0,3)),np.array([])),(np.ones((2,3)),np.array([0.,np.nan])),(np.zeros((2,3)),np.ones(2))]:
            with self.assertRaises(ValueError):adaptive_decode(p,q)
if __name__=='__main__':unittest.main()
