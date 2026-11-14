import itertools,unittest
import numpy as np
from controlswitch.nonlinear_boundary import floor_decode
from controlswitch.boundary_had import adaptive_decode

class EventFloorTests(unittest.TestCase):
    def test_default_reproduces031_decoder(self):
        rng=np.random.default_rng(33);p=rng.random((100,3));p/=p.sum(1,keepdims=True);q=rng.random(100)
        np.testing.assert_array_equal(floor_decode(p,q),adaptive_decode(p,q,1.))
    def test_soft_floor_matches_exhaustive_path_search(self):
        p=np.array([[.8,.15,.05],[.3,.65,.05],[.75,.2,.05],[.1,.1,.8]]);q=np.array([.1,.99,.99,.2]);floor=.05
        def score(path):
            value=sum(np.log(p[i,k]) for i,k in enumerate(path))
            for i in range(1,len(path)):
                a,b=path[i-1],path[i]
                if a!=b:value-=3 if 2 in (a,b) else 12*np.clip(2-2*q[i],floor,2)
            return value
        best=max(itertools.product(range(3),repeat=len(p)),key=score)
        np.testing.assert_array_equal(floor_decode(p,q,floor),best)
    def test_invalid_controls_rejected(self):
        p=np.full((2,3),1/3);q=np.full(2,.5)
        for f in (0,2,np.nan):
            with self.assertRaises(ValueError):floor_decode(p,q,f)
if __name__=='__main__':unittest.main()
