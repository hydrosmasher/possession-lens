import unittest
import numpy as np
from controlswitch.deadball import fuse,observation_features

class DeadBallTests(unittest.TestCase):
    def test_factorisation_keeps_conditional_owner(self):
        p=np.array([[.6,.2,.2],[.1,.3,.6],[0.,0.,1.]])
        out=fuse(p,np.array([.2,.9,.5]))
        np.testing.assert_allclose(out.sum(1),1)
        np.testing.assert_allclose(out[:2,0]/out[:2,1],[3.,1/3])
        np.testing.assert_allclose(out[:,2],[.2,.9,.5])
        np.testing.assert_allclose(out[2],[.25,.25,.5])
    def test_rejects_invalid_probabilities(self):
        for q in [np.array([np.nan]),np.array([1.1]),np.array([-.1])]:
            with self.assertRaises(ValueError):fuse(np.ones((1,3)),q)
    def test_team_invariant_and_labels_never_used(self):
        rng=np.random.default_rng(3);n=40
        s={'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
        a=observation_features(s,np.array([105,68]))
        swapped={**s,'teams':1-s['teams'],'y':np.full(n,2)}
        np.testing.assert_allclose(a,observation_features(swapped,np.array([105,68])))
        mask=np.ones(n,bool)
        missing=observation_features(s,np.array([105,68]),mask)
        self.assertTrue(np.isfinite(missing).all())
        self.assertEqual(missing.shape,(n,240))
if __name__=='__main__':unittest.main()
