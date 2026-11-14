import unittest
import numpy as np
from controlswitch.control_correction import control_features,blend_control

class ControlCorrectionTests(unittest.TestCase):
    def test_blend_uses_conditional_live_probabilities(self):
        r=np.array([[.6,.2,.2],[.1,.4,.5]])
        c=np.array([[.2,.8,0],[.8,.2,0]])
        p=blend_control(r,c,.5)
        np.testing.assert_allclose(p,[[.475,.525,0],[.5,.5,0]])
        np.testing.assert_allclose(p.sum(1),1)
    def test_blend_is_team_equivariant(self):
        a=np.array([[.7,.2,.1]]);b=np.array([[.1,.9,0]])
        np.testing.assert_allclose(blend_control(a[:,[1,0,2]],b[:,[1,0,2]],.25),blend_control(a,b,.25)[:,[1,0,2]])
        with self.assertRaises(ValueError):blend_control(a,b,np.nan)
    def test_features_do_not_require_or_use_possession_labels(self):
        n=100;rng=np.random.default_rng(3)
        s={'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
        x=control_features(s,np.array([105,68]),np.ones(n,bool))
        np.testing.assert_array_equal(x,control_features({**s,'y':np.full(n,2)},np.array([105,68]),np.ones(n,bool)))
        self.assertEqual(x.shape,(n,244));self.assertTrue(np.isfinite(x).all())
if __name__=='__main__':unittest.main()
