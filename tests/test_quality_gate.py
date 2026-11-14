import unittest
import numpy as np
from controlswitch.quality_gate import quality_features,gated_dead

class QualityGateTests(unittest.TestCase):
    def sequence(self):
        n=60;rng=np.random.default_rng(8)
        return {'xy':rng.normal(size=(n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)}
    def test_quality_is_team_invariant_and_label_independent(self):
        s=self.sequence();a=quality_features(s);b=quality_features({**s,'teams':1-s['teams'],'y':np.full(60,2)})
        self.assertEqual(a.shape,(60,27));np.testing.assert_allclose(a,b)
    def test_missing_ball_does_not_generate_nonfinite_features(self):
        x=quality_features(self.sequence(),np.ones(60,bool));self.assertTrue(np.isfinite(x).all());self.assertFalse(x[:,0].any())
    def test_gate_endpoints_and_probability_bounds(self):
        old=np.array([.1,.9]);new=np.array([.8,.2]);q=np.ones(2)
        np.testing.assert_allclose(gated_dead(old,new,q,1.),new)
        neutral=gated_dead(old,new,np.zeros(2),1.);expected=1/(1+np.exp(-.5*(np.log(old/(1-old))+np.log(new/(1-new)))))
        np.testing.assert_allclose(neutral,expected)
        with self.assertRaises(ValueError):gated_dead(old,new,np.array([0,np.nan]),1.)
if __name__=='__main__':unittest.main()
