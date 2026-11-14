import unittest
import numpy as np
from controlswitch.tracker_robustness import noisy_tracking,interval_mask,condition

class TrackerRobustnessTests(unittest.TestCase):
    def sequence(self):
        n=250
        return {'xy':np.zeros((n,4,2)), 'speed':np.ones((n,4)), 'teams':np.array([0,0,1,1]), 'ball':np.zeros((n,2)), 'ball_speed':np.ones(n)}
    def test_degradation_preserves_input_and_excludes_reference_labels(self):
        s=self.sequence();original=s['xy'].copy();s['y']=np.full(250,2)
        degraded=noisy_tracking(s,1)
        self.assertNotIn('y',degraded);np.testing.assert_array_equal(s['xy'],original)
        self.assertEqual(degraded['xy'].shape,(250,6,2));np.testing.assert_array_equal(degraded['teams'],[0,0,1,1,0,1])
        self.assertLessEqual(np.isfinite(degraded['ball']).all(1).mean(),.6)
        self.assertTrue((~np.isfinite(degraded['xy'][:,:4]).all(2)).any())
    def test_repeatable_and_label_independent(self):
        a=noisy_tracking(self.sequence(),22);s=self.sequence();s['y']=np.zeros(250,int);b=noisy_tracking(s,22)
        for k in a:np.testing.assert_array_equal(a[k],b[k])
        train,_=condition(s,'tracker_noise',0,True);validation,_=condition(s,'tracker_noise',0,False)
        self.assertFalse(np.allclose(train['xy'],validation['xy'],equal_nan=True))
    def test_noise_never_turns_missing_source_positions_into_observations(self):
        s=self.sequence();s['ball'][:10]=np.nan;s['xy'][:10,0]=np.nan
        a=noisy_tracking(s,2)
        self.assertTrue(np.isnan(a['ball'][:10]).all());self.assertTrue(np.isnan(a['xy'][:10,0]).all())
    def test_invalid_mask_cannot_loop_forever(self):
        for spec in [(0,.4,1,2),(10,1.1,1,2),(10,.4,0,2),(10,.4,3,2)]:
            with self.assertRaises(ValueError):interval_mask(spec[0],np.random.default_rng(1),*spec[1:])
if __name__=='__main__':unittest.main()
