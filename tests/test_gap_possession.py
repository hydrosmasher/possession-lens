import unittest
import numpy as np
from controlswitch.gap_possession import repair_ball,gap_features,long_mask,diagnostics

class GapPossessionTests(unittest.TestCase):
    def sequence(self,n=100):
        return {'xy':np.zeros((n,2,2)),'speed':np.ones((n,2)),'teams':np.array([0,1]),'ball':np.column_stack([np.arange(n)*.1,np.zeros(n)]),'ball_speed':np.full(n,2.5)}
    def test_repair_keeps_input_missingness_and_does_not_extrapolate(self):
        s=self.sequence();original=s['ball'].copy();mask=np.zeros(100,bool);mask[:3]=True;mask[15:25]=True;mask[-3:]=True
        repaired,seen,filled=repair_ball(s,mask,25)
        np.testing.assert_array_equal(s['ball'],original)
        self.assertEqual(filled.sum(),10);self.assertTrue(np.isnan(repaired['ball'][:3]).all());self.assertTrue(np.isnan(repaired['ball'][-3:]).all())
        np.testing.assert_allclose(repaired['ball'][15:25],original[15:25]);self.assertFalse(seen[15:25].any())
    def test_long_gap_and_impossible_anchor_speed_remain_missing(self):
        s=self.sequence();mask=np.zeros(100,bool);mask[20:80]=True
        r,_,filled=repair_ball(s,mask,50);self.assertFalse(filled.any());self.assertTrue(np.isnan(r['ball'][mask]).all())
        s['ball'][26]=[1000,0];mask[:]=False;mask[20:26]=True
        _,_,filled=repair_ball(s,mask,25);self.assertFalse(filled.any())
    def test_features_ignore_labels_and_are_finite_when_all_ball_missing(self):
        s=self.sequence();mask=np.ones(100,bool)
        a=gap_features(s,np.array([105,68]),mask)
        b=gap_features({**s,'y':np.full(100,2)},np.array([105,68]),mask)
        np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all());self.assertEqual(a.shape,(100,484))
    def test_same_team_spells_separated_by_dead_are_merge_diagnostic(self):
        y=np.array([0]*30+[2]*20+[0]*30)
        d=diagnostics(y,np.zeros(80,int))
        self.assertEqual(d['merges'],1)
        perfect=diagnostics(y,y,np.zeros(80,int));self.assertEqual(perfect['missed'],0);self.assertEqual(perfect['merges'],0)
    def test_long_mask_is_repeatable_and_label_independent(self):
        a=long_mask(1000,42);np.testing.assert_array_equal(a,long_mask(1000,42));self.assertGreaterEqual(a.mean(),.4)
if __name__=='__main__':unittest.main()
