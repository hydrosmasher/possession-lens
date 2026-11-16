import unittest,numpy as np
from controlswitch.observation_quality import quality_report
class ObservationQualityTests(unittest.TestCase):
    def scene(self,n=200):
        return {'ball':np.zeros((n,2)), 'xy':np.tile(np.arange(12).reshape(6,2)[None],(n,1,1)).astype(float),'teams':np.array([0,0,0,1,1,1])}
    def test_counts_and_no_labels_or_coordinates_exported(self):
        s=self.scene();s['ball'][30:90]=np.nan;s['y']=np.zeros(200,int);a=quality_report(s);assert a['summary']['ball_missing_frames']==60 and a['summary']['long_gap_frames']==60;s['y'][:]=2;self.assertEqual(a,quality_report(s));assert 'xy' not in a and 'ball' not in a;assert not a['possession_labels_used']
    def test_half_reset_and_outside_frames_excluded(self):
        s=self.scene();s['ball'][40:70]=np.nan;s['ball'][110:140]=np.nan;s['ball'][100:]=100
        # End and start gaps must not join across an excluded halftime interval.
        s['ball'][110:140]=np.nan;r=quality_report(s,[(10,70),(110,170)])
        assert r['summary']['analysed_frames']==120;assert r['summary']['ball_missing_frames']==60;assert r['summary']['long_gap_frames']==0;assert r['summary']['implausible_ball_step_frames']==0
    def test_player_visibility_and_impossible_ball_step(self):
        s=self.scene();s['xy'][20:25,0]=np.nan;s['ball'][50]=[100,0];r=quality_report(s);assert r['summary']['limited_player_frames']==5;assert r['summary']['implausible_ball_step_frames']==2
    def test_rejects_overlapping_blocks(self):
        with self.assertRaises(ValueError):quality_report(self.scene(),[(0,100),(90,150)])
