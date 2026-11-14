import unittest
import numpy as np
from controlswitch.uncertainty_had import abstain,evaluate
from controlswitch.tracker_robustness import CONDITIONS

class MissingBallAbstentionTests(unittest.TestCase):
    def test_only_missing_ball_and_weak_live_team_evidence_abstains(self):
        labels=np.array([0,0,2,1]);p=np.array([[.49,.49,.02],[.49,.49,.02],[.49,.49,.02],[.05,.9,.05]])
        got=abstain(labels,p,np.array([False,True,False,False]),.1)
        np.testing.assert_array_equal(got,[3,0,2,1]);np.testing.assert_array_equal(labels,[0,0,2,1])
    def test_official_temporal_fill_and_raw_frame_penalty_are_separate(self):
        y=np.zeros(50,int);p=y.copy();p[20:30]=3
        rows=[{'y':y,'condition':c} for c in CONDITIONS];score=evaluate(rows,[p.copy() for c in CONDITIONS])
        self.assertEqual(score['mean_spell_f1'],1)
        self.assertEqual(score['totals']['missed'],0);self.assertEqual(score['totals']['false_positive'],0)
        self.assertLess(score['clean']['macro_f1'],1/3)
    def test_invalid_confidence_rejected(self):
        with self.assertRaises(ValueError):abstain(np.zeros(2,int),np.ones((2,3)),np.zeros(2,bool),np.nan)
if __name__=='__main__':unittest.main()
