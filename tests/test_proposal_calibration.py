import unittest,numpy as np
from controlswitch.proposal_calibration import apply_scores,choose_threshold,retention
class ProposalCalibrationTests(unittest.TestCase):
    def test_calibration_retains_same_recovered_spell(self):
        y=np.r_[np.zeros(80,int),np.ones(20,int),np.zeros(100,int)];base=np.zeros(200,int);events=[{'start':80,'end':100,'team':1},{'start':150,'end':155,'team':1}];full=apply_scores(base,events,[1,1],0);case={'y':y,'current':base,'prototype':full,'events':events};threshold,audit=choose_threshold([case],[np.array([.8,.2])]);filtered=apply_scores(base,events,[.8,.2],threshold);self.assertGreater(threshold,.2);self.assertEqual(retention(y,base,full,filtered)['lost_recovered'],0);np.testing.assert_array_equal(filtered,y)
    def test_low_scoring_true_recovery_prevents_harmful_threshold(self):
        y=np.r_[np.zeros(80,int),np.ones(20,int),np.zeros(100,int)];base=np.zeros(200,int);events=[{'start':80,'end':100,'team':1},{'start':150,'end':155,'team':1}];full=apply_scores(base,events,[1,1],0);threshold,_=choose_threshold([{'y':y,'current':base,'prototype':full,'events':events}],[np.array([.1,.9])]);self.assertLessEqual(threshold,.1)
