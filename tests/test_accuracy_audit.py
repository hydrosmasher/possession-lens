import unittest
import numpy as np
from controlswitch.accuracy_audit import frame_report

class FrameAccuracyAuditTests(unittest.TestCase):
    def test_unknown_predictions_count_as_errors_but_unknown_reference_is_unscored(self):
        r=frame_report(np.array([0,1,2,3]),np.array([0,3,2,0]))
        self.assertEqual(r['n_scored'],3);self.assertEqual(r['correct'],2);self.assertAlmostEqual(r['accuracy'],2/3)
        self.assertEqual(r['errors']['abstention'],1)
    def test_error_types_are_separate_and_account_for_every_mistake(self):
        r=frame_report(np.array([0,1,2,0,1,2]),np.array([1,2,0,3,1,2]))
        self.assertEqual(sum(r['errors'].values()),4)
        self.assertEqual(r['errors'],{'wrong_live_team':1,'missed_dead_ball':1,'false_dead_ball':1,'abstention':1})
    def test_no_reference_labels_cannot_claim_accuracy(self):
        with self.assertRaises(ValueError):frame_report(np.full(10,3),np.zeros(10,int))
if __name__=='__main__':unittest.main()
