import unittest
import numpy as np
from controlswitch.rnn_boundaries import pulse_audit
class PulseAuditTests(unittest.TestCase):
 def test_label_gap_clipping_is_not_flicker(self):
  pred=np.zeros(100,np.int8);y=np.full(100,3,np.int8);y[20:23]=0;y[50:55]=0
  a=pulse_audit(y,pred);self.assertEqual(a['scored_spells_under_0_4_seconds'],2);self.assertEqual(a['scored_short_fragments_from_longer_predictions'],2);self.assertEqual(a['full_prediction_spells_under_0_4_seconds'],0)
 def test_real_short_run_is_retained(self):
  pred=np.zeros(100,np.int8);pred[20:26]=1;y=np.zeros(100,np.int8);a=pulse_audit(y,pred)
  self.assertEqual(a['full_prediction_spells_under_0_4_seconds'],1);self.assertEqual(a['scored_spells_under_0_4_seconds'],1);self.assertEqual(a['scored_short_fragments_from_longer_predictions'],0)
 def test_exact_threshold_and_shape(self):
  pred=np.zeros(100,np.int8);pred[20:30]=1;a=pulse_audit(pred,pred);self.assertEqual(a['full_prediction_spells_under_0_4_seconds'],0)
  with self.assertRaises(ValueError):pulse_audit(np.zeros(3),np.zeros(4))
