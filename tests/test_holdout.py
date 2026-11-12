import tempfile,unittest
from pathlib import Path
from controlswitch.rnn_holdout import verify_frozen
from controlswitch.pipeline import digest
class HoldoutTests(unittest.TestCase):
 def test_changed_weights_or_scoring_settings_are_rejected_before_loading_test(self):
  with tempfile.TemporaryDirectory() as d:
   a=Path(d)/'a';b=Path(d)/'b';a.write_bytes(b'frozen-a');b.write_bytes(b'frozen-b')
   p={'match_id':'J03WR9','tuning_on_this_match':False,'competition_data_used':False,'candidate':'a','reference':'b','checkpoint_sha256':{'a':digest(a),'b':digest(b)},'baseline_window':50,'dropout_seed':300};verify_frozen(p,a,b)
   a.write_bytes(b'changed')
   with self.assertRaises(ValueError):verify_frozen(p,a,b)
   a.write_bytes(b'frozen-a');p['dropout_seed']=301
   with self.assertRaises(ValueError):verify_frozen(p,a,b)
