import unittest,tempfile,json
from pathlib import Path
import numpy as np
from controlswitch.skillcorner_europe import interpolate,load_match

class EuropeAdapterTests(unittest.TestCase):
 def test_interpolation_does_not_bridge_missing_tracking(self):
  a=np.array([[0.,0.],[1.,2.],[np.nan,np.nan],[3.,6.]])
  p=interpolate(a,np.array([0.,.4,1.,1.4,2.,2.4,3.]))
  np.testing.assert_allclose(p[:3],[[0,0],[.4,.8],[1,2]])
  self.assertTrue(np.isnan(p[3:6]).all());np.testing.assert_allclose(p[-1],[3,6])
 def test_null_is_unknown_and_referee_is_excluded(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)/'unit';p.mkdir();m={'home_team':{'id':1},'players':[{'team_id':1,'trackable_object':11},{'team_id':2,'trackable_object':22}],'ball':{'trackable_object':55},'pitch_length':105,'pitch_width':68}
   frames=[{'period':period,'frame':i+period*100,'possession':{'group':None if i==1 else 'home team'},'data':[{'x':i,'y':0,'trackable_object':11},{'x':2,'y':0,'trackable_object':22},{'x':1,'y':0,'trackable_object':55},{'x':0,'y':0,'group_name':'referee','track_id':9}]} for period in (1,2) for i in range(3)]
   (p/'match_data.json').write_text(json.dumps(m));(p/'structured_data.json').write_text(json.dumps(frames))
   parts,audit=load_match(tmp,'unit')
   self.assertEqual(len(parts),2)
   for s,_ in parts:
    self.assertTrue(3 in s['y']);self.assertFalse(2 in s['y']);self.assertTrue(np.isnan(s['xy'][:,2:]).all())
   self.assertEqual(audit[0]['labelled_native_frames'],2)
