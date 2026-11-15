import unittest,numpy as np
from pathlib import Path
from controlswitch.recurrent_boundary import evidence,event_features,decode_head
from controlswitch.recurrent_pipeline import load_pipeline,infer
class RecurrentBoundaryTests(unittest.TestCase):
 def test_features_finite_and_shape_checked(self):
  x=np.zeros((30,116),np.float32);e=np.linspace(0,1,30);f=event_features(x,e)
  self.assertEqual(f.shape,(30,120));self.assertTrue(np.isfinite(f).all());np.testing.assert_array_equal(f[:,:116],x)
  with self.assertRaises(ValueError):event_features(x,np.full(30,np.nan))
  with self.assertRaises(ValueError):event_features(x[:,:115],e)
 def test_evidence_preserves_frozen_emissions_and_ignores_labels(self):
  r=Path(__file__).resolve().parents[1];b=load_pipeline(r/'reports/external/transfer-052-config.json');rng=np.random.default_rng(54);n=80;s={'xy':rng.normal(size=(n,4,2)),'teams':np.array([0,0,1,1]),'speed':np.ones((n,4)),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)};pitch=np.array([105,68]);mask=np.arange(n)%4==0;q,x,e=evidence(b,s,pitch,mask);labels,p=infer(b,s,pitch,mask);np.testing.assert_array_equal(p,q)
  np.testing.assert_array_equal(decode_head(b['base']['models']['boundary'],q,x,e),labels)
  q2,x2,e2=evidence(b,{**s,'y':np.full(n,2)},pitch,mask)
  for a,z in ((q,q2),(x,x2),(e,e2)):np.testing.assert_array_equal(a,z)
if __name__=='__main__':unittest.main()
