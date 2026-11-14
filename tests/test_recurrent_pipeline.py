import json,tempfile,unittest
from pathlib import Path
import numpy as np
from controlswitch.recurrent_pipeline import load_pipeline,checked_path,VERSION
try:
 import torch
 from controlswitch.recurrent_pipeline import infer
 from controlswitch.motion_gru import predict_variants
except ImportError:
 torch=None
ROOT=Path(__file__).resolve().parents[1]

class RecurrentConfigTests(unittest.TestCase):
 def test_escape_and_unsupported_mixture_are_rejected(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp)
   with self.assertRaises(ValueError):checked_path(root,{'path':'../escape.npz','sha256':'bad'})
   p=root/'config.json';p.write_text(json.dumps({'version':VERSION,'gru_mix':.6}))
   with self.assertRaises(ValueError):load_pipeline(p,root)

@unittest.skipIf(torch is None,'Optional PyTorch inference dependency')
class RecurrentSavedModelTests(unittest.TestCase):
 def test_saved_pipeline_matches_study_and_ignores_reference_labels(self):
  b=load_pipeline(ROOT/'reports/external/motion-gru-043-config.json');rng=np.random.default_rng(43);n=80;s={'xy':rng.normal(size=(n,4,2)),'teams':np.array([0,0,1,1]),'speed':np.ones((n,4)),'ball':rng.normal(size=(n,2)),'ball_speed':np.ones(n)};mask=np.arange(n)%3==0;pitch=np.array([105,68]);pred,p=infer(b,s,pitch,mask);_,old=predict_variants(b['base'],b['gru'],s,pitch,mask);np.testing.assert_array_equal(pred,old['gru0.5']);q,r=infer(b,{**s,'y':np.full(n,2)},pitch,mask);np.testing.assert_array_equal(pred,q);np.testing.assert_array_equal(p,r)
if __name__=='__main__':unittest.main()
