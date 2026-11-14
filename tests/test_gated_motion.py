import tempfile,unittest
from pathlib import Path
import numpy as np
try:
 import torch
 from controlswitch.gated_rnn import GatedPossession,predict,save_checkpoint,load_checkpoint
except ImportError:
 torch=None

@unittest.skipIf(torch is None,'Optional PyTorch dependency')
class GatedMotionLayoutTests(unittest.TestCase):
 def test_richer_layout_and64unit_roundtrip(self):
  torch.set_num_threads(1);torch.manual_seed(43);m=GatedPossession(layers=2,bidirectional=True,hidden=64,projection=64,input_features=508,boundary=True);x=np.random.default_rng(43).normal(size=(20,508)).astype(np.float32);mean=np.zeros(508,np.float32);scale=np.ones(508,np.float32);p,e=predict(m,x,mean,scale)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'model.npz';save_checkpoint(path,m,mean,scale,12);loaded=load_checkpoint(path);q,f=predict(loaded['model'],x,loaded['mean'],loaded['scale'])
  np.testing.assert_array_equal(p,q);np.testing.assert_array_equal(e,f);self.assertEqual(p.shape,(20,3));self.assertTrue(np.isfinite(p).all())
if __name__=='__main__':unittest.main()
