import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch.europe_context import observation_features,observation_probabilities,save_model
from controlswitch import rnn_deep as net
from controlswitch.rnn_runtime import load_frozen,infer_frozen
from controlswitch.model import decode

class ContextTests(unittest.TestCase):
 def sequence(self,n=100):
  rng=np.random.default_rng(4)
  return {'xy':rng.normal(size=(n,4,2)).astype(np.float32),'speed':np.zeros((n,4),np.float32),'ball':rng.normal(size=(n,2)).astype(np.float32),'ball_speed':np.zeros(n,np.float32),'teams':np.array([0,0,1,1]),'y':np.zeros(n,np.int8)}
 def test_context_uses_tracking_not_labels_and_has_expected_width(self):
  s=self.sequence();pitch=np.array([105,68]);a=observation_features(s,pitch,context=True);self.assertEqual(a.shape,(100,120));s['y'][:]=3;np.testing.assert_array_equal(a,observation_features(s,pitch,context=True))
 def test_context_checkpoint_roundtrip_and_symmetry(self):
  s=self.sequence();pitch=np.array([105,68]);x=observation_features(s,pitch,context=True);m=net.initialise(x,(4,));p=observation_probabilities(m,s,pitch,context=True)
  swap={**s,'teams':1-s['teams']};np.testing.assert_allclose(p[:,[1,0,2]],observation_probabilities(m,swap,pitch,context=True),atol=1e-6)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'m.npz';save_model(path,m,12,True);f=load_frozen(path);pred,restored=infer_frozen(f,s,pitch);np.testing.assert_allclose(p,restored,atol=1e-6);np.testing.assert_array_equal(pred,decode(p,12))
 def test_dropout_applies_before_temporal_context(self):
  s=self.sequence();mask=np.ones(100,bool);x=observation_features(s,[105,68],mask,context=True);self.assertTrue(np.isfinite(x).all());self.assertTrue((x[:,0]==0).all())
 def test_additions_import_is_opaque_and_bad_hash_does_not_install_test(self):
  import zipfile,hashlib
  from unittest.mock import patch
  from controlswitch.context_data import import_additions
  raw=b'opaque heldout contents, deliberately not JSON';size=len(raw);sha=hashlib.sha1(f'blob {size}\0'.encode()+raw).hexdigest()
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp);archive=p/'a.zip'
   with zipfile.ZipFile(archive,'w') as z:z.writestr('release/data/matches/2841/structured_data.json',raw);z.writestr('release/LICENSE','MIT test fixture')
   with patch('controlswitch.context_data.ASSETS',{'2841':[('structured_data.json',size,'0'*40)]}):
    with self.assertRaises(ValueError):import_additions(archive,p/'bad')
    self.assertFalse((p/'bad/2841').exists())
   with patch('controlswitch.context_data.ASSETS',{'2841':[('structured_data.json',size,sha)]}):
    result=import_additions(archive,p/'good');self.assertFalse(result['reserved_tracking_parsed']);self.assertEqual((p/'good/2841/structured_data.json').read_bytes(),raw)
    with self.assertRaises(ValueError):import_additions(archive,p/'good')
