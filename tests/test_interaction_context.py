import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch.interaction_context import observations,observation_probabilities
from controlswitch import rnn_deep as net
from controlswitch.rnn_runtime import load_frozen,infer_frozen,INTERACTION_CONTEXT_VERSION
class InteractionTests(unittest.TestCase):
 def test_features_exclude_labels_and_frozen_runtime_roundtrips(self):
  rng=np.random.default_rng(3);n=80;s={'xy':rng.normal(size=(n,4,2)).astype(np.float32),'speed':np.zeros((n,4),np.float32),'ball':rng.normal(size=(n,2)).astype(np.float32),'ball_speed':np.zeros(n,np.float32),'teams':np.array([0,0,1,1]),'y':np.zeros(n,np.int8)};pitch=np.array([105,68]);x=observations(s,pitch);self.assertEqual(x.shape,(n,130));s['y'][:]=3;np.testing.assert_array_equal(x,observations(s,pitch));m=net.initialise(x,(3,));p=observation_probabilities(m,s,pitch)
  with tempfile.TemporaryDirectory() as tmp:
   path=Path(tmp)/'m.npz';np.savez_compressed(path,**m,version=np.array(INTERACTION_CONTEXT_VERSION),symmetric=np.array(True),transition_penalty=np.array(48.));_,restored=infer_frozen(load_frozen(path),s,pitch);np.testing.assert_allclose(p,restored,atol=1e-6)
