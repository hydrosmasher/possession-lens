import unittest,tempfile
from pathlib import Path
import numpy as np
from controlswitch.rnn import initialise,RNN_DWELL_VERSION,RNN_SYMMETRIC_VERSION,observation_probabilities
from controlswitch.model import instantaneous_features,decode
from controlswitch.rnn_runtime import load_frozen,infer_frozen

class FrozenRuntimeTests(unittest.TestCase):
 def test_saved_symmetric_and_dwell_models_roundtrip(self):
    rng=np.random.default_rng(6);s={'xy':rng.normal(size=(25,4,2)),'speed':np.ones((25,4)),
    'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(25,2)),'ball_speed':np.ones(25)}
    pitch=np.array([105,68]);m=initialise(instantaneous_features(s,pitch),hidden=4)
    with tempfile.TemporaryDirectory() as tmp:
        p=Path(tmp)/'m.npz'
        for version in (RNN_SYMMETRIC_VERSION,RNN_DWELL_VERSION):
            np.savez(p,**m,version=np.array(version),symmetric=np.array(True),transition_penalty=np.array(6),min_frames=np.array(1))
            labels,probs=infer_frozen(load_frozen(p),s,pitch)
            expected=observation_probabilities(m,s,pitch,True)
            np.testing.assert_allclose(probs,expected)
            np.testing.assert_array_equal(labels,decode(expected,6))
        np.savez(p,**m,version=np.array('unsupported'))
        with self.assertRaises(ValueError):load_frozen(p)
