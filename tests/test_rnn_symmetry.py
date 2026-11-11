import unittest
import numpy as np
from controlswitch.rnn import initialise,observation_probabilities
from controlswitch.model import instantaneous_features

class TeamSymmetryTests(unittest.TestCase):
 def fixture(self):
    rng=np.random.default_rng(9);n=30
    return {'xy':rng.normal(size=(n,4,2))*10,'speed':rng.uniform(0,6,(n,4)),
            'ball':rng.normal(size=(n,2))*10,'ball_speed':rng.uniform(0,12,n),'teams':np.array([0,0,1,1])}
 def test_swapping_team_assignment_exactly_swaps_home_away_outputs(self):
    s=self.fixture();pitch=np.array([105.,68.]);m=initialise(instantaneous_features(s,pitch),hidden=4)
    other={**s,'teams':1-s['teams']}
    p=observation_probabilities(m,s,pitch,True);q=observation_probabilities(m,other,pitch,True)
    np.testing.assert_allclose(p,q[:,[1,0,2]],atol=1e-7)
    np.testing.assert_allclose(p.sum(axis=1),1,atol=1e-7)
 def test_symmetry_averaging_does_not_use_future_observations(self):
    s=self.fixture();pitch=np.array([105.,68.]);m=initialise(instantaneous_features(s,pitch),hidden=4)
    changed={k:v.copy() for k,v in s.items()}
    for k in ('xy','speed','ball','ball_speed'):changed[k][15:]+=100
    np.testing.assert_allclose(observation_probabilities(m,s,pitch,True)[:15],observation_probabilities(m,changed,pitch,True)[:15],atol=1e-7)
