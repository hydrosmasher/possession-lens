import itertools
import unittest
import numpy as np
from controlswitch.rnn_boundaries import dwell_decode,invariant_boundary_features
from controlswitch.rnn import initialise,observation_probabilities
from controlswitch.model import instantaneous_features,decode

class DwellDecoderTests(unittest.TestCase):
 def test_matches_exhaustive_paths_with_truncated_end_runs(self):
    p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.8,.1],[.8,.1,.1],[.8,.1,.1]])
    boundary=np.array([.1,.9,.4,.8,.2]);penalty=.3;strength=1.;minimum=2
    costs=penalty*np.clip(1+strength*(1-2*boundary),.15,3)
    def allowed(path):
        cuts=[0]+[i for i in range(1,len(path)) if path[i]!=path[i-1]]+[len(path)]
        return all(b-a>=minimum for a,b in list(zip(cuts,cuts[1:]))[1:-1])
    def score(path):return sum(np.log(p[i,k]) for i,k in enumerate(path))-sum(costs[i] for i in range(1,len(path)) if path[i]!=path[i-1])
    expected=max((x for x in itertools.product(range(3),repeat=len(p)) if allowed(x)),key=score)
    np.testing.assert_array_equal(dwell_decode(p,penalty,boundary,strength,minimum),expected)
 def test_minimum_one_preserves_existing_boundary_decoder(self):
    p=np.random.default_rng(3).dirichlet([1,1,1],60);b=np.linspace(0,1,60)
    np.testing.assert_array_equal(dwell_decode(p,6,b,1,1),decode(p,6,b,1))
 def test_floor_blocks_one_frame_excursion_but_keeps_sustained_control(self):
    p=np.tile([.99,.005,.005],(30,1));p[10]=[.005,.99,.005];p[20:]=[.005,.99,.005]
    labels=dwell_decode(p,.1,np.full(30,.9),1,5)
    self.assertEqual(int((labels[1:]!=labels[:-1]).sum()),1)
    self.assertEqual(labels[10],0);self.assertEqual(labels[-1],1)
 def test_boundary_features_are_exactly_team_invariant(self):
    rng=np.random.default_rng(1);s={'xy':rng.normal(size=(40,4,2)),'speed':np.ones((40,4)),
    'teams':np.array([0,0,1,1]),'ball':rng.normal(size=(40,2)),'ball_speed':np.ones(40)}
    pitch=np.array([105,68]);m=initialise(instantaneous_features(s,pitch),hidden=4)
    p=observation_probabilities(m,s,pitch,True);other={**s,'teams':1-s['teams']}
    np.testing.assert_allclose(invariant_boundary_features(s,pitch,p),invariant_boundary_features(other,pitch,p[:,[1,0,2]]),atol=1e-7)
