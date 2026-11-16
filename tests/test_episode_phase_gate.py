import unittest,numpy as np
from controlswitch.episode_phase_gate import episode_filter
class EpisodePhaseGateTests(unittest.TestCase):
    def test_whole_episode_opposition_required_unknown_and_conflicts_preserved(self):
        p=np.zeros(100,int);events=[{'start':40,'end':50,'team':1}];opposition=[{'start':35,'end':55,'team':0,'code':'build up'}];q,r=episode_filter(p,events,opposition);np.testing.assert_array_equal(q,p);assert r[0]['phase_veto']
        for tags in ([],[{'start':40,'end':45,'team':0,'code':'build up'}],opposition+[{'start':45,'end':46,'team':1,'code':'build up'}]):
            q,r=episode_filter(p,events,tags);assert (q[40:50]==1).all();assert not r[0]['phase_veto']
    def test_team_swap(self):
        p=np.zeros(100,int);events=[{'start':40,'end':50,'team':1}];tags=[{'start':35,'end':55,'team':0,'code':'progression'}];a,_=episode_filter(p,events,tags);b,_=episode_filter(1-p,[{**e,'team':1-e['team']} for e in events],[{**e,'team':1-e['team']} for e in tags]);np.testing.assert_array_equal(a,1-b)
