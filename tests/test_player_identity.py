import unittest
import numpy as np
from controlswitch.player_identity import link_fragments,score_links,fragments

def f(start,end,x,team=0,truth='a',velocity=(0,0)):
 return {'start':start,'end':end,'first':np.array([x,0.]),'last':np.array([x,0.]),'velocity':np.array(velocity,dtype=float),'team':team,'slot':0,'truth':truth}

class IdentityLinkTests(unittest.TestCase):
 def test_motion_link_does_not_use_truth_or_slot_number(self):
    fs=[f(0,10,0,velocity=(2,0)),f(15,25,.48,velocity=(2,0))]
    ids,links=link_fragments(fs,max_gap=25,max_error=.2)
    self.assertEqual(ids[0],ids[1]);self.assertEqual(len(links),1)
    fs[1]['truth']='different';fs[1]['slot']=100
    other,_=link_fragments(fs,max_gap=25,max_error=.2);np.testing.assert_array_equal(ids,other)
 def test_overlapping_tracks_and_other_teams_never_merge(self):
    fs=[f(0,20,0),f(10,30,0),f(31,40,0,team=1)]
    ids,links=link_fragments(fs);self.assertEqual(len(set(ids)),3);self.assertEqual(links,[])
 def test_ambiguous_join_is_withheld(self):
    fs=[f(0,10,-.1,truth='a'),f(0,10,.1,truth='b'),f(15,20,0)]
    ids,links=link_fragments(fs,max_error=1,ambiguity_margin=.5)
    self.assertEqual(len(set(ids)),3);self.assertEqual(links,[])
 def test_fragment_simulation_is_reproducible_and_truth_is_only_for_scoring(self):
    xy=np.zeros((1000,2,2));xy[:,1,0]=10;teams=np.array([0,1])
    a=fragments(xy,teams,12,['a','b'],True,7);b=fragments(xy,teams,12,['x','y'],True,7)
    self.assertEqual([(f['start'],f['end']) for f in a],[(f['start'],f['end']) for f in b])
    ids,links=link_fragments(a,max_gap=50);score=score_links(a,links)
    self.assertEqual(score['false_links'],0)
