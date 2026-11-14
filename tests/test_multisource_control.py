import unittest
from unittest.mock import patch
import numpy as np
from controlswitch.multisource_control import skill_training

class MultiSourceSplitTests(unittest.TestCase):
    def test_added_training_never_opens_validation_or_viewed_test_matches(self):
        n=100
        s={'xy':np.zeros((n,4,2)),'speed':np.ones((n,4)),'teams':np.array([0,0,1,1]),'ball':np.zeros((n,2)),'ball_speed':np.zeros(n),'y':np.array([0]*30+[1]*30+[3]*40)}
        opened=[]
        def loader(root,match):
            opened.append(match)
            return [(s,np.array([105,68])),(s,np.array([105,68]))],[]
        with patch('controlswitch.multisource_control.load_match',side_effect=loader):x,y,audit=skill_training('unused')
        self.assertTrue(set(opened).isdisjoint({'2269','2841','2417','3442'}))
        self.assertEqual(len(opened),5);self.assertEqual(x.shape[1],244)
        self.assertEqual(set(y),{0,1});self.assertEqual(int((y==0).sum()),int((y==1).sum()))
        self.assertTrue(all(row['sampled_unique_frames']==60 for row in audit))
if __name__=='__main__':unittest.main()
