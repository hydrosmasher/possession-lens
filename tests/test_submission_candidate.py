import unittest
from unittest.mock import patch
import numpy as np
from controlswitch.submission import predict_halves,submission_filename

class SubmissionCandidateTests(unittest.TestCase):
    def observations(self):
        return {'xy':np.zeros((12,2,2)),'speed':np.ones((12,2)),'teams':np.array([0,1]),'ball':np.zeros((12,2)),'ball_speed':np.ones(12)}
    def test_half_resets_and_exterior_abstentions(self):
        seen=[]
        def infer(candidate,s,pitch):
            seen.append(len(s['ball']))
            self.assertNotIn('y',s)
            return np.full(len(s['ball']),len(seen)-1),None
        with patch('controlswitch.submission.infer_candidate',side_effect=infer):
            labels=predict_halves({},self.observations(),np.array([105,68]),[(2,5),(7,11)])
        self.assertEqual(seen,[3,4]);self.assertEqual(''.join(labels),'XXHHHXXAAAAX')
    def test_unicode_team_filename_and_path_rejection(self):
        self.assertEqual(submission_filename("Visca el Barça, visca Catalunya.csv"), "Visca el Barça, visca Catalunya.csv")
        for value in ("../answer.csv", "folder/answer.csv", "folder\\answer.csv", "answer.txt", ""):
            with self.assertRaises(ValueError):submission_filename(value)
    def test_overlapping_halves_rejected(self):
        with patch('controlswitch.submission.infer_candidate',return_value=(np.zeros(4,int),None)):
            with self.assertRaises(ValueError):predict_halves({},self.observations(),np.array([105,68]),[(1,5),(4,8)])
if __name__=='__main__':unittest.main()
