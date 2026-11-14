import copy,unittest
from controlswitch.boundary_alignment import eligible
from controlswitch.tracker_robustness import CONDITIONS

class BoundaryAlignmentSelectionTests(unittest.TestCase):
    def setUp(self):
        scores={c:{'spell_f1_mean':.5,'dead_f1':.9} for c in CONDITIONS}
        better={c:{'spell_f1_mean':.51,'dead_f1':.9} for c in CONDITIONS}
        self.folds=[{'current':copy.deepcopy(scores),'candidates':{'candidate':copy.deepcopy(better)}} for _ in range(2)]
    def test_average_gain_cannot_hide_single_noise_fold_regression(self):
        self.folds[1]['candidates']['candidate']['tracker_noise']['spell_f1_mean']=.49
        guards=eligible(self.folds,'candidate',{'mean_spell_f1':.5},{'mean_spell_f1':.51})
        self.assertTrue(guards['mean_gain_at_least_003']);self.assertFalse(guards['tracker_noise'])
    def test_small_gain_or_damaged_dead_state_is_ineligible(self):
        self.assertFalse(eligible(self.folds,'candidate',{'mean_spell_f1':.5},{'mean_spell_f1':.502})['mean_gain_at_least_003'])
        self.folds[0]['candidates']['candidate']['clean']['dead_f1']=.89
        self.assertFalse(eligible(self.folds,'candidate',{'mean_spell_f1':.5},{'mean_spell_f1':.51})['dead_f1'])
if __name__=='__main__':unittest.main()
