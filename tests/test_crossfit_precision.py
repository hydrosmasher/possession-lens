import unittest
try:
 import torch
 from controlswitch.crossfit_precision import folds,training_files
except ImportError:torch=None
@unittest.skipIf(torch is None,'optional torch not installed')
class CrossfitTests(unittest.TestCase):
 def test_each_match_is_excluded_exactly_once(self):
  ids=['a','b','c','d','e'];spec=folds(ids);self.assertEqual([x['held'] for x in spec],ids)
  for f in spec:self.assertNotIn(f['held'],f['train']);self.assertEqual(set(f['train'])|{f['held']},set(ids))
 def test_duplicate_matches_rejected(self):
  with self.assertRaises(ValueError):folds(['a','b','a'])

 def test_normalisation_and_training_file_selection_excludes_held_match(self):
  paths={'a':['a1','a2'],'b':['b1'],'c':['c1']}
  self.assertEqual(training_files(paths,{'held':'a','train':['b','c']}),['b1','c1'])
 def test_leaking_fold_file_selection_is_rejected(self):
  paths={'a':['a1'],'b':['b1'],'c':['c1']}
  with self.assertRaises(ValueError):training_files(paths,{'held':'a','train':['a','b']})
