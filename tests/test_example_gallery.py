import unittest,json,tempfile
from pathlib import Path
from controlswitch.example_gallery import roster,named_ledger

class GalleryTests(unittest.TestCase):
 def test_roster_rejects_ambiguous_or_missing_names(self):
  m={'players':[{'trackable_object':4,'first_name':'Ada','last_name':'Smith','number':7}]};self.assertEqual(roster(m)[0]['player_name'],'Ada Smith')
  m['players'].append(m['players'][0].copy())
  with self.assertRaises(ValueError):roster(m)
  with self.assertRaises(ValueError):roster({'players':[{'trackable_object':5}]})
 def test_anonymous_association_is_unassigned_never_renamed(self):
  players=[{'person_id':'4','player_name':'Ada Smith','shirt_number':7}]
  ledger={'bouts':[{'track_slot':1,'team':'H'},{'track_slot':2,'team':'H'}],'rows':[{'track_slot':1,'team':'H','associated_control_seconds':2.},{'track_slot':2,'team':'H','associated_control_seconds':3.},{'track_slot':'UNASSIGNED','team':'H','associated_control_seconds':5.},{'track_slot':'UNASSIGNED','team':'A','associated_control_seconds':0.}]}
  rows,missing=named_ledger(ledger,players,{'H':10.,'A':0.});self.assertEqual(missing['H'],8.);self.assertEqual(len(rows),1);self.assertEqual(rows[('H','4')]['associated_bouts'],1)
 def test_bundled_library_has_mapped_people_and_reconciles(self):
  path=Path(__file__).resolve().parents[1]/'examples/match-library.json'
  if not path.exists():self.skipTest('Library is an optional generated artifact')
  library=json.loads(path.read_text());self.assertEqual(len(library['matches']),10);seen=set()
  for match in library['matches']:
   self.assertNotIn(match['example_id'],seen);seen.add(match['example_id']);self.assertEqual(match['mode'],'external_named');self.assertIsNone(match['tracklets'])
   for team in 'HA':
    rows=[r for r in match['named_players']['rows'] if r['team']==team];self.assertAlmostEqual(sum(r['percent_of_team_possession'] or 0 for r in rows),100.,places=5)
    for r in rows:
     if r['person_id']!='UNASSIGNED':self.assertTrue(r['player_name'].strip());self.assertNotIn('Anonymous',r['player_name'])
