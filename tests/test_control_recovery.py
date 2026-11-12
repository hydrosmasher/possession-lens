import unittest
import numpy as np
from controlswitch.control_recovery import recover_ids,bridge_control,ledger_from_ownership
class RecoveryTests(unittest.TestCase):
 def frames(self,end=7,group='home team'):
  return [{'frame':i,'period':1,'data':[{'track_id':5,'trackable_object':tid} if tid is not None else {'track_id':5,'group_name':group}]} for i,tid in enumerate([7,None,None,end])]
 def test_identity_requires_same_id_track_team_and_contiguous_frames(self):
  x=self.frames();self.assertEqual(recover_ids(x,{7:0}),2);self.assertEqual(x[1]['data'][0]['trackable_object'],7)
  for x in [self.frames(end=8),self.frames(group='away team')]:self.assertEqual(recover_ids(x,{7:0,8:0}),0)
  x=self.frames();x[2]['frame']=4;self.assertEqual(recover_ids(x,{7:0}),0)
 def test_control_bridge_rejects_ambiguity_and_team_change(self):
  xy=np.zeros((8,3,2),np.float32);xy[:,1:]=20;ball=np.zeros((8,2),np.float32);ball[2:6]=np.nan;teams=np.array([0,0,1]);y=np.full(8,'H')
  owners,added=bridge_control(y,xy,ball,teams,2,4);self.assertEqual(int(added.sum()),4);self.assertTrue((owners[2:6]==0).all())
  ledger=ledger_from_ownership(y,owners,teams);self.assertAlmostEqual(sum(r['associated_control_seconds'] for r in ledger['rows']),8/25)
  xy[3,1]=[1,0];self.assertFalse(bridge_control(y,xy,ball,teams,2,4)[1].any());xy[:,1]=20;y[4]='A';self.assertFalse(bridge_control(y,xy,ball,teams,2,4)[1].any())
 def test_long_gap_and_missing_player_never_filled(self):
  xy=np.zeros((8,2,2));xy[:,1]=20;ball=np.zeros((8,2));ball[2:6]=np.nan;y=np.full(8,'H');teams=np.array([0,1])
  self.assertFalse(bridge_control(y,xy,ball,teams,1,3)[1].any());xy[3,0]=np.nan;self.assertFalse(bridge_control(y,xy,ball,teams,1,4)[1].any())
