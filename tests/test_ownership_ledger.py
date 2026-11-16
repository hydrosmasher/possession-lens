import unittest
import numpy as np
from controlswitch.ownership_ledger import association, decode_ledger



class OwnershipLedgerTests(unittest.TestCase):
    def test_pass_and_brief_interception(self):
        q = np.tile([.9, .1], (100, 1)); r = np.tile([.9, .05], (100, 1))
        # Flight with a brief opponent-nearness error preserves the team ledger.
        r[30:60] = .02; q[40:43] = [.2, .8]
        labels, _ = decode_ledger(q, r, np.zeros(100, int))
        assert (labels == 0).all()
        # Strong evidence permits a short real interception without duration floor.
        q[40:48] = [.001, .999]; r[40:48] = [.001, .999]
        labels, _ = decode_ledger(q, r, np.zeros(100, int))
        assert (labels[40:48] == 1).all()
        assert labels[39] == labels[48] == 0


    def test_missing_ball_and_dead_mask(self):
        x = np.zeros((100, 2, 16, 21), np.float32); x[..., 20] = 1
        _, r = association(x)
        assert (r == 0).all()
        original = np.zeros(100, int); original[45:55] = 2
        labels, states = decode_ledger(np.tile([.8, .2], (100, 1)), r, original)
        assert np.array_equal(labels == 2, original == 2)
        assert (states[45:55] == -1).all()


    def test_team_swap(self):
        rng = np.random.default_rng(31)
        q = rng.uniform(.01, 1, (200, 2)); r = rng.uniform(0, 1, (200, 2))
        frozen = np.zeros(200, int); frozen[70:75] = 2
        a, _ = decode_ledger(q, r, frozen)
        b, _ = decode_ledger(q[:, ::-1], r[:, ::-1], frozen)
        assert np.array_equal(np.where(a == 2, 2, 1-a), b)

    def test_absent_evidence_preserves_existing_team_predictions(self):
        frozen = np.r_[np.zeros(20, int), np.ones(2, int), np.zeros(20, int), [2]]
        q = np.tile([.99, .01], (len(frozen), 1))
        r = np.zeros_like(q)
        labels, states = decode_ledger(q, r, frozen, preserve_unsupported=True)
        np.testing.assert_array_equal(labels, frozen)
        assert (states == -1).all()
