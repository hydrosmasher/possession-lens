"""084: preserve predicted short spells while refining longer control."""
import numpy as np


def preservation_anchors(frozen, candidate, seen):
    """Only predictions/visibility, no reference labels.

    At IoU>=.5 a matched predicted spell cannot exceed twice the reference
    duration. Therefore protect predicted H/A spells <=150frames plus their
    immediate flanks to cover <=75frame references in fully scored runs.
    Actual reference-run/short retention is still checked externally.
    """
    frozen, candidate, seen = np.asarray(frozen), np.asarray(candidate), np.asarray(seen, bool)
    n = len(frozen)
    if frozen.shape != (n,) or candidate.shape != (n,) or seen.shape != (n,) or not np.isin(frozen, [0, 1, 2]).all() or not np.isin(candidate, [0, 1, 2]).all() or not np.array_equal(frozen == 2, candidate == 2) or not np.array_equal(frozen[~seen], candidate[~seen]):
        raise ValueError('Invalid short-preservation inputs')
    anchors = np.full(n, -1, int)
    missing = ~seen & (frozen < 2); anchors[missing] = frozen[missing]
    changed = candidate != frozen; anchors[changed] = candidate[changed]
    edges = np.r_[0, np.flatnonzero(candidate[1:] != candidate[:-1])+1, n]
    protected = []
    if not n: return anchors, protected
    for a, b in zip(edges[:-1], edges[1:]):
        if candidate[a] == 2 or b-a > 150: continue
        lo, hi = max(0, a-1), min(n, b+1)
        ix = np.arange(lo, hi); ix = ix[candidate[ix] < 2]
        anchors[ix] = candidate[ix]
        protected.append({'start':int(a),'end':int(b),'team':int(candidate[a])})
    return anchors, protected
