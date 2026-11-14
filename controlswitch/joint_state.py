"""508-feature combination of relative and collective motion evidence."""
import numpy as np
from .relative_state import relative_features,swap_relative
from .collective_state import collective_features
from .model import rolling,shifted

DYNAMIC=np.r_[np.arange(12,22),np.arange(38,48)]
def joint_features(s,pitch,mask=None):
    base=relative_features(s,pitch,mask);c=collective_features(s,pitch);motion=c[:,DYNAMIC];short=rolling(motion,12)
    return np.column_stack([base,c,rolling(motion,50),shifted(short,25)-shifted(short,-25)]).astype(np.float32)

def swap_joint(x):
    x=np.asarray(x)
    if x.ndim!=2 or x.shape[1]!=508:raise ValueError('Expected508 joint-control features')
    c=np.r_[np.arange(26,52),np.arange(26)];m=np.r_[np.arange(10,20),np.arange(10)]
    return np.column_stack([swap_relative(x[:,:416]),x[:,416:468][:,c],x[:,468:488][:,m],x[:,488:508][:,m]])
