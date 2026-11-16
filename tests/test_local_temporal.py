import unittest,tempfile
from pathlib import Path
import numpy as np,torch
from controlswitch.local_temporal import LocalTemporal,predict,save,load,locked_dead_decode
from controlswitch.boundary_had import adaptive_decode
class LocalTemporalTests(unittest.TestCase):
    def test_exact_decoder_replay_with_locked_dead_mask(self):
        rng=np.random.default_rng(69);p=rng.uniform(.001,1,(900,3));p/=p.sum(1,keepdims=True);event=rng.uniform(size=900);frozen=adaptive_decode(p,event,1);out=locked_dead_decode(p[:,:2],event,frozen);np.testing.assert_array_equal(out,frozen)
    def test_future_context_and_finite_receptive_field(self):
        torch.manual_seed(169);m=LocalTemporal().eval();x=torch.randn(1,180,508);a=m(x);near=x.clone();near[:,91]+=2;b=m(near);far=x.clone();far[:,150]+=2;c=m(far)
        assert (a[:,90]-b[:,90]).abs().max()>1e-6;torch.testing.assert_close(a[:,90],c[:,90])
    def test_windowed_inference_export_and_gradients(self):
        torch.manual_seed(169);torch.set_num_threads(1);m=LocalTemporal();x=torch.randn(2,100,508);y=m(x);torch.nn.functional.cross_entropy(y.reshape(-1,2),torch.zeros(200,dtype=torch.long)).backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        m.eval();rng=np.random.default_rng(69);xx=rng.normal(size=(700,508)).astype(np.float32);mean=np.zeros(508,np.float32);scale=np.ones(508,np.float32);a=predict(m,xx,mean,scale)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'model.npz';save(p,m,mean,scale);b=load(p);np.testing.assert_array_equal(a,predict(b['model'],xx,b['mean'],b['scale']))
        # Window context exceeds the 42frame convolution radius.
        with torch.inference_mode():full=m(torch.from_numpy(xx[None])).softmax(-1).numpy()[0]
        np.testing.assert_allclose(a[50:-50],full[50:-50],atol=2e-6,rtol=2e-6)
