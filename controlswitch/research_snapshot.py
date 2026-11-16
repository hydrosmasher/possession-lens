"""Read-only reconstruction of verified external research checkpoints."""
from pathlib import Path
import json,numpy as np
from .narrow_boundary import bundle040
from .gated_rnn import load_checkpoint
from .contact_training import infer_models
from .robust_dead_state import robust_dead_features,blend_dead
from .deadball import observation_features
from .boundary_had import boundary_observations,adaptive_decode
from .model import probabilities
from . import tree_model

def best_descriptor(run_root):
    root=Path(run_root).resolve()
    for source in ('robust-dead-059','spell-overlap-058','contact-057','short-spell-053'):
        p=root/source/'report.json'
        if not p.exists():continue
        report=json.loads(p.read_text())
        if not report['recommended_for_freeze']:continue
        v=report['selected'];d={'source':source,'name':v['name'],'scores':v['scores'],'mean_spell_f1':v['scores']['mean_spell_f1']}
        if source=='robust-dead-059':d.update(control_source=report['initial_source'],control_name=report['initial_name'],dead_features=v['features'],dead_mix=v['mix'])
        else:d.update(control_source=source,control_name=v['name'])
        return d
    raise ValueError('No verified external snapshot')

def make_snapshot(descriptor,fi,roots,run_root):
    root=Path(run_root).resolve();base,_=bundle040(fi,roots,root/'joint-040');p=(root/descriptor['control_source']/f'fold-{fi}'/(descriptor['control_name']+'.npz')).resolve()
    if not p.is_relative_to(root):raise ValueError('Escaping control checkpoint')
    b={'descriptor':descriptor,'base':base,'gru':load_checkpoint(p),'mix':.5}
    if descriptor['source']=='robust-dead-059':b['dead_refinement']=tree_model.load(root/'robust-dead-059'/f'fold-{fi}'/f"dead{descriptor['dead_features']}.npz")
    return b

def infer_snapshot(bundle,s,pitch,mask=None):
    predictions,emissions=infer_models(bundle['base'],{'reference':bundle['gru']},s,pitch,mask);q=emissions['reference']
    if 'dead_refinement' not in bundle:return predictions['reference'],q
    d=bundle['descriptor'];x=robust_dead_features(s,pitch,mask) if d['dead_features']==487 else observation_features(s,pitch,mask);q=blend_dead(q,tree_model.predict(bundle['dead_refinement'],x)[:,1],d['dead_mix']);e=probabilities(bundle['base']['models']['boundary'],boundary_observations(s,pitch,q,mask))[:,1];return adaptive_decode(q,e,1.),q

def cached_case(descriptor,fi,index,run_root):
    root=Path(run_root);source=descriptor['source'];name=descriptor['name']
    with np.load(root/source/f'fold-{fi}/validation-{index}.npz',allow_pickle=False) as a:
        y=a['reference'].copy();p=a[name].copy()
        if source=='robust-dead-059':q=blend_dead(a['reference_emissions'],a['dead'+str(descriptor['dead_features'])],descriptor['dead_mix'])
        elif source!='short-spell-053':q=a['emissions_'+name].copy()
        else:q=None
    if q is None:
        with np.load(root/f'precision-055-cache/fold-{fi}/validation-{index}.npz',allow_pickle=False) as a:q=a['emissions'].copy();np.testing.assert_array_equal(a['current'],p);np.testing.assert_array_equal(a['reference'],y)
    return y,p,q
