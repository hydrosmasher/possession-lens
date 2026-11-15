"""055: predeclared precision check on frozen053 emissions; no retraining."""
from pathlib import Path
import json,numpy as np,torch
from .boundary_had import adaptive_decode
from .boundary_alignment import eligible
from .control_state import evaluate_accuracy
from .pipeline import digest
from .recurrent_boundary import evidence
from .model import probabilities
from .external import period_arrays
from .tracker_robustness import CONDITIONS,noisy_tracking
from .experiment import missing_mask
from .gap_possession import long_mask
from .narrow_boundary import bundle040
from .gated_rnn import load_checkpoint

VARIANTS=[{'name':'penalty15','penalty':15.},{'name':'penalty18','penalty':18.}]
def run(root):
    root=Path(root);decision=json.loads((root/'reports/external/morning-053-054-decision.json').read_text())
    if decision['candidate']!='053' or decision.get('branch')!='short_focus':raise ValueError('Only run if audited053 is selected by the original comparison')
    out=root/'local/precision-055'
    if out.exists():raise ValueError('Choose new precision run')
    protocol=json.loads((root/'reports/external/precision-055-protocol.json').read_text());out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');folds=[];all_rows=[];all_base=[];all_preds={v['name']:[] for v in VARIANTS}
    for fi in (0,1):
        rows=[];base=[];preds={v['name']:[] for v in VARIANTS}
        for i in range(8):
            with np.load(root/'local/precision-055-cache'/f'fold-{fi}/validation-{i}.npz',allow_pickle=False) as a:y,p,q,event=(a[k] for k in ('reference','current','emissions','event'))
            np.testing.assert_array_equal(adaptive_decode(q,event,1.),p);rows.append({'y':y,'condition':CONDITIONS[i%4]});base.append(p)
            for v in VARIANTS:preds[v['name']].append(adaptive_decode(q,event,1.,penalty=v['penalty']))
            np.savez_compressed(out/f'validation-{fi}-{i}.npz',reference=y,current=p,**{k:v[-1] for k,v in preds.items()})
        fold={'current':evaluate_accuracy(rows,base),'candidates':{k:evaluate_accuracy(rows,p,base) for k,p in preds.items()}};folds.append(fold);all_rows+=rows;all_base+=base
        for k in preds:all_preds[k]+=preds[k]
    current=evaluate_accuracy(all_rows,all_base);np.testing.assert_allclose(current['mean_spell_f1'],decision['mean_spell_f1'],rtol=0,atol=1e-14);trials=[]
    for v in VARIANTS:
        score=evaluate_accuracy(all_rows,all_preds[v['name']],all_base);guards=eligible(folds,v['name'],current,score);guards['false_positive_no_increase']=score['totals']['false_positive']<=current['totals']['false_positive'];trials.append({**v,'scores':score,'guards':guards,'eligible':all(guards.values())})
    choices=[t for t in trials if t['eligible']];selected=max(choices,key=lambda t:t['scores']['mean_spell_f1']) if choices else None;(out/'freeze.json').write_text(json.dumps({'selected':selected,'reference_config_sha256':digest(root/decision['config']),'before_new_draw_audit':True},indent=2)+'\n');print('PRECISION FREEZE',selected,flush=True);audit=None
    if selected:
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);roots=json.loads((root/'local/roots.json').read_text());rows=[];base=[];preds=[]
        for fi,mid in enumerate(('J03WN1','J03WMX')):
            component,_=bundle040(fi,roots,root/'local/joint-040');bundle={'base':component,'gru':load_checkpoint(root/f'local/short-spell-053/fold-{fi}/short_focus.npz'),'mix':.5}
            with np.load(next(Path('/Users/hydrosmasher/Desktop/Possession Lens/data/external/cache').glob(mid+'-*.npz'))) as a:d={k:a[k] for k in a.files}
            for half in (0,1):
                seq=period_arrays(d,half)
                for ci,c in enumerate(CONDITIONS):
                    print('Precision frozen audit',fi,half,c,flush=True);s={k:v for k,v in seq.items() if k!='y'};mask=None
                    if c=='masked':mask=missing_mask(len(seq['y']),6514+half)
                    elif c=='long_gap':mask=long_mask(len(seq['y']),6714+half)
                    elif c=='tracker_noise':s=noisy_tracking(seq,7118+half)
                    q,x,e=evidence(bundle,s,d['pitch'],mask);event=probabilities(component['models']['boundary'],x)[:,1];old=adaptive_decode(q,event,1.);new=adaptive_decode(q,event,1.,penalty=selected['penalty']);rows.append({'y':seq['y'],'condition':c});base.append(old);preds.append(new);np.savez_compressed(out/f'audit-{fi}-{half*4+ci}.npz',current=old,candidate=new)
        a=evaluate_accuracy(rows,base);b=evaluate_accuracy(rows,preds,base);audit={'current':a,'candidate':b,'mean_improves':b['mean_spell_f1']>a['mean_spell_f1'],'false_positive_no_increase':b['totals']['false_positive']<=a['totals']['false_positive'],'same_matches_new_noise_only':True}
    report={'protocol':protocol,'folds':folds,'current':current,'trials':trials,'selected':selected,'new_draw_audit':audit,'recommended_for_freeze':bool(selected and audit['mean_improves'] and audit['false_positive_no_increase']),'promoted_to_default':False,'competition_percentile_known':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('PRECISION',t['name'],t['scores']['mean_spell_f1'],t['guards'],flush=True)
    if audit:print('PRECISION AUDIT',audit['current']['mean_spell_f1'],audit['candidate']['mean_spell_f1'],audit['false_positive_no_increase'],flush=True)
if __name__=='__main__':run(Path.cwd())
