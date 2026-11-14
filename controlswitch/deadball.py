"""External-only factorised live/dead and team-control experiment (014)."""
from pathlib import Path
import argparse,json,platform
import numpy as np
from .pipeline import digest,pm
from .external import period_arrays
from .experiment import missing_mask,baseline_prediction,aggregate
from .model import features,decode
from .rnn_runtime import load_frozen,infer_frozen
from . import tree_model

VERSION='factorised-deadball-014'
TRAIN='J03WMX'
VALIDATION='J03WN1'

def observation_features(sequence,pitch,mask=None):
    # Exact invariance to home/away permutation; retains closest and furthest
    # team geometry via sum and absolute contrast, without names or labels.
    a=features(sequence,pitch,ball_missing=mask)
    b=features(sequence,pitch,swap=True,ball_missing=mask)
    return np.column_stack([(a+b)*.5,np.abs(a-b)]).astype(np.float32)

def fuse(team,dead,weight=1.):
    team=np.asarray(team,float);dead=np.asarray(dead,float)
    if team.ndim!=2 or team.shape[1]!=3 or dead.shape!=(len(team),):raise ValueError('Probability shape mismatch')
    if not np.isfinite(team).all() or not np.isfinite(dead).all() or (team<0).any() or ((dead<0)|(dead>1)).any() or not np.isfinite(weight) or weight<=0:raise ValueError('Invalid probabilities or weight')
    live=team[:,:2]/np.maximum(team[:,:2].sum(axis=1,keepdims=True),1e-12)
    live=np.where(team[:,:2].sum(axis=1,keepdims=True)>0,live,.5)
    q=np.clip(dead,1e-6,1-1e-6);q=1/(1+np.exp(-weight*np.log(q/(1-q))))
    return np.column_stack([live*(1-q[:,None]),q])

def load_candidate(run_dir,team_checkpoint,dead_checkpoint=None):
    run_dir=Path(run_dir)
    settings=json.loads((run_dir if run_dir.is_file() else run_dir/'settings.json').read_text())
    dead_checkpoint=Path(dead_checkpoint) if dead_checkpoint is not None else run_dir/'dead-estimator.npz'
    if settings['version']!=VERSION or digest(team_checkpoint)!=settings['team_checkpoint_sha256'] or digest(dead_checkpoint)!=settings['dead_checkpoint_sha256']:
        raise ValueError('Candidate asset/version mismatch')
    return {'settings':settings,'team':load_frozen(team_checkpoint),'dead':tree_model.load(dead_checkpoint)}

def infer_candidate(candidate,sequence,pitch,mask=None):
    """Observation-only inference; no scikit-learn or reference labels required."""
    _,team=infer_frozen(candidate['team'],sequence,pitch,mask)
    q=tree_model.predict(candidate['dead'],observation_features(sequence,pitch,mask))[:,1]
    settings=candidate['settings'];p=fuse(team,q,settings['dead_weight'])
    return decode(p,settings['penalty'],stop_penalty=settings['stop_penalty']),p

def metrics(rows,preds):
    result={}
    for condition in ('clean','masked'):
        pairs=[(s['y'],p) for (s,pitch,c,mask),p in zip(rows,preds) if c==condition]
        score=aggregate(pairs)
        y=np.concatenate([a for a,b in pairs]);p=np.concatenate([b for a,b in pairs])
        valid=y<3;cm=np.zeros((3,3),int);np.add.at(cm,(y[valid],p[valid]),1)
        score['per_class_f1']={name:float(2*cm[k,k]/max(cm[k].sum()+cm[:,k].sum(),1)) for k,name in enumerate('HAD')}
        score['dead_precision']=float(cm[2,2]/max(cm[:,2].sum(),1));score['dead_recall']=float(cm[2,2]/max(cm[2].sum(),1))
        result[condition]=score
    result['selection_spell_f1']=float(np.mean([result[c]['spell_f1_mean'] for c in ('clean','masked')]))
    return result

def run(cache_root,out,team_checkpoint,production_checkpoint,train_id=TRAIN,validation_id=VALIDATION,fixed_settings=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import sklearn
    from threadpoolctl import threadpool_limits
    if {train_id,validation_id}!={TRAIN,VALIDATION}:raise ValueError('Only audited distinct Bundesliga matches supported')
    out=Path(out)
    if out.exists():raise ValueError('Choose a new output directory')
    out.mkdir(parents=True)
    # Protocol saved before data access or fitting. No competition inputs accepted.
    protocol={'version':VERSION,'seed':114,'training_match':train_id,'validation_match':validation_id,'league':'Bundesliga (DFL competition 000001)','labels':'provider BallStatus: 0 dead, 1 live; live ownership BallPossession 1/2; unknown excluded','label_source':'Bassek et al. 2025 IDSSE/Sportec, CC BY 4.0, doi:10.1038/s41597-025-04505-y','history':'Both matches previously viewed; roles explicitly repurposed. No independent test.','training_sample_stride':5,'augmentation':'clean and independently masked, each half; masks seed 114+i; no identity features','validation_masks':'40% target contiguous ball gaps, seeds 214+i, tracking only','grid':{'team_penalty':[12,24],'stop_penalty':[3,12],'dead_weight':[.75,1.]},'selection':'mean clean/masked official spellF1; first maximum; no production promotion','dead_estimator':{'max_iter':100,'learning_rate':.08,'max_leaf_nodes':15,'min_samples_leaf':40,'l2_regularization':10,'early_stopping':False},'competition_data_used':False,'team_checkpoint_sha256':digest(team_checkpoint),'production_checkpoint_sha256':digest(production_checkpoint)}
    if fixed_settings is not None:
        protocol['fixed_settings']=fixed_settings
        protocol['selection']='Frozen settings from experiment014; no selection on reversed match'
        protocol['grid']={k:[fixed_settings[k]] for k in ('penalty','stop_penalty','dead_weight')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    records={};audit=[]
    for mid in (train_id,validation_id):
        paths=list(Path(cache_root).glob(mid+'-*.npz'))
        if len(paths)!=1:raise ValueError('Expected one original cache per match')
        with np.load(paths[0],allow_pickle=False) as d:records[mid]={k:d[k] for k in d.files}
        if str(records[mid]['match_id'])!=mid:raise ValueError('Match identity mismatch')
        for i in (0,1):
            s=period_arrays(records[mid],i);y=s['y']
            if not np.isin(y,[0,1,2,3]).all():raise ValueError('Invalid reference labels')
            audit.append({'match_id':mid,'half':i+1,'cache_sha256':digest(paths[0]),'source_xml_sha256':str(records[mid]['source_sha256']),'counts':{name:int((y==k).sum()) for k,name in enumerate('HADX')},'ball_observed_fraction':float(np.isfinite(s['ball']).all(1).mean())})
    (out/'label-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    xs=[];ys=[]
    for i in (0,1):
        s=period_arrays(records[train_id],i)
        for masked in (False,True):
            x=observation_features(s,records[train_id]['pitch'],missing_mask(len(s['y']),114+i) if masked else None)
            ix=np.flatnonzero(s['y']<3)[::5];xs.append(x[ix]);ys.append((s['y'][ix]==2).astype(int))
    x=np.concatenate(xs);y=np.concatenate(ys)
    print('Training dead-ball classifier',x.shape,np.bincount(y).tolist(),flush=True)
    model=HistGradientBoostingClassifier(**protocol['dead_estimator'],random_state=114)
    with threadpool_limits(limits=1):model.fit(x,y)
    tree_model.save(out/'dead-estimator.npz',model,'hist_boost');exported=tree_model.load(out/'dead-estimator.npz')
    if not np.allclose(model.predict_proba(x[:10000]),tree_model.predict(exported,x[:10000])[:,:2],atol=1e-9):raise AssertionError('Export mismatch')
    rows=[];teams=[];deads=[];productions=[];context=[];backbone=load_frozen(team_checkpoint);prod=load_frozen(production_checkpoint)
    for i in (0,1):
        s=period_arrays(records[validation_id],i);pitch=records[validation_id]['pitch']
        for c in ('clean','masked'):
            mask=missing_mask(len(s['y']),214+i) if c=='masked' else None
            print('Inference',i+1,c,flush=True)
            rows.append((s,pitch,c,mask));labels,p=infer_frozen(backbone,s,pitch,mask);context.append(labels);teams.append(p)
            z=observation_features(s,pitch,mask)
            with threadpool_limits(limits=1):q=model.predict_proba(z)[:,1]
            if not np.allclose(q,tree_model.predict(exported,z)[:,1],atol=1e-9):raise AssertionError('Validation export mismatch')
            deads.append(q);productions.append(infer_frozen(prod,s,pitch,mask)[0])
    comparison={'baseline_smooth50':metrics(rows,[baseline_prediction(s,50,mask) for s,_,_,mask in rows]),'production_rnn':metrics(rows,productions),'context_rnn':metrics(rows,context)}
    trials=[];best=None;best_preds=None
    for penalty in ([fixed_settings['penalty']] if fixed_settings else (12,24)):
        for stop in ([fixed_settings['stop_penalty']] if fixed_settings else (3,12)):
            for weight in ([fixed_settings['dead_weight']] if fixed_settings else (.75,1.)):
                preds=[decode(fuse(p,q,weight),penalty,stop_penalty=stop) for p,q in zip(teams,deads)]
                t={'penalty':penalty,'stop_penalty':stop,'dead_weight':weight,**metrics(rows,preds)};trials.append(t)
                print('Trial',penalty,stop,weight,t['selection_spell_f1'],flush=True)
                if best is None or t['selection_spell_f1']>best['selection_spell_f1']:best=t;best_preds=preds
    comparison['factorised_selected']=best
    for j,((s,pitch,c,mask),pred) in enumerate(zip(rows,best_preds)):
        np.savez_compressed(out/f'validation-{j}.npz',reference=s['y'],predicted=pred,dead_probability=deads[j])
    report={'protocol':protocol,'label_audit':audit,'training_samples':len(y),'training_classes':np.bincount(y).tolist(),'models':comparison,'selected':{k:best[k] for k in ('penalty','stop_penalty','dead_weight')},'export_matches_native':True,'model_sha256':digest(out/'dead-estimator.npz'),'versions':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'fresh_test_results':None,'promoted_to_default':False,'limitations':['Single training and reused validation match; selection bias and club overlap remain.','Provider states are not independent human annotations; tracking quality differs from challenge.','Legacy production comparator has second-division training ancestry; new dead estimator is first-division only.','Context backbone is frozen separately trained SkillCorner model; no competition fitting.','Ball masking is simulated and does not reproduce all tracker errors.','No smart tags used; no suitable matched external tags audited.','Offline context and Viterbi use future tracking.']}
    (out/'trials.json').write_text(json.dumps(trials,indent=2)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    settings={'version':VERSION,'team_checkpoint_sha256':digest(team_checkpoint),'dead_checkpoint_sha256':digest(out/'dead-estimator.npz'),**report['selected']}
    (out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache-root',required=True,type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--team-model',required=True,type=Path);p.add_argument('--production-model',required=True,type=Path);p.add_argument('--reverse-check',action='store_true',help='Reverse the two reused matches, with experiment014 decoder settings frozen');a=p.parse_args()
    kwargs={}
    if a.reverse_check:
        saved=json.loads((Path(__file__).resolve().parents[1]/'reports/external/deadball-014-settings.json').read_text())
        kwargs={'train_id':VALIDATION,'validation_id':TRAIN,'fixed_settings':{k:saved[k] for k in ('penalty','stop_penalty','dead_weight')}}
    r=run(a.cache_root,a.out,a.team_model,a.production_model,**kwargs)
    print(json.dumps(r['models'],indent=2))
if __name__=='__main__':main()
