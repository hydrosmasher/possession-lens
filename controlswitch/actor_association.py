"""061: exact provider-owner supervision for an identity-free actor scorer."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from torch.nn import functional as F
from .player_set_state import player_nodes,transform_nodes,NODES,FEATURES
from .player_set_rnn import PlayerSetPossession
from .skillcorner_europe import load_match
from .pipeline import digest
TRAIN_IDS=('4039','3749','3518','2068')
VALIDATION_IDS=('2440',)
BASE_CONFIG={'cell':'gru','layers':2,'bidirectional':True,'hidden':64,'projection':64,'dropout':.25,'boundary':True,'input_features':508}


def owner_targets(nodes,indices,owners):
    nodes=np.asarray(nodes);indices=np.asarray(indices);owners=np.asarray(owners)
    if nodes.shape!=(len(owners),2,NODES,FEATURES) or indices.shape!=nodes.shape[:-1]:raise ValueError('Invalid owner-node alignment')
    flat=nodes.reshape(len(nodes),2*NODES,FEATURES);matches=(indices.reshape(len(nodes),2*NODES)==owners[:,None])&(owners[:,None]>=0)&(flat[:,:,20]>.5)&(flat[:,:,10]>.5);count=matches.sum(1);target=np.full(len(nodes),-1,np.int64);good=count==1;target[good]=matches[good].argmax(1);return target


def actor_logits(model,nodes):
    if nodes.ndim!=4 or nodes.shape[1:]!=(2,NODES,FEATURES):raise ValueError('Invalid actor batch')
    valid=nodes[...,20]>.5;embedding=model.encoder(nodes[...,:20]);return model.attention(embedding).squeeze(-1).masked_fill(~valid,-1e4).flatten(1)


def augment_points(nodes,targets,codes):
    x=np.array(nodes,copy=True);y=np.array(targets,copy=True);codes=np.asarray(codes)
    if codes.shape!=(len(x),) or not np.isin(codes,np.arange(8)).all():raise ValueError('Invalid point augmentation')
    swap=(codes&1)>0;x[swap]=x[swap][:,::-1];y[swap]=(y[swap]+NODES)%(2*NODES);sign=np.ones((len(x),1,1,FEATURES),np.float32)
    for col in (0,2,4,11):sign[(codes&2)>0,0,0,col]=-1
    for col in (1,3,5,12):sign[(codes&4)>0,0,0,col]=-1
    return x*sign,y


def build(raw,out):
    raw=Path(raw);out=Path(out)
    if out.exists():raise ValueError('Use new actor corpus directory')
    out.mkdir(parents=True);protocol={'version':'provider-owner-corpus-061','source':'Existing historical MIT SkillCorner first-division training release; no downloads or challenge data.','training_games':list(TRAIN_IDS),'component_validation_games':list(VALIDATION_IDS),'excluded_games':['2269','2841','3442','2417'],'mapping':'Only exact provider possession.trackable_object in provider roster, unique current-frame player observation, visible ball, consistent provider team. No ID recovery, nearest-player relabelling or manual names. Unknown/unobserved/ambiguous labels excluded.','sampling':'Existing10Hz data resampled25Hz with oldloader. Sample every5th25Hz frame, exactly everysecond nativeframe, to avoid interpolated owner targets. Player nodes contain no IDs/labels; owner slot is a training target only.','history':'All five matches previously used elsewhere in project;2440 is held out from actor-scorer fitting only,not a fresh project-level holdout. No verified dead-ball labels.','features':'2teams x16nodes x21fixed physical features including validity. Labels aligned through explicit returned original slot indices.','deployment':'No candidate decision from corpus audit; all generated arrays are private and excluded fromGitHub.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');entries=[];audits=[]
    for mid in TRAIN_IDS+VALIDATION_IDS:
        print('Audit/load provider owner labels',mid,flush=True);parts,audit=load_match(raw,mid,recover_provider_ids=False,include_owner=True);kept=0;aligned=0
        for half,(seq,pitch) in enumerate(parts):
            ix=np.arange(0,len(seq['owner']),5);owners=seq['owner'][ix];nodes,indices=player_nodes(seq,pitch,return_indices=True);x=nodes[ix];target=owner_targets(x,indices[ix],owners);good=target>=0;role='train' if mid in TRAIN_IDS else 'component_validation';name=f'{mid}-h{half}';np.save(out/(name+'-points.npy'),x[good],allow_pickle=False);np.save(out/(name+'-targets.npy'),target[good],allow_pickle=False);entry={'match_id':mid,'half':half+1,'role':role,'points':name+'-points.npy','targets':name+'-targets.npy','aligned_5hz_frames':len(ix),'exact_owner_labels_at_aligned_frames':int((owners>=0).sum()),'retained_points':int(good.sum()),'shape':[int(good.sum()),2,NODES,FEATURES]};entries.append(entry);kept+=int(good.sum());aligned+=len(ix);print('Actor points',entry,flush=True);del nodes,indices,x
        audits.append({'match_id':mid,'role':'train' if mid in TRAIN_IDS else 'component_validation','periods':audit,'retained_points':kept,'aligned_frames':aligned,'source_sha256':{n:digest(raw/mid/n) for n in ('match_data.json','structured_data.json')}});del parts
        (out/'progress.json').write_text(json.dumps({'entries':entries,'audit':audits},indent=2)+'\n')
    result={'protocol':protocol,'entries':entries,'audit':audits,'complete':True,'train_points':sum(e['retained_points'] for e in entries if e['role']=='train'),'component_validation_points':sum(e['retained_points'] for e in entries if e['role']=='component_validation')};(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print('Actor corpus complete',result['train_points'],result['component_validation_points'],flush=True);return result


def point_metrics(model,x,y):
    model.eval();preds=[];probabilities=[]
    with torch.inference_mode():
        for a in range(0,len(x),512):
            p=actor_logits(model,torch.from_numpy(np.array(x[a:a+512]))).softmax(-1).numpy();probabilities.append(p);preds.append(p.argmax(1))
    p=np.concatenate(probabilities);pred=np.concatenate(preds);flat=x.reshape(len(x),2*NODES,FEATURES);distance=flat[:,:,8];valid=flat[:,:,20]>.5;nearest=np.where(valid,distance,np.inf).argmin(1);team=y//NODES;ambiguous=np.all(np.where(valid,5*distance,np.inf).reshape(len(x),2,NODES).min(2)<=np.log(3),axis=1);soft=np.exp(-10*distance)*valid;soft=soft/np.maximum(soft.sum(1,keepdims=True),1e-8);tprob=p.reshape(len(x),2,NODES).sum(2);nearprob=soft.reshape(len(x),2,NODES).sum(2);onehot=np.eye(2)[team]
    return {'points':len(y),'actor_top1':float(np.mean(pred==y)),'team_top1':float(np.mean(pred//NODES==team)),'nearest_actor_top1':float(np.mean(nearest==y)),'nearest_team_top1':float(np.mean(nearest//NODES==team)),'team_brier':float(np.mean(np.sum((tprob-onehot)**2,1))),'distance_prior_team_brier':float(np.mean(np.sum((nearprob-onehot)**2,1))),'ambiguous_both_teams_within2m_points':int(ambiguous.sum()),'ambiguous_team_top1':float(np.mean((pred//NODES==team)[ambiguous])) if ambiguous.any() else None,'ambiguous_nearest_team_top1':float(np.mean((nearest//NODES==team)[ambiguous])) if ambiguous.any() else None}


def fit(corpus,out):
    corpus=Path(corpus).resolve();out=Path(out)
    if out.exists():raise ValueError('Use new actor training directory')
    manifest=json.loads((corpus/'manifest.json').read_text())
    if not manifest['complete'] or manifest['train_points']<1000 or manifest['component_validation_points']<100:raise ValueError('Insufficient exact actor supervision')
    out.mkdir(parents=True);torch.set_num_threads(1);torch.use_deterministic_algorithms(True);protocol={'version':'provider-actor-scorer-061','corpus_manifest_sha256':digest(corpus/'manifest.json'),'training_ids':list(TRAIN_IDS),'component_validation_ids':list(VALIDATION_IDS),'training':'Shared20->32->32nodeMLP +sharedattention logits,global32actor softmax.10fixed epochs overall exact train points,batch512,AdamWlr.001/weightdecay.001,gradientclip1,seed161beforeinitialisation. Randomteam/x/yreflections perpoint; no RNN,projection or strong-match labels trained here. No early stopping/checkpoint selection.','validation':'2440 actor labels excluded from fitting; previously used elsewhere in project,notfresh whole-pipeline holdout. Report actor/team accuracy,ambiguity subset andBrier vsnearest/distance baseline. No verified dead-ball reference.','component_gate':'Fixedfinal weights considered for ledger experiment if actor_top1 improves by>=.01 withteam_top1 no worse thannearest-.005 ORteam_top1 improves>=.005 withactor no worse thannearest-.005; ORteamBrier improves>=.01 vsdistance-prior withactor/team top1 no worse thannearest-.005; ambiguous-team decline<=.01 if>=100points. This is NOT end-to-end possession acceptance.','history':'Provider-supplied weak owner identifiers, not human-verified touches. No guessed IDs/names or challenge data/video/ground truth/publicscore used.'};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');data={}
    for role in ('train','component_validation'):
        entries=[e for e in manifest['entries'] if e['role']==role];data[role]=(np.concatenate([np.load(corpus/e['points'],allow_pickle=False) for e in entries]),np.concatenate([np.load(corpus/e['targets'],allow_pickle=False) for e in entries]))
    torch.manual_seed(161);rng=np.random.default_rng(161);model=PlayerSetPossession(BASE_CONFIG.copy(),pooling='global');parameters=list(model.encoder.parameters())+list(model.attention.parameters());opt=torch.optim.AdamW(parameters,lr=.001,weight_decay=.001);x,y=data['train'];trace=[]
    for epoch in range(1,11):
        model.train();order=rng.permutation(len(x));losses=[]
        for a in range(0,len(order),512):
            ix=order[a:a+512];xx,yy=augment_points(x[ix],y[ix],rng.integers(8,size=len(ix)));opt.zero_grad(set_to_none=True);logits=actor_logits(model,torch.from_numpy(xx));target=torch.from_numpy(yy);valid=torch.from_numpy(xx[...,20].reshape(len(xx),2*NODES)>.5)
            if not valid[torch.arange(len(xx)),target].all():raise ValueError('Actor target hidden or missing')
            loss=F.cross_entropy(logits,target)
            if not torch.isfinite(loss):raise ValueError('Nonfinite actor loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);opt.step();losses.append(float(loss.detach()))
        row={'epoch':epoch,'mean_actor_ce':float(np.mean(losses))};trace.append(row);print('Actor training',row,flush=True);(out/'learning.json').write_text(json.dumps(trace,indent=2)+'\n')
    model.eval();metrics={k:point_metrics(model,*v) for k,v in data.items()};v=metrics['component_validation'];gate=((v['actor_top1']>=v['nearest_actor_top1']+.01 and v['team_top1']>=v['nearest_team_top1']-.005) or (v['team_top1']>=v['nearest_team_top1']+.005 and v['actor_top1']>=v['nearest_actor_top1']-.005));calibration=(v['team_brier']<=v['distance_prior_team_brier']-.01 and v['team_top1']>=v['nearest_team_top1']-.005 and v['actor_top1']>=v['nearest_actor_top1']-.005);gate=gate or calibration;ambiguous=(v['ambiguous_both_teams_within2m_points']<100 or v['ambiguous_team_top1']>=v['ambiguous_nearest_team_top1']-.01);meta={'version':'provider-actor-encoder-v1','nodes_per_team':NODES,'node_features':FEATURES,'node_width':32,'pooling':'global','training_ids':list(TRAIN_IDS),'component_validation_ids':list(VALIDATION_IDS)};weights={k:p.detach().numpy() for k,p in model.state_dict().items() if k.startswith(('encoder.','attention.'))};np.savez_compressed(out/'actor-encoder.npz',meta=np.array(json.dumps(meta)),**weights);report={'protocol':protocol,'trace':trace,'metrics':metrics,'component_gate_passed':bool(gate and ambiguous),'encoder_sha256':digest(out/'actor-encoder.npz'),'end_to_end_possession_validated':False,'independent_project_test':False,'promoted_to_default':False};(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('Actor component validation',v,flush=True);print('Component gate',report['component_gate_passed'],flush=True);return report

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['build','fit']);p.add_argument('--raw',type=Path);p.add_argument('--corpus',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args();build(a.raw,a.out) if a.action=='build' else fit(a.corpus,a.out)
