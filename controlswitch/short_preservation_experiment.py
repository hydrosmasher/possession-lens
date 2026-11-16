"""084: protect existing short predictions during fine-tuned GRU decoding."""
from pathlib import Path
import argparse,json
import numpy as np
import torch
from .short_preservation import preservation_anchors
from .continuity_refinement import residual_live, anchored_decode
from .continuity_refinement_experiment import conditional, BRANCHES
from .gated_rnn import load_checkpoint
from .joint_state import joint_features, swap_joint
from .receipt_context_experiment import record_for, short_flags
from .research_snapshot import cached_case
from .external import period_arrays
from .tracker_robustness import CONDITIONS, condition
from .proposal_calibration import retention, flags
from .local_temporal_experiment import short_matches
from .control_state import evaluate_accuracy
from .pipeline import digest


def run(root, cache):
    root,cache=Path(root).resolve(),Path(cache).resolve();out=root/'local/short-preservation-084';out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    previous=json.loads((root/'local/continuity-refinement-083/report.json').read_text())
    protocol={'version':'short-preservation-084','motivation':'083recoversall46newspellsbutloses6othermatchedshortspells. Protectexistingpredictedshortspansandflanksduringlong-contextrefinement,withoutreferenceinput.',
              'data':previous['protocol']['data'],'training':'Noadditionaltraining;frozen083replay_ce/replay_continuitycheckpoints. No newmatch or augmentationdownload.',
              'anchors':'081predictionsonly:protecteverypredictedHAspell<=150framesPLUSoneframeflankateachend;Dalreadylocked.150=2*75fromIoU>=.5necessarypredictedlengthboundfor<=75framereferencesinfullyscoredruns. Notchosenfromheldouteventlengths. Alsofreeze053missinglabelsand081newreceiptspans.',
              'inference':'Same083arithmeticGRUresidual,frozen053Dmask/cached031costs,constrainedViterbi. Newshort-protectionanchorsonlychangevs083. Bothtrainedbranchesandmatchedsame-anchor-no-trainingcontrolreported.',
              'selection':previous['protocol']['selection'],'audit':previous['protocol']['audit'],
              'limits':'Adaptivefollow-upafter083ontwoheavilyreusedfull-HADgames. Longunscoredreferenceboundariescaninvalidateglobal-lengthargument;exactretentionalwayschecked. Protectionalsofreezesfalsepredictedshortspellsandmaylimitprecisiongains. No independenttestclaim.',
              'source_sha256':{n:digest(root/'controlswitch'/n) for n in ('short_preservation.py','short_preservation_experiment.py','continuity_refinement.py')},
              'trained_model_sha256':{f'{fi}-{name}':digest(root/f'local/continuity-refinement-083/fold-{fi}/{name}.npz') for fi in (0,1) for name in BRANCHES},
              'input_report_sha256':digest(root/'local/continuity-refinement-083/report.json')}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    names=('short_anchors_control',*BRANCHES);allrows=[];allbase=[];allstart=[];allpred={k:[] for k in names};folds=[]
    for fi,mid in enumerate(('J03WN1','J03WMX')):
        fd=out/f'fold-{fi}';fd.mkdir();record=record_for(cache,mid)
        initial=load_checkpoint(root/f'local/short-spell-053/fold-{fi}/short_focus.npz')
        models={k:load_checkpoint(root/f'local/continuity-refinement-083/fold-{fi}/{k}.npz') for k in BRANCHES}
        rows=[];base=[];start=[];pred={k:[] for k in names};reviews=[]
        losses={k:{'short_spells_lost_vs_070':0,'additional_053_spells_lost_vs_070':0,'prototype_recovered':0,'retained_recovered':0,'lost_recovered':0} for k in names}
        for half in (0,1):
          seq=period_arrays(record,half)
          for ci,kind in enumerate(CONDITIONS):
            print('Short-preservation evaluation',fi,half,kind,flush=True);s,mask=condition(seq,kind,half,False)
            y,current,q=cached_case({'source':'short-spell-053','name':'short_focus'},fi,half*4+ci,root/'local');np.testing.assert_array_equal(y,seq['y'])
            with np.load(root/f'local/precision-055-cache/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:event=a['event'].copy()
            with np.load(root/f'local/gap-fragment-081/fold-{fi}/validation-{half*4+ci}.npz',allow_pickle=False) as a:prototype,candidate=a['prototype'].copy(),a['gap_fragment'].copy()
            seen=np.isfinite(s['ball']).all(1)
            if mask is not None:seen&=~mask
            anchors,protected=preservation_anchors(current,candidate,seen)
            h=q[:,:2]/q[:,:2].sum(1,keepdims=True);x=joint_features(s,record['pitch'],mask);sx=swap_joint(x);old=conditional(initial,x,sx)
            values={'short_anchors_control':h}
            for name in BRANCHES:values[name]=residual_live(h,old,conditional(models[name],x,sx))
            case={}
            for name,live in values.items():
                p=anchored_decode(live,event,current,anchors)
                np.testing.assert_array_equal(p==2,current==2);np.testing.assert_array_equal(p[~seen],current[~seen]);np.testing.assert_array_equal(p[anchors>=0],anchors[anchors>=0])
                for e in protected:
                    a,b=e['start'],e['end'];np.testing.assert_array_equal(p[max(0,a-1):min(len(p),b+1)],candidate[max(0,a-1):min(len(p),b+1)])
                loss=retention(y,current,prototype,p);loss['short_spells_lost_vs_070']=int((short_flags(y,prototype)&~short_flags(y,p)).sum());loss['additional_053_spells_lost_vs_070']=int((flags(y,current)&flags(y,prototype)&~flags(y,p)).sum())
                for k,v in loss.items():losses[name][k]+=v
                pred[name].append(p);case[name]=p;reviews.append({'half':half+1,'condition':kind,'branch':name,'protected_prediction_spells':len(protected),'retention':loss})
            rows.append({'y':y,'condition':kind});base.append(current);start.append(candidate)
            np.savez_compressed(fd/f'validation-{half*4+ci}.npz',reference=y,current=current,starting_candidate=candidate,**case)
        f={'evaluate':mid,'reference':evaluate_accuracy(rows,base),'starting_candidate':evaluate_accuracy(rows,start,base),'variants':{k:evaluate_accuracy(rows,p,base) for k,p in pred.items()},'retention':losses,'reviews':reviews}
        folds.append(f);(fd/'report.json').write_text(json.dumps(f,indent=2)+'\n');allrows+=rows;allbase+=base;allstart+=start
        for k in names:allpred[k]+=pred[k]
    reference=evaluate_accuracy(allrows,allbase);starting=evaluate_accuracy(allrows,allstart,allbase);scores={k:evaluate_accuracy(allrows,p,allbase) for k,p in allpred.items()};trials=[]
    for name,p in allpred.items():
        score=scores[name];short=sum(short_matches(r['y'],v) for r,v in zip(allrows,p));lost={k:sum(f['retention'][name][k] for f in folds) for k in folds[0]['retention'][name]}
        guards={'spell_gain':score['mean_spell_f1']>=reference['mean_spell_f1']+.002,'neutral_control':score['mean_spell_f1']>=scores['short_anchors_control']['mean_spell_f1'],
                'fp_reduction':score['totals']['false_positive']<starting['totals']['false_positive'],'fp_vs_053':score['totals']['false_positive']<=reference['totals']['false_positive'],
                'misses':score['totals']['missed']<=reference['totals']['missed'],'short_gain':short>sum(short_matches(r['y'],v) for r,v in zip(allrows,allbase)),
                'same_recoveries':lost['lost_recovered']==0,'all_short_retained':lost['short_spells_lost_vs_070']==0,'existing_retained':lost['additional_053_spells_lost_vs_070']==0,
                'noise':all(f['variants'][name]['tracker_noise']['spell_f1_mean']>=f['reference']['tracker_noise']['spell_f1_mean'] for f in folds),
                'clean_gap':all(f['variants'][name][c]['spell_f1_mean']>=f['reference'][c]['spell_f1_mean']-.01 for f in folds for c in CONDITIONS[:3])}
        trials.append({'name':name,'scores':score,'short_matched':short,'retention':lost,'guards':guards,'eligible':name in BRANCHES and all(guards.values())})
    eligible=[t for t in trials if t['eligible']];report={'protocol':protocol,'folds':folds,'reference':reference,'starting_candidate':starting,'trials':trials,'selected':max(eligible,key=lambda t:t['scores']['mean_spell_f1']) if eligible else None,'promoted_to_default':False,'independent_test':False,'frozen_audit_required':bool(eligible)}
    (out/'freeze.json').write_text(json.dumps({'selected':report['selected'],'before_new_draw_audit':True},indent=2)+'\n');(out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    for t in trials:print('RESULT',t['name'],t['scores']['mean_spell_f1'],t['scores']['totals'],'short',t['short_matched'],t['retention'],t['guards'],flush=True)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path.cwd());p.add_argument('--cache-root',type=Path,required=True);a=p.parse_args();run(a.root,a.cache_root)
