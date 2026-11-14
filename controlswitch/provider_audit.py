"""034: frozen031/032 compatibility on reused partial SkillCorner supervision."""
from pathlib import Path
import argparse,json
import numpy as np
from .skillcorner_europe import load_match
from .ranked_pipeline import load_pipeline,infer
from .tracker_robustness import CONDITIONS,condition
from .control_state import evaluate_accuracy

def run(raw,out):
    out=Path(out)
    if out.exists():raise ValueError('Use a new directory')
    out.mkdir(parents=True)
    protocol={'version':'provider-compatibility-034','match':'PSG–Marseille2269, previously reused validation, NOT fresh','source':'Existing historical MIT SkillCorner release; no download or challenge data used','models':['frozen031','frozen032'],'purpose':'Different-provider compatibility diagnostic only; no tuning or re-selection from this audit','supervision':'Partial native team-possession groups; unknown staysX. Resampled tracking/labels are not independent frames. No verified dead-ball reference available. Valid-run censoring changes spell objects; scores cannot be equated to full Sportec H/A/D results.','conditions':list(CONDITIONS),'training_or_calibration':False,'competition_percentile_known':False};(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    parts,meta=load_match(raw,'2269');bundles={'031':load_pipeline('reports/external/boundary-031-config.json'),'032':load_pipeline('reports/external/spell-training-032-config.json')};rows=[];preds={k:[] for k in bundles}
    for half,(seq,pitch) in enumerate(parts):
        if not np.isin(seq['y'],[0,1,3]).all():raise ValueError('Unexpected reference label; require partial H/A/X only')
        for ci,c in enumerate(CONDITIONS):
            print('Provider audit',half,c,flush=True);s,mask=condition(seq,c,half,False);rows.append({'y':seq['y'],'condition':c})
            for name,b in bundles.items():preds[name].append(infer(b,s,pitch,mask)[0])
            np.savez_compressed(out/f'half-{half}-{c}.npz',reference=seq['y'],**{name:p[-1] for name,p in preds.items()})
    # Existing accuracy aggregator assumes all labels known. Report masked frame
    # accuracy separately, while its official aggregate excludes unknown labels.
    scores={k:evaluate_accuracy(rows,p) for k,p in preds.items()}
    for k,p in preds.items():
        for c in CONDITIONS:
            ix=[i for i,x in enumerate(rows) if x['condition']==c];n=sum(int((rows[i]['y']<2).sum()) for i in ix);correct=sum(int(((rows[i]['y']==p[i])&(rows[i]['y']<2)).sum()) for i in ix);false_dead=sum(int(((p[i]==2)&(rows[i]['y']<2)).sum()) for i in ix);scores[k][c]['frame_accuracy']=correct/max(n,1);scores[k][c]['known_team_label_frames']=n;scores[k][c]['dead_predictions_on_known_team_frames']=false_dead;scores[k][c]['dead_f1']=None
    report={'protocol':protocol,'scores':scores,'match_metadata':meta,'independent_test':False,'deployment_decision_from_this_report':False};(out/'report.json').write_text(json.dumps(report,indent=2,default=str,allow_nan=False)+'\n');print('PARTIAL-PROVIDER031',scores['031']['mean_spell_f1'],flush=True);print('PARTIAL-PROVIDER032',scores['032']['mean_spell_f1'],flush=True);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--raw',required=True,type=Path);p.add_argument('--out',required=True,type=Path);a=p.parse_args();run(a.raw,a.out)
if __name__=='__main__':main()
