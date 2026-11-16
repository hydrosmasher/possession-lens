"""Observation availability flags, explicitly not calibrated model confidence."""
import numpy as np


def quality_report(sequence, blocks=None, fps=25):
    ball=np.asarray(sequence['ball'],dtype=float);xy=np.asarray(sequence['xy'],dtype=float);teams=np.asarray(sequence['teams']);n=len(ball)
    if ball.shape!=(n,2) or xy.ndim!=3 or xy.shape[0]!=n or xy.shape[2]!=2 or teams.shape!=(xy.shape[1],) or not np.isin(teams,[0,1]).all() or not all((teams==t).any() for t in (0,1)) or not np.isfinite(fps) or fps<=0:
        raise ValueError('Invalid observation-quality input')
    blocks=[(0,n)] if blocks is None else list(blocks);last=0;periods=[];intervals=[];totals={k:0 for k in ('analysed_frames','ball_observed_frames','ball_missing_frames','long_gap_frames','limited_player_frames','implausible_ball_step_frames','flagged_frames')}
    for period,(a,b) in enumerate(blocks,1):
        if not isinstance(a,(int,np.integer)) or not isinstance(b,(int,np.integer)) or not 0<=last<=a<b<=n:raise ValueError('Invalid observation-quality blocks')
        last=b;seen=np.isfinite(ball[a:b]).all(1);missing=~seen;valid=np.isfinite(xy[a:b]).all(2);limited=np.minimum(*[valid[:,teams==t].sum(1) for t in (0,1)])<3;long=np.zeros(b-a,bool)
        edges=np.flatnonzero(np.diff(np.r_[False,missing,False]))
        for lo,hi in edges.reshape(-1,2):
            if hi-lo>=2*fps:long[lo:hi]=True
        jump=np.zeros(b-a,bool)
        if b-a>1:jump[1:]=seen[1:]&seen[:-1]&(np.linalg.norm(np.diff(ball[a:b],axis=0),axis=1)*fps>60)
        flags=missing.astype(np.uint8)+2*long+4*limited+8*jump
        counts={'analysed_frames':b-a,'ball_observed_frames':int(seen.sum()),'ball_missing_frames':int(missing.sum()),'long_gap_frames':int(long.sum()),'limited_player_frames':int(limited.sum()),'implausible_ball_step_frames':int(jump.sum()),'flagged_frames':int((flags>0).sum())}
        for k,v in counts.items():totals[k]+=v
        periods.append({'period':period,'start_frame':int(a),'end_frame_exclusive':int(b),**counts})
        cuts=np.r_[0,np.flatnonzero(flags[1:]!=flags[:-1])+1,len(flags)]
        for lo,hi in zip(cuts[:-1],cuts[1:]):
            code=int(flags[lo])
            if not code:continue
            reasons=[name for bit,name in ((1,'ball_missing'),(2,'ball_gap_at_least_2s'),(4,'fewer_than_3_visible_players_on_a_team'),(8,'ball_step_above_60m_per_s')) if code&bit]
            intervals.append({'period':period,'start_frame':int(a+lo),'end_frame_exclusive':int(a+hi),'duration_seconds':float((hi-lo)/fps),'reasons':reasons})
    total=totals['analysed_frames']
    return {'version':'observation-quality-v1','meaning':'Availability and plausibility flags only; not calibrated prediction confidence, possession correctness or verified events. A flag does not change possession labels.','policy':{'long_ball_gap_seconds':2,'minimum_visible_players_per_team':3,'maximum_ball_step_m_per_s':60,'half_boundaries':'Reset gap and velocity checks within each supplied block; outside-block frames excluded.'},'fps':float(fps),'summary':{**totals,'ball_observed_percent':100*totals['ball_observed_frames']/max(total,1),'flagged_percent':100*totals['flagged_frames']/max(total,1)},'periods':periods,'flagged_intervals':intervals,'fitting_performed':False,'possession_labels_used':False}
