"""Plot aggregate external model evidence; no licensed raw data read."""
from pathlib import Path
import os,json
r=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(r/'local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=json.loads((r/'reports/external/progress-052.json').read_text());d=json.loads((r/'reports/external/morning-053-054-decision.json').read_text());names=p['candidates'][:];scores=p['mean_spell_f1'][:];inter=d.get('original_choice')
if inter and inter['candidate']!='052':names.append(inter['candidate']);scores.append(inter['mean_spell_f1'])
if d['candidate']!='052' and names[-1]!=d['candidate']:names.append(d['candidate']);scores.append(d['mean_spell_f1'])
old=json.loads((r/'reports/external/transfer-052-report.json').read_text())['selected']['scores'];new=d.get('scores',old);conditions=['clean','masked','long_gap','tracker_noise'];labels=['Clean','Short ball gaps','Long ball gaps','Tracking noise'];fig,axes=plt.subplots(1,2,figsize=(11,4.6),layout='constrained');ax=axes[0];ax.plot(names,scores,color='#165ac7',marker='o',lw=2);ax.axhspan(.85,.90,color='#a71948',alpha=.12,label='0.85–0.90 target');ax.axhline(.90,color='#a71948',ls='--',lw=1)
for i,v in enumerate(scores):ax.annotate(f'{v:.3f}',(i,v),xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
ax.set_ylim(.40,.94);ax.set_title('Comparable development mean');ax.set_ylabel('Official possession-spell F1');ax.set_xlabel('Checkpoint');ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.2)
ax=axes[1];x=list(range(4));ax.bar([i-.18 for i in x],[old[c]['spell_f1_mean'] for c in conditions],width=.36,label='052',color='#b7bec9');ax.bar([i+.18 for i in x],[new[c]['spell_f1_mean'] for c in conditions],width=.36,label=d['candidate'],color='#a71948');ax.set_xticks(x,labels,rotation=17);ax.set_ylim(0,1);ax.set_title('Conditions behind the mean');ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);fig.suptitle('Possession Lens · 15 November model session',fontsize=15,fontweight='bold');fig.supxlabel('Two heavily reused external matches. Simulated errors are not independent-match or competition tests.',fontsize=8);fig.savefig(r/'reports/external/progress-morning.png',dpi=170);plt.close(fig);(r/'reports/external/progress-morning.json').write_text(json.dumps({'candidates':names,'mean_spell_f1':scores,'independent_test':False,'competition_percentile_known':False,'target':[.85,.9]},indent=2)+'\n')
