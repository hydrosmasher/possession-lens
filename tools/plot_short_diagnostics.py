"""Plot saved external duration diagnostics; no raw match data read."""
from pathlib import Path
import os,json
r=Path(__file__).resolve().parents[1];os.environ['MPLCONFIGDIR']=str(r/'local/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
report=json.loads((r/'reports/external/short-spell-053-diagnostic.json').read_text());data=report['duration_buckets'];buckets=['<1s','1-3s','3-10s','10s+'];fig,ax=plt.subplots(figsize=(8,4.4),layout='constrained')
for key,label,color in [('clean','Clean','#165ac7'),('masked','Short ball gaps','#9d3464'),('long_gap','Long ball gaps','#c18c16'),('tracker_noise','Tracking noise','#4e6b69')]:
 vals=[data[key][b]['matched']/data[key][b]['total'] for b in buckets];ax.plot(buckets,vals,marker='o',label=label,color=color)
ax.set_ylim(0,1);ax.set_ylabel('Fraction of reference spells matched (IoU ≥ 0.5)');ax.set_xlabel('Reference spell duration');ax.set_title('Candidate052: short possessions are the main recall gap');ax.grid(axis='y',alpha=.2);ax.legend(frameon=False);fig.supxlabel('Reused external development matches; not headline mean spell F1 or competition results.',fontsize=8);fig.savefig(r/'reports/external/short-spell-053-diagnostic.png',dpi=160);plt.close(fig)
