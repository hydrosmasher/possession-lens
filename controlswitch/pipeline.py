"""Local-only pipeline. No match assets or extracts are sent to external services."""
from pathlib import Path
import hashlib
import json
import platform
import sys
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'vendor' / 'mtah'
sys.path.insert(0, str(UPSTREAM))
from games import GAMES, GAME, half_video_blocks
from starter.baseline import smooth, write_submission, VARIANTS
sys.path.insert(0, str(UPSTREAM / 'evaluation'))
import possession_metrics as pm

N_FRAMES = GAMES[GAME]['n_video_frames']
FPS = GAMES[GAME]['fps']
BLOCKS = [(a, b + 1) for _, a, b in half_video_blocks(GAME)]

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def discover(root):
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f'Dataset folder does not exist: {root}')
    files = {}
    for key, name in [('metadata', 'BAR-ATH_tactical_atd.xml'),
                      ('tracking', 'BAR-ATH_tactical_atd.txt'),
                      ('tagging', 'BAR-ATH_tactical_smart_tagging.xml'),
                      ('homography', 'BAR-ATH_tactical_homography.csv')]:
        matches = sorted(root.rglob(name))
        if len(matches) > 1:
            raise ValueError(f'Ambiguous {key}: {len(matches)} matching files. Use a folder with one copy.')
        if matches:
            files[key] = matches[0]
        elif key in ('metadata', 'tracking'):
            raise ValueError(f'Missing required file: {name}')
    return files

def teams_from_metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter('Team'))
    if len(teams) != 2 or teams[0].get('id') == teams[1].get('id'):
        raise ValueError('Expected exactly two distinct teams in metadata')
    ids = [t.get('id') for t in teams]
    slots = [p.get('teamId') for p in root.iter('Player')]
    if not slots or any(s not in ids for s in slots):
        raise ValueError('Unknown team or no track slots in metadata')
    return np.array([ids.index(s) for s in slots]), [t.findtext('Name') or t.get('id') for t in teams]

def read_tracking(path, slot_teams, n_frames=N_FRAMES):
    """Validate every frame; compute the official nearest-player evidence in metres."""
    codes = np.full(n_frames, -1, dtype=np.int8)
    observed = np.zeros(n_frames, dtype=bool)
    visible = np.zeros(n_frames, dtype=np.int16)
    pitch = np.array(GAMES[GAME]['pitch'])
    count = 0
    with Path(path).open() as handle:
        for line_no, line in enumerate(handle, 1):
            if line_no > n_frames:
                raise ValueError(f'Too many tracking records; expected {n_frames}')
            parts = line.strip().split(':')
            if len(parts) != 3 or int(parts[0]) != line_no:
                raise ValueError(f'Tracking alignment error at record {line_no}; expected frame ID {line_no}')
            groups = parts[1].rstrip(';').split(';')
            if len(groups) != len(slot_teams):
                raise ValueError(f'Track-slot count mismatch at record {line_no}')
            players = np.array([[float(v) for v in group.split(',')] for group in groups])
            ball = np.array([float(v) for v in parts[2].split(',')])
            if players.shape != (len(slot_teams), 3) or ball.shape != (3,):
                raise ValueError(f'Invalid coordinate dimensions at record {line_no}')
            valid = np.isfinite(players[:, :2]).all(axis=1)
            visible[line_no - 1] = valid.sum()
            if np.isfinite(ball[:2]).all():
                observed[line_no - 1] = True
                if valid.any():
                    d = np.sum(((players[:, :2] - ball[:2]) * pitch) ** 2, axis=1)
                    d[~valid] = np.inf
                    codes[line_no - 1] = slot_teams[np.argmin(d)]
            count = line_no
    if count != n_frames:
        raise ValueError(f'Incomplete tracking: {count} records; expected {n_frames}')
    return codes, observed, visible

def intervals(labels):
    changes = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1, len(labels)]
    return [[int(a), int(b), str(labels[a])] for a, b in zip(changes[:-1], changes[1:])]

def describe(labels, blocks=BLOCKS):
    segments = []
    selected = []
    for a, b in blocks:
        selected.append(labels[a:b])
        segments += [[x+a, y+a, label] for x, y, label in intervals(labels[a:b])]
    arr = np.concatenate(selected)
    spells = [s for s in segments if s[2] in ('H', 'A')]
    return {'counts': {k: int((arr == k).sum()) for k in 'HADX'},
            'possession_count': len(spells),
            'mean_possession_seconds': round(float(np.mean([(b-a)/FPS for a,b,_ in spells])), 3) if spells else 0,
            'short_possessions_under_1s': sum(b-a < FPS for a,b,_ in spells),
            'coverage_frames': len(arr)}

def run_baselines(data_root, out):
    files = discover(data_root)
    out = Path(out).resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('Output folder is not empty. Choose a new run folder to preserve previous experiments.')
    slot_teams, names = teams_from_metadata(files['metadata'])
    codes, observed, visible = read_tracking(files['tracking'], slot_teams)
    mask = np.zeros(N_FRAMES, bool)
    for a,b in BLOCKS:
        mask[a:b] = True
    summary = {'schema_version': 1, 'mode': 'licensed_local', 'title': 'Official baseline reproduction',
               'model_status': 'Baseline only; Control Switch candidate verification is not implemented.',
               'score_status': 'Not scored. Format validation is not an accuracy measurement.',
               'n_frames': N_FRAMES, 'fps': FPS, 'blocks': BLOCKS, 'teams': {'H': names[0], 'A': names[1]},
               'audit': {'records': N_FRAMES, 'alignment_valid': True, 'track_slots': len(slot_teams),
                         'ball_observed_fraction_graded': round(float(observed[mask].mean()), 6),
                         'mean_visible_players_graded': round(float(visible[mask].mean()), 3)},
               'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
               'upstream_commit': (UPSTREAM/'UPSTREAM_COMMIT').read_text().strip(), 'variants': []}
    timelines = {}
    out.mkdir(parents=True, exist_ok=True)
    for name, window in VARIANTS:
        labels = smooth(codes, window)
        table = pd.DataFrame({'frame': np.arange(N_FRAMES), 'label': labels})
        reason = pm.validate_submission(table, N_FRAMES)
        if reason:
            raise ValueError(f'Official validator rejected {name}: {reason}')
        csv_path = out / f'{name}.csv'
        write_submission(str(csv_path), labels)
        summary['variants'].append({'name': name, 'half_window_frames': window,
                                    'format_valid': True, 'sha256': digest(csv_path), **describe(labels)})
        timelines[name] = intervals(labels)
    manifest = {'inputs': {key: {'path': str(path), 'bytes': path.stat().st_size,
                              **({'sha256': digest(path)} if key in ('metadata', 'tracking') else {})}
                           for key,path in files.items()},
                'used_assets': ['metadata', 'tracking'], 'optional_assets_discovered_only': [k for k in files if k not in ('metadata', 'tracking')]}
    for filename, value in [('summary.json', summary), ('timeline.json', timelines), ('manifest.json', manifest)]:
        (out/filename).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    return summary

def demo_data():
    """Entirely invented sequence: never derived from the competition match."""
    n = 3000
    rng = np.random.default_rng(42)
    codes = np.repeat(np.array([0,-1,1,0,-1,1,0,-1,1,0],dtype=np.int8), 300)
    noise = rng.choice(n, 280, replace=False)
    codes[noise] = rng.choice([-1,0,1], len(noise))
    variants=[]; timelines={}
    for name, window in VARIANTS:
        labels=smooth(codes,window)
        variants.append({'name':name,'half_window_frames':window,'format_valid':False,'sha256':None,**describe(labels,[(0,n)])})
        timelines[name]=intervals(labels)
    return {'schema_version':1,'mode':'synthetic_demo','title':'Synthetic walkthrough',
            'model_status':'Invented movement-independent labels illustrating baseline smoothing. No match data or trained model.',
            'score_status':'Synthetic example. No accuracy or competition score.', 'n_frames':n,'fps':FPS,'blocks':[[0,n]],
            'teams':{'H':'Home team','A':'Away team'},'variants':variants,
            'audit':None},timelines
