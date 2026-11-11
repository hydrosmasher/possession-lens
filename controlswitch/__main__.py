import argparse
import sys
from pathlib import Path
from .pipeline import run_baselines

def main():
    parser=argparse.ArgumentParser(description='Control Switch: local possession research toolkit')
    sub=parser.add_subparsers(dest='command',required=True)
    base=sub.add_parser('baseline',help='Validate tactical tracking and reproduce the four official baselines')
    base.add_argument('--data-root',required=True,type=Path)
    base.add_argument('--out',required=True,type=Path)
    web=sub.add_parser('serve',help='Open the local judge dashboard (synthetic demo by default)')
    web.add_argument('--run-dir',type=Path)
    web.add_argument('--port',type=int,default=8765)
    demo=sub.add_parser('export-demo',help='Create a self-contained synthetic HTML demo; no server needed')
    demo.add_argument('--out',type=Path,default=Path('dashboard.html'))
    dashboard=sub.add_parser('dashboard',help='Build a private offline dashboard from a local prediction run')
    dashboard.add_argument('--run-dir',type=Path,required=True)
    dashboard.add_argument('--out',type=Path,default=Path('local'))
    local_import=sub.add_parser('import-external',help='Import already downloaded external XMLs without network access')
    local_import.add_argument('--source',type=Path,default=Path.home()/'Downloads')
    local_import.add_argument('--destination',type=Path,default=Path('data/external/raw'))
    download=sub.add_parser('download-external',help='Download pinned CC-BY-4.0 external match files')
    download.add_argument('--out',type=Path,default=Path('data/external/raw'))
    download.add_argument('--allow-network',action='store_true',help='Explicitly allow fetching missing external files after checking Downloads')
    download.add_argument('--source',type=Path,default=Path.home()/'Downloads')
    download.add_argument('--matches',nargs='+',default=['J03WOH','J03WPY','J03WMX'])
    experiment=sub.add_parser('benchmark',help='Fit and evaluate using external matches only')
    experiment.add_argument('--external-root',required=True,type=Path)
    experiment.add_argument('--out',required=True,type=Path)
    experiment.add_argument('--single-match-pilot',action='store_true',help='Explicitly permit a weaker first-half/second-half pilot when exactly one external match exists')
    development=sub.add_parser('develop',help='Validate interaction features and causal forecasting without reusing the viewed test set')
    development.add_argument('--external-root',required=True,type=Path)
    development.add_argument('--pilot-dir',required=True,type=Path)
    development.add_argument('--out',required=True,type=Path)
    layout=sub.add_parser('organise-data',help='Create private organised links to original match assets')
    layout.add_argument('--source',type=Path,required=True)
    layout.add_argument('--destination',type=Path,default=Path('data/raw/provided'))
    recurrent=sub.add_parser('rnn-train',help='Train an actual recurrent network on external matches only')
    recurrent.add_argument('--external-root',type=Path,required=True)
    recurrent.add_argument('--out',type=Path,required=True)
    recurrent.add_argument('--epochs',type=int,default=5)
    recurrent.add_argument('--single-match-pilot',action='store_true')
    inference=sub.add_parser('rnn-predict',help='Apply frozen external RNN weights; export labels and possession estimates')
    inference.add_argument('--data-root',type=Path,required=True)
    inference.add_argument('--model',type=Path,required=True)
    inference.add_argument('--out',type=Path,required=True)
    player=sub.add_parser('player-report',help='Export anonymous-track contribution and interval CSVs; no identity claims')
    player.add_argument('--run-dir',type=Path,required=True)
    player.add_argument('--out',type=Path,default=Path('local/player-control'))
    args=parser.parse_args()
    try:
        if args.command=='player-report':
            from .player_report import export_player_report
            export_player_report(args.run_dir,args.out)
            print('Anonymous-track reports exported locally. Player identities remain unverified.')
        elif args.command=='dashboard':
            from .dashboard import export_local
            export_local(args.run_dir,args.out)
            print('Private local dashboard saved. Nothing uploaded.')
        elif args.command=='baseline':
            summary=run_baselines(args.data_root,args.out)
            print('Tracking alignment verified; four full-length CSVs passed the official format validator.')
            print(f'Run saved to {args.out.resolve()}')
            print('Accuracy not measured; this is the official baseline, not a trained Control Switch model.')
        elif args.command=='organise-data':
            from .data_setup import organise
            manifest=organise(args.source,args.destination)
            print(f'Organised {len(manifest["assets"])} assets in {args.destination.resolve()}; originals unchanged.')
        elif args.command=='rnn-train':
            if args.epochs<1:raise ValueError('Epochs must be positive')
            from .rnn_project import train_external
            report=train_external(args.external_root,args.out,args.epochs,args.single_match_pilot)
            print(f'RNN checkpoint saved. External validation spellF1: {report["validation_metrics"]["spell_f1_mean"]:.4f}. Not a competition score.')
        elif args.command=='rnn-predict':
            from .rnn_project import predict_provided
            predict_provided(args.data_root,args.model,args.out)
            print(f'Predictions and statistics saved in {args.out.resolve()}. No competition accuracy check performed.')
        elif args.command=='import-external':
            from .local_data import import_local
            report=import_local(args.source,args.destination)
            print(f'Imported or verified {len(report["assets"])} existing external files. No network access; originals unchanged.')
        elif args.command=='download-external':
            from .external import download_external
            download_external(args.out,args.matches,allow_network=args.allow_network,source=args.source)
        elif args.command=='develop':
            from .development import develop
            report=develop(args.external_root,args.pilot_dir,args.out)
            print(f'Validation-only development report: {args.out.resolve() / "report.html"}')
            for r in report['detection']:print(f"{r['condition']} {r['model']}: validation spellF1={r['spell_f1_mean']:.4f}")
            for r in report['forecast']:print(f"{r['condition']} forecast: validation AP={r['average_precision']:.4f}; prevalence={r['prevalence']:.4f}; Brier={r['brier_score']:.4f}; prior Brier={r['training_prior_brier_score']:.4f}")
        elif args.command=='benchmark':
            from .experiment import benchmark
            report=benchmark(args.external_root,args.out,args.single_match_pilot)
            print(f'External experiment complete: {args.out.resolve() / "report.html"}')
            print(report['mode'])
            for row in report['results']:
                print(f"{row['condition']:7} {row['model']:20} spellF1={row['spell_f1_mean']:.4f} macroF1={row['macro_f1']:.4f}")
        elif args.command=='export-demo':
            from .export import export_demo
            export_demo(args.out)
            print(f'Synthetic standalone demo saved to {args.out.resolve()}')
        else:
            from .server import serve
            serve(args.run_dir,args.port)
    except (ValueError,OSError) as e:
        parser.exit(2,f'Error: {e}\n')

if __name__=='__main__':
    main()
