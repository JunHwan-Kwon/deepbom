"""Run qualified optional framework checks with the selected Python installation.

Install deepbom[torch,tensorflow,report,autolog,lightning,metrics,tracking], h5py
and the repository's Node development dependencies first. Set DEEPBOM_ENGINE
and its digest when checking source code rather than an installed wheel.
"""
import argparse
from pathlib import Path
import subprocess
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--torchlens',action='store_true',help='Also test the separately installed experimental TorchLens 2.36.1 backend')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
checks=['check-native-monitor-inspection.py','check-automatic-integrations.py',
        'check-analysis-jobs.py','check-managed-analysis.py','check-analyzer-runner.py','check-builtin-metrics.py','check-training-board-scale.py','check-hdf5-source.py',
        'check-tensorflow-source-numerics.py','check-tensorflow-checkpoint.py',
        'check-torch-source-artifacts.py','check-redesign-codegen.py']
if args.torchlens:checks.append('check-torchlens-prototype.py')
for script in checks:
    print(f'Running {script}',flush=True)
    subprocess.run([sys.executable,str(root/'scripts'/script)],cwd=root,check=True)
print(f'Qualified development integrations passed ({len(checks)} suites)')
