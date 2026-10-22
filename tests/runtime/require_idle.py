"""Explicit opt-in for manual GPU checks; never terminates a process."""
import os
import subprocess


def require_idle():
    if os.environ.get('Q6_RUN_GPU_TESTS')!='1':
        raise SystemExit('GPU checks are opt-in: use Q6_RUN_GPU_TESTS=1 on an idle test machine')
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if active:
        raise SystemExit('A GPU compute process is running; no process was stopped')
