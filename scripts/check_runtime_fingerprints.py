#!/usr/bin/env python3
"""Verify version-specific APIs without importing torch or creating CUDA contexts."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,default=Path('/usr/local/lib/python3.12/dist-packages'))
    args=p.parse_args();expected=json.loads((ROOT/'provenance/runtime-api-files.json').read_text())
    errors=[]
    for name,digest in expected.items():
        file=args.site/name
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
            errors.append(name)
    if errors:raise SystemExit('Runtime API fingerprint mismatch:\n'+'\n'.join(errors))
    print(f'Checked {len(expected)} version-specific runtime source files')


if __name__=='__main__':main()
