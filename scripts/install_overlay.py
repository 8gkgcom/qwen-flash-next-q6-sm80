#!/usr/bin/env python3
"""Install a checked source overlay during an isolated image build."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def plan_overlay(target, source, manifest):
    changes = []
    errors = []
    for relative, item in manifest.items():
        rel = Path(relative)
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Invalid overlay path')
        src, dst = source/rel, target/rel
        if not dst.resolve().is_relative_to(target.resolve()):
            raise ValueError('Overlay target escapes destination')
        if dst.exists() and not dst.is_file():
            errors.append(relative+': destination is not a regular file')
        if digest(src) != item['after_sha256']:
            errors.append(relative+': packaged source changed')
        actual = digest(dst)
        if actual == item['after_sha256']:
            continue
        if actual != item['before_sha256']:
            errors.append(relative+': incompatible upstream source')
        changes.append((src,dst))
    if errors:
        raise ValueError('\n'.join(errors))
    return changes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', choices=['sm80','plugin'], required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    source = ROOT/('runtime/overlay' if args.kind=='sm80' else 'runtime/plugin-overlay')
    manifest = json.loads((ROOT/('provenance/overlay-files.json' if args.kind=='sm80' else 'provenance/plugin-files.json')).read_text())
    changes = plan_overlay(args.target.resolve(), source, manifest)
    if not args.check_only:
        # BuildKit need not expose /.dockerenv during RUN. Restrict installation
        # to the explicit image-build flag and the Dockerfile's fixed roots.
        allowed = Path('/usr/local/lib/python3.12/dist-packages') if args.kind=='sm80' else Path('/opt/q6-deps/plugin')
        if os.environ.get('Q6_IMAGE_BUILD') != '1' or ROOT != Path('/opt/q6-sm80') or args.target.resolve()!=allowed:
            raise SystemExit('Installation is restricted to the isolated Docker build; use --check-only for inspection')
        for src,dst in changes:
            dst.parent.mkdir(parents=True,exist_ok=True)
            temp = dst.with_name(dst.name+'.q6-new')
            shutil.copyfile(src,temp)
            os.replace(temp,dst)
    print(json.dumps({'kind':args.kind,'checked':len(manifest),'changes':len(changes),'check_only':args.check_only}))


if __name__=='__main__':
    main()
