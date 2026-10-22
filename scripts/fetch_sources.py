#!/usr/bin/env python3
"""Download hash-pinned source archives into a new local build directory."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def extract_checked(archive, target):
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        top = {m.name.split('/')[0] for m in members}
        if len(top) != 1:
            raise ValueError('Expected one archive root')
        for member in members:
            name = Path(member.name)
            if name.is_absolute() or '..' in name.parts or member.issym() or member.islnk():
                raise ValueError('Unsafe archive member')
            if not member.isfile() and not member.isdir():
                raise ValueError('Unsupported archive member')
        tar.extractall(target, members=members, filter='data')
    return target / next(iter(top))


def fetch(name, item, output):
    output.mkdir(parents=True, exist_ok=True)
    destination = output / name
    if destination.exists():
        marker = destination / '.q6-source.json'
        if not marker.is_file() or json.loads(marker.read_text()) != item:
            raise ValueError(f'{name}: existing directory is not the pinned source')
        print(f'{name}: already downloaded; use a fresh build directory after source edits')
        return destination
    with tempfile.TemporaryDirectory(prefix='q6-fetch-', dir=output) as tmp:
        temp = Path(tmp)
        archive = temp / 'source.tar.gz'
        with urllib.request.urlopen(item['url'], timeout=120) as response, archive.open('wb') as out:
            while block := response.read(1024 * 1024):
                out.write(block)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError(f'{name}: source checksum mismatch')
        unpacked = extract_checked(archive, temp / 'unpacked')
        (unpacked / '.q6-source.json').write_text(json.dumps(item, indent=2)+'\n')
        os.rename(unpacked, destination)
    print(f'{name}: verified {item["revision"]}')
    return destination


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT/'build/deps')
    p.add_argument('--only', nargs='+', choices=['sm80','plugin','llama'], default=['sm80','plugin','llama'])
    args = p.parse_args()
    lock = json.loads((ROOT/'dependencies.lock.json').read_text())
    for name in args.only:
        fetch(name, lock[name], args.output.resolve())


if __name__ == '__main__':
    main()
