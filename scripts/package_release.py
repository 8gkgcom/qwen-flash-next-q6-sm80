#!/usr/bin/env python3
"""Audit and package source files; excludes data, caches, Git history and secrets."""
import argparse
import hashlib
from pathlib import Path
import zipfile
from audit_release import ROOT,audit,files


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'dist/flashnext-q6-sm80-0.1.0.zip')
    args=parser.parse_args()
    output=args.output.resolve()
    if output.is_relative_to(ROOT) and not output.is_relative_to(ROOT/'dist'):
        raise SystemExit('Place the archive under dist/ or outside the source tree')
    result=audit(ROOT)
    if result['issues']:raise SystemExit(str(result['issues']))
    if args.output.exists():raise SystemExit('Archive already exists; choose a new output filename')
    selected=[p for p in files(ROOT) if p.name!='SHA256SUMS']
    sums=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in selected)
    (ROOT/'SHA256SUMS').write_text(sums,encoding='utf-8',newline='\n')
    selected.append(ROOT/'SHA256SUMS')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for p in selected:
            info=zipfile.ZipInfo('qwen-flash-next-q6-sm80/'+p.relative_to(ROOT).as_posix(),date_time=(2026,10,22,0,0,0))
            info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
            archive.writestr(info,p.read_bytes())
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    args.output.with_suffix(args.output.suffix+'.sha256').write_text(digest+'  '+args.output.name+'\n')
    print(f'{len(selected)} source files; {args.output.stat().st_size} bytes; sha256={digest}')


if __name__=='__main__':main()
