#!/usr/bin/env python3
"""Check distributable files for common secret formats, private paths and binaries."""
from pathlib import Path
import argparse
import ast
import json
import re

ROOT=Path(__file__).resolve().parents[1]
EXCLUDED={'.git','__pycache__','build','dist','.venv','.venv-test','data','runs','cache'}
BLOCKED_SUFFIXES={'.gguf','.safetensors','.so','.dll','.exe','.key','.pem','.pt','.bin','.log','.pyc'}
PATTERNS={
    'private-network-address':re.compile(r'(?<![\w.])(?:192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3})(?![\w.])'),
    'machine-specific-path':re.compile(r'(?:[A-Z]:[/\\]Users[/\\]|/(?:media/nvme\d|home/user|opt/model-console)(?:/|\b))',re.I),
    'gpu-identity':re.compile(r'GPU-[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}',re.I),
    'huggingface-token':re.compile(r'\bhf_[A-Za-z0-9]{24,}\b'),
    'github-token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})\b'),
    'private-key':re.compile(r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----'),
    'literal-password':re.compile(r'''(?i)\bpassword\s*[=:]\s*["'][^"'\n]+["']'''),
}


def files(root):
    for p in sorted(root.rglob('*')):
        rel=p.relative_to(root)
        # Only the root models/ holds weights; runtime/.../models/ holds source.
        if rel.parts[0]=='models':continue
        if any(part in EXCLUDED or part.startswith('.venv') or part.endswith('.egg-info') for part in rel.parts):continue
        if p.is_file():yield p


def audit(root):
    issues=[];count=0
    for p in files(root):
        rel=p.relative_to(root).as_posix();count+=1
        if p.is_symlink():issues.append((rel,'symlink'));continue
        if p.suffix.lower() in BLOCKED_SUFFIXES or p.name.startswith('.env') or p.name.startswith('local.'):
            issues.append((rel,'private/generated file'));continue
        if p.stat().st_size>5*1024**2:issues.append((rel,'unexpected large file'));continue
        try:text=p.read_text(encoding='utf-8')
        except UnicodeError:issues.append((rel,'non-text artifact'));continue
        for label,pattern in PATTERNS.items():
            if pattern.search(text):issues.append((rel,label))
        if p.suffix=='.py':
            try:ast.parse(text,filename=rel)
            except SyntaxError as e:issues.append((rel,'Python syntax: '+str(e)))
        if p.suffix=='.json':
            try:json.loads(text)
            except ValueError as e:issues.append((rel,'JSON syntax: '+str(e)))
    return dict(files_checked=count,issues=[dict(file=f,kind=k) for f,k in issues],
                limitation='Pattern and artifact audit, not a guarantee that every possible secret format is detectable')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    args=p.parse_args();result=audit(args.root.resolve());print(json.dumps(result,indent=2))
    if result['issues']:raise SystemExit(1)


if __name__=='__main__':main()
