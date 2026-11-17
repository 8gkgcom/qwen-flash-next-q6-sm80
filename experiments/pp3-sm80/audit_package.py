"""Offline public-package integrity and sensitive-pattern check."""
from pathlib import Path
import ast,hashlib,json,re
root=Path(__file__).resolve().parent
manifest=json.loads((root/'source-manifest.json').read_text())
patterns=[r'GPU-[0-9a-fA-F-]{20,}',r'(?:192\.168\.|10\.\d+\.|172\.(?:1[6-9]|2\d|3[01])\.)\d+\.\d+',r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',r'/media/nvme[0-9]/',r'/opt/[a-z-]*console',r'C:[\\/]Users[\\/]']
for name,digest in manifest.items():
 p=root/name
 assert p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
 text=p.read_text()
 if p.suffix=='.py':ast.parse(text,filename=name)
 if p.suffix=='.json':json.loads(text)
 if name!='audit_package.py':
  for pattern in patterns:assert not re.search(pattern,text), (name,pattern)
print('PASS: hashes, Python/JSON syntax and sensitive-pattern scan; manual review still required')
