#!/usr/bin/env python3
"""Compare matched single-request experiments; reject sampling/prompt changes."""
import argparse
import json
from pathlib import Path
import statistics


def compare(a,b):
    for key in ['model','sampling','max_tokens','prompt_sha256','metric']:
        if a[key]!=b[key]:raise ValueError('Not a matched experiment: '+key)
    groups={}
    for label,obj in [('baseline',a),('candidate',b)]:
        for row in obj['runs']:
            groups.setdefault(row['prompt_id'],{}).setdefault(label,[]).append(row)
    result=[]
    for prompt,pair in groups.items():
        if set(pair)!= {'baseline','candidate'}:raise ValueError('Missing prompt: '+prompt)
        x,y=pair['baseline'],pair['candidate']
        first,second=statistics.median(r['decode_tps'] for r in x),statistics.median(r['decode_tps'] for r in y)
        result.append(dict(prompt_id=prompt,baseline_tps=first,candidate_tps=second,
            change_percent=(second/first-1)*100,
            all_outputs_identical=len({r['text_sha256'] for r in [*x,*y]})==1))
    return dict(prompts=result,mean_change_percent=statistics.mean(r['change_percent'] for r in result),
                note='Hash agreement is a finite regression check, not a code-correctness score or an equivalence proof')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('baseline',type=Path);p.add_argument('candidate',type=Path)
    a=p.parse_args();print(json.dumps(compare(json.loads(a.baseline.read_text()),json.loads(a.candidate.read_text())),indent=2))


if __name__=='__main__':main()
