#!/usr/bin/env python3
"""Repeat the original three single-request speed probes against an explicit API."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import statistics
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def request(base,model,prompt,tokens,temperature,timeout,save_text=False):
    payload=dict(model=model,messages=[dict(role='user',content=prompt)],seed=42,
                 temperature=temperature,top_p=.95,top_k=20,min_p=0.,presence_penalty=.5,repetition_penalty=1.,
                 chat_template_kwargs=dict(enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=False),
                 max_tokens=tokens,min_tokens=tokens,ignore_eos=True,stream=True,stream_options=dict(include_usage=True))
    headers={'Content-Type':'application/json'}
    if key:=os.environ.get('Q6_API_KEY'):headers['Authorization']='Bearer '+key
    req=urllib.request.Request(base.rstrip('/')+'/v1/chat/completions',json.dumps(payload).encode(),headers)
    start=time.perf_counter();first=last=None;usage={};content='';reasoning=''
    with urllib.request.urlopen(req,timeout=timeout) as response:
        for raw in response:
            line=raw.decode('utf-8').strip()
            if not line.startswith('data:'):continue
            data=line[5:].strip()
            if data=='[DONE]':break
            event=json.loads(data)
            if event.get('error'):raise RuntimeError(str(event['error']))
            usage=event.get('usage') or usage
            for choice in event.get('choices',[]):
                delta=choice.get('delta') or {}
                c=delta.get('content') or '';r=delta.get('reasoning') or delta.get('reasoning_content') or ''
                if c or r:
                    now=time.perf_counter();first=now if first is None else first;last=now
                    content+=c;reasoning+=r
    if first is None or last is None or last<=first or not usage.get('completion_tokens'):
        raise RuntimeError('Missing stream timing or usage; cannot calculate speed')
    row=dict(seconds=time.perf_counter()-start,ttft=first-start,
             decode_tps=(usage['completion_tokens']-1)/(last-first),usage=usage,
             text_sha256=hashlib.sha256((reasoning+'\0'+content).encode()).hexdigest())
    if save_text:row.update(content=content,reasoning=reasoning)
    return row


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True)
    p.add_argument('--model',default='flashnext-q6')
    p.add_argument('--label',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=512)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--temperature',type=float,default=1.)
    p.add_argument('--timeout',type=float,default=300)
    p.add_argument('--save-text',action='store_true')
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Result exists; use a new output file')
    if args.tokens<2 or args.repeats<1:raise SystemExit('Need at least two tokens and one repeat')
    prompts=json.loads((ROOT/'benchmarks/prompts.json').read_text(encoding='utf-8'))
    sampling=dict(temperature=args.temperature,top_p=.95,top_k=20,min_p=0.,presence_penalty=.5,repetition_penalty=1.,seed=42,reasoning_effort='xhigh',preserve_thinking=False)
    out=dict(label=args.label,model=args.model,sampling=sampling,max_tokens=args.tokens,repeats=args.repeats,
             metric='(completion_tokens-1)/(last_nonempty_delta-first_nonempty_delta)',
             prompt_sha256=hashlib.sha256((ROOT/'benchmarks/prompts.json').read_bytes()).hexdigest(),runs=[])
    out['warmup']=request(args.base_url,args.model,prompts[0]['prompt'],args.tokens,args.temperature,args.timeout)
    for repeat in range(args.repeats):
        for prompt in prompts:
            row=request(args.base_url,args.model,prompt['prompt'],args.tokens,args.temperature,args.timeout,args.save_text)
            row.update(prompt_id=prompt['id'],repeat=repeat);out['runs'].append(row)
            print(json.dumps({k:v for k,v in row.items() if k not in ('content','reasoning')}),flush=True)
    out['median_decode_tps']=statistics.median(r['decode_tps'] for r in out['runs'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Saved',args.output)


if __name__=='__main__':main()
