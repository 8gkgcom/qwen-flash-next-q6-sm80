#!/usr/bin/env python3
"""Reproduce cold-prefix long input and short-request latency probes."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import os
import time
import uuid


def main():
    import httpx
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--model',default='flashnext-q6')
    p.add_argument('--label',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--input-tokens',type=int,choices=[32768,262000],default=32768)
    p.add_argument('--mixed',action='store_true',help='Submit a short request during the long request')
    args=p.parse_args()
    if args.output.exists():raise SystemExit('Output exists; choose a new result file')
    headers={}
    if key:=os.environ.get('Q6_API_KEY'):headers['Authorization']='Bearer '+key
    with httpx.Client(base_url=args.base_url.rstrip('/'),headers=headers,timeout=900) as client:
        response=client.get('/queue/status');response.raise_for_status();queue=response.json()
        if queue['running'] or queue['waiting']:raise RuntimeError('API has active requests; use an idle test service')
        repeats=max(4000,args.input_tokens//8+1000)
        content='记住 START_CODE = 7319。下面是档案材料。\n'+('The archive contains ordinary records about blue skies and green fields.\n'*repeats)+'\nEND_CODE = 9053\n请按顺序只输出 START_CODE 和 END_CODE 两个数字。'
        r=client.post('/tokenize',json=dict(model=args.model,messages=[dict(role='user',content=content)],chat_template_kwargs=dict(enable_thinking=False)))
        r.raise_for_status();tokens=r.json()['tokens'];extra=len(tokens)-args.input_tokens
        if extra<=0:raise RuntimeError('Tokenizer produced too few tokens')
        tokens=tokens[:1000]+tokens[1000+extra:]
        assert len(tokens)==args.input_tokens
        def long_request():
            start=time.perf_counter();first=None;answer='';usage={}
            with client.stream('POST','/v1/completions',json=dict(model=args.model,prompt=tokens,cache_salt=str(uuid.uuid4()),max_tokens=32,temperature=0,stream=True,stream_options=dict(include_usage=True))) as stream:
                stream.raise_for_status()
                for line in stream.iter_lines():
                    if not line.startswith('data:'):continue
                    data=line[5:].strip()
                    if data=='[DONE]':break
                    event=json.loads(data)
                    if event.get('error'):raise RuntimeError(str(event['error']))
                    usage=event.get('usage') or usage
                    for choice in event.get('choices',[]):
                        text=choice.get('text') or ''
                        if text:first=time.perf_counter() if first is None else first;answer+=text
            if first is None:raise RuntimeError('Empty generation')
            if not all(v in answer for v in ['7319','9053']):raise RuntimeError('Marker retrieval failed: '+answer)
            return dict(seconds=time.perf_counter()-start,ttft=first-start,usage=usage,marker_retrieval_passed=True)
        result=dict(label=args.label,input_tokens=args.input_tokens,temperature=0,enable_thinking=False,fresh_cache_salt=True,single=long_request())
        if args.mixed:
            with ThreadPoolExecutor(1) as pool:
                future=pool.submit(long_request);time.sleep(1)
                active_at_submission=not future.done();start=time.perf_counter()
                r=client.post('/v1/chat/completions',json=dict(model=args.model,messages=[dict(role='user',content='只输出数字：17+25等于多少？')],temperature=0,max_tokens=32,chat_template_kwargs=dict(enable_thinking=False)))
                r.raise_for_status()
                if r.json()['choices'][0]['message']['content'].strip()!='42':raise RuntimeError('Short request failed')
                result['short_during_long']=dict(seconds=time.perf_counter()-start,long_active_at_submission=active_at_submission,long_active_at_completion=not future.done())
                result['mixed_long']=future.result()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
