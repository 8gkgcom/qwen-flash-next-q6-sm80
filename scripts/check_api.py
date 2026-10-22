#!/usr/bin/env python3
"""Small functional API probe. Run only against the intended test service."""
import argparse
import concurrent.futures
import json
import os
import urllib.request


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url',required=True);p.add_argument('--model',default='flashnext-q6')
    args=p.parse_args();base=args.base_url.rstrip('/')
    headers={'Content-Type':'application/json'}
    if key:=os.environ.get('Q6_API_KEY'):headers['Authorization']='Bearer '+key
    def call(path,payload=None):
        request=urllib.request.Request(base+path,None if payload is None else json.dumps(payload).encode(),headers)
        with urllib.request.urlopen(request,timeout=180) as r:
            data=r.read()
            return json.loads(data) if data else None
    call('/health')
    models=call('/v1/models')
    assert args.model in [item['id'] for item in models['data']],'Requested model name not served'
    def generation(a,b):
        d=call('/v1/chat/completions',dict(model=args.model,messages=[dict(role='user',content=f'只输出数字：{a}+{b}等于多少？')],temperature=0,max_tokens=32,chat_template_kwargs=dict(enable_thinking=False)))
        assert d['choices'][0]['message']['content'].strip()==str(a+b),d
        return dict(expected=a+b,completion_tokens=d['usage']['completion_tokens'])
    with concurrent.futures.ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(generation,17,25),pool.submit(generation,31,58)]
        results=[f.result() for f in futures]
    queue=call('/queue/status')
    assert queue['running']==0 and queue['waiting']==0,queue
    print(json.dumps(dict(health=True,simultaneously_submitted_requests=results,queue=queue),indent=2))


if __name__=='__main__':main()
