"""Streaming single-request decode baseline, no credentials in artifacts."""
import json
import re
import subprocess
import threading
import time
import urllib.request
from pathlib import Path

import os
ROOT=Path(os.environ.get('Q6_TEST_OUTPUT','./test-output'))
ROOT.mkdir(parents=True,exist_ok=True)
API_BASE=os.environ.get('Q6_API_BASE','http://127.0.0.1:8000').rstrip('/')
settings={'api_key':os.environ.get('Q6_API_KEY',''), 'served_model_name':os.environ['Q6_MODEL_NAME']}
headers={'Authorization':'Bearer '+settings['api_key'],'Content-Type':'application/json'}
def metrics():
    raw=urllib.request.urlopen(API_BASE+'/metrics',timeout=10).read().decode()
    return {k:sum(float(v) for v in re.findall(r'^vllm:'+k+r'\{[^}]*\} ([0-9.e+]+)',raw,re.M))
            for k in ['spec_decode_num_drafts_total','spec_decode_num_draft_tokens_total',
                      'spec_decode_num_accepted_tokens_total']}

samples=[]
stop=threading.Event()
def monitor():
    while not stop.wait(.5):
        p=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,clocks.sm,power.draw,temperature.gpu',
                          '--format=csv,noheader,nounits'],capture_output=True,text=True)
        samples.append({'time':time.time(),'gpus':p.stdout.strip()})

def run(name,prompt):
    body=dict(model=settings['served_model_name'],messages=[{'role':'user','content':prompt}],
              temperature=0,seed=1234,max_tokens=1024,min_tokens=1024,ignore_eos=True,
              stream=True,stream_options={'include_usage':True},
              chat_template_kwargs={'enable_thinking':False})
    request=urllib.request.Request(API_BASE+'/v1/chat/completions',headers=headers,
                                   data=json.dumps(body).encode())
    before=metrics();start=time.monotonic();first=last=None;text=[];usage={}
    with urllib.request.urlopen(request,timeout=180) as response:
        for line in response:
            if not line.startswith(b'data: '):continue
            raw=line[6:].strip()
            if raw==b'[DONE]':break
            item=json.loads(raw)
            if item.get('usage'):usage=item['usage']
            for choice in item.get('choices',[]):
                content=choice.get('delta',{}).get('content')
                if content:
                    now=time.monotonic();first=first or now;last=now;text.append(content)
    end=time.monotonic();after=metrics()
    delta={k:after[k]-before[k] for k in before}
    result=dict(case=name,usage=usage,ttft_seconds=first-start,elapsed_seconds=end-start,
                decode_tps=(usage['completion_tokens']-1)/(last-first),
                end_to_end_tps=usage['completion_tokens']/(end-start),
                speculative=delta,answer=''.join(text))
    print(json.dumps({k:v for k,v in result.items() if k!='answer'}),flush=True)
    return result

if __name__=='__main__':
    cases=[
      ('english','Write a detailed 2000-word practical tutorial about Python dictionaries, including iteration, nesting, comprehensions, common mistakes, and runnable examples. Explain every example.'),
      ('chinese','请写一篇不少于2500字的中文教程，详细解释数据库索引的原理、B树和B+树、联合索引、覆盖索引、最左前缀、常见错误及SQL实例。不要提前总结，充分展开每个部分。'),
      ('code','Implement a complete Python in-memory key value database with expiring keys, transactions, nested transactions, thread safety, persistence, and unit tests. Include all source code and explanations. Write at least 250 lines.'),
      ('long_8k','Archive of engineering notes:\n'+''.join(f'Record {i:04d}: We use clear interfaces and tests to keep our database code maintainable.\n' for i in range(400))+'\nWrite a detailed 2000-word guide on designing a reliable database service, including concrete Python examples and failure recovery.')]
    thread=threading.Thread(target=monitor);thread.start()
    results=[]
    try:
        for name,prompt in cases:results.append(run(name,prompt))
    finally:
        stop.set();thread.join()
        (ROOT/'single-results.json').write_text(json.dumps({'results':[{k:v for k,v in r.items() if k!='answer'} for r in results]},ensure_ascii=False,indent=2))
