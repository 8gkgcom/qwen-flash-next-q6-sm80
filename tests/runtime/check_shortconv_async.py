import dataclasses, functools, inspect, json, pathlib, sys, textwrap, types
from require_idle import require_idle
require_idle()
import torch
from vllm.v1.attention.backends import short_conv_attn as m
from vllm.utils.torch_utils import async_tensor_h2d
from flashnext_gguf.optimizations.q6_shortconv_async import transform

def upload(t, device):
    assert t.device.type == 'cpu'
    return t if device.type == 'cpu' else async_tensor_h2d(t, device=device)

source=textwrap.dedent(inspect.getsource(m.PleShortConvAttentionMetadataBuilder.build))
scope=dict(m.PleShortConvAttentionMetadataBuilder.build.__globals__)
# The table-mapping helper is outside the changed code and model-dependent.
scope['mamba_get_block_table_tensor']=lambda table,*args:table
old_scope=dict(scope)
exec(compile(source,'<original>','exec'),old_scope)
new_scope=dict(scope,_q6_upload_index=upload)
exec(compile(transform(source),'<patched>','exec'),new_scope)

def builder(device, graph):
    b=object.__new__(m.PleShortConvAttentionMetadataBuilder)
    b.use_spec_decode=True;b.num_spec=2;b.kv_cache_spec=None
    b.vllm_config=types.SimpleNamespace(cache_config=types.SimpleNamespace(mamba_cache_mode='align'))
    b.use_full_cuda_graph=graph;b.decode_cudagraph_max_bs=8;b.decode_cudagraph_max_tokens=24
    for name,dtype,size in [('spec_state_indices_tensor',torch.int32,8),('spec_sequence_masks',torch.bool,8),
      ('spec_query_start_loc',torch.int32,9),('num_accepted_tokens',torch.int32,8)]:
        setattr(b,name,torch.empty(size,dtype=dtype,device=device))
    return b

cases=[([3],[2],False),([3],[2],True),([3,3],[2,2],False),([3,3],[2,2],True),
       ([3,0],[2,-1],True),([3,1],[2,-1],False),([1,3],[-1,2],False),
       ([3,4],[2,-1],False),([4,3],[-1,2],False),([1,3,5],[-1,2,-1],False),
       ([3,1,3],[2,-1,2],False),([2,2],[1,1],True)]
passed=[]
def equal(a,b):
 if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 return a==b
for dev in ['cpu','cuda:0']:
 device=torch.device(dev)
 if device.type=='cuda':torch.cuda.set_per_process_memory_fraction(.015,device)
 for lens,drafts,graph in cases:
    n=len(lens);qcpu=torch.tensor([0]+list(torch.tensor(lens).cumsum(0).tolist()),dtype=torch.int32)
    q=qcpu.to(device);table=torch.arange(n,dtype=torch.int32,device=device).reshape(n,1)+7
    seq=torch.tensor(lens,dtype=torch.int32,device=device)+100
    metadata=types.SimpleNamespace(query_start_loc=q,query_start_loc_cpu=qcpu,num_reqs=n,
          num_actual_tokens=sum(lens),block_table_tensor=table,seq_lens=seq,
          compute_num_computed_tokens=lambda: seq-torch.diff(q))
    accepted=torch.full((n,),2,dtype=torch.int32,device=device)
    kwargs=dict(num_accepted_tokens=accepted,num_decode_draft_tokens_cpu=torch.tensor(drafts))
    old=old_scope['build'](builder(device,graph),0,metadata,**kwargs)
    new=new_scope['build'](builder(device,graph),0,metadata,**kwargs)
    for field in dataclasses.fields(old):
        a,b=getattr(old,field.name),getattr(new,field.name)
        assert equal(a,b),(dev,lens,field.name)
    passed.append(dict(device=dev,lens=lens,graph=graph))
# Check pinned sources survive asynchronous reuse and delayed execution.
with torch.cuda.stream(torch.cuda.Stream()):
    torch.cuda._sleep(2000000)
    outputs=[upload(torch.tensor([i],dtype=torch.int64),torch.device('cuda:0')) for i in range(256)]
    combined=torch.cat(outputs)
    assert torch.equal(combined.cpu(),torch.arange(256)), 'Pinned staging lifetime failed'
print(json.dumps(dict(metadata_cases=len(passed),pinned_lifetime_rows=256,passed=passed)))
