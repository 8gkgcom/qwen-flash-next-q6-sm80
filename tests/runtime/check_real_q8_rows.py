import json, os, pathlib, sys, tempfile
root=pathlib.Path(os.environ.get('FLASHNEXT_GGUF_ROOT','/data/runtime'))
output=pathlib.Path(tempfile.mkdtemp(prefix='q6-runtime-test-'))
import time, types
import numpy as np,torch,gguf
from flashnext_gguf.q8 import Q8Rows
from flashnext_gguf.adapter import manifest,map_name
torch.set_num_threads(1)
entry=next(e for e in manifest() if e['name']=='per_layer_token_embd.weight')
table=Q8Rows(entry,resident=False)
rng=np.random.default_rng(20261022)
ids=torch.tensor([0,entry['shape'][1]-1,*rng.integers(0,entry['shape'][1],1024).tolist()],dtype=torch.int64)
expected=torch.from_numpy(gguf.dequantize(table.data[ids.numpy()].copy(),gguf.GGMLQuantizationType.Q8_0))
out=table.gather(ids,dtype=torch.float32)
assert torch.equal(out,expected),float((out-expected).abs().max())
bf=table.gather(ids,dtype=torch.bfloat16)
assert torch.equal(bf,expected.bfloat16())
for bad in [-1,table.rows]:
 sentinel=torch.full((2,160),123.,dtype=torch.bfloat16)
 try:table.gather(torch.tensor([0,bad]),sentinel);raise AssertionError('Invalid row was accepted')
 except IndexError:pass
 assert torch.all(sentinel==123.)
assert table.gather(torch.empty((0,),dtype=torch.int64)).shape==(0,160)
# Exhaustive FP16 scales, including subnormals/infinities/NaNs, and all signed Q8 values.
scales=np.arange(65536,dtype=np.uint16)
packed=np.empty((65536,34),dtype=np.uint8)
packed[:,:2]=scales.view(np.uint8).reshape(-1,2)
packed[:,2:]=np.resize(np.arange(256,dtype=np.uint8),(65536,32))
p=output/'test-scales.q8';p.write_bytes(packed.tobytes())
s=Q8Rows({'type':'Q8_0','shape':[32,65536],'bytes':packed.nbytes,'offset':0,'file':str(p)})
with np.errstate(invalid='ignore',over='ignore'):
 reference=torch.from_numpy(gguf.dequantize(packed,gguf.GGMLQuantizationType.Q8_0))
for dtype in [torch.float32,torch.bfloat16]:
 a=s.gather(torch.arange(65536),dtype=dtype);b=reference.to(dtype)
 assert torch.all((a==b)|(torch.isnan(a)&torch.isnan(b)))
timing=[]
for tokens in [1,3,32,256]:
 select=ids[:tokens*16] if tokens<32 else torch.tensor(rng.integers(0,table.rows,tokens*16),dtype=torch.int64)
 buf=torch.empty((tokens*16,160),dtype=torch.bfloat16)
 for _ in range(30):table.gather(select,buf)
 t=time.perf_counter()
 for _ in range(1000):table.gather(select,buf)
 timing.append({'tokens':tokens,'microseconds':(time.perf_counter()-t)*1000})
coverage={}
for draft in [False,True]:
 entries=manifest(draft);mapped=[map_name(e['name'],draft) for e in entries]
 coverage['draft' if draft else 'target']={'total':len(entries),'mapped':sum(n is not None for n in mapped),'packed_ple_handled_separately':sum(n is None for n in mapped)}
result={'real_rows_checked':len(ids),'float32_exact':True,'bf16_exact':True,'all_65536_fp16_scales_checked':True,'out_of_bounds_rejected_without_output_write':True,'timing_scope':'Repeated resident row-set, excludes full model and cold disk access','timing':timing,'mapping_coverage':coverage}
(output/'row-tests.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
