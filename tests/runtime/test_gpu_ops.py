import json, os, pathlib, sys, tempfile
root=pathlib.Path(os.environ.get('FLASHNEXT_GGUF_ROOT','/data/runtime'))
output=pathlib.Path(tempfile.mkdtemp(prefix='q6-runtime-test-'))
from require_idle import require_idle
require_idle()
os.environ['CUDA_LAUNCH_BLOCKING']='1';os.environ['VLLM_PLUGINS']='flashnext_gguf';os.environ['TRITON_CACHE_DIR']=str(root/'cache/triton')
import torch,numpy as np,gguf
from flashnext_gguf import register
register()
from vllm_gguf_plugin.quantization.linear import _fused_mul_mat_gguf
from vllm_gguf_plugin.quantization.fused_moe import _fused_moe_gguf
torch.set_num_threads(1);torch.manual_seed(20261022)
entries=[e for e in json.loads((root/'gguf-manifest.json').read_text()) if 'MTP-draft' not in e['file'] and 'mmproj' not in e['file']]
def read(e,experts=None):
 shape=list(reversed(e['shape']));block,size=gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(e['qtype'])];shape[-1]=shape[-1]//block*size
 a=np.memmap(e['file'],mode='r',dtype=np.uint8,offset=e['offset'],shape=tuple(shape))
 return np.array(a if experts is None else a[:experts])
results=[]
for suffix in ['attn_q.weight','attn_output.weight','ffn_gate_shexp.weight','ffn_down_shexp.weight']:
 e=next(e for e in entries if e['name'].endswith(suffix));a=read(e);w=torch.from_numpy(a).cuda();dense=torch.from_numpy(gguf.dequantize(a,gguf.GGMLQuantizationType(e['qtype']))).cuda().float()
 for m in [1,2,3,4,64,128]:
  x=torch.randn(m,e['shape'][0],device='cuda',dtype=torch.bfloat16)
  print('linear',e['name'],m,flush=True);y=_fused_mul_mat_gguf(x,w,e['qtype']);torch.cuda.synchronize()
  ref=x.float()@dense.T;err=float((y.float()-ref).norm()/ref.norm());assert err<.04,err
  results.append({'op':e['name'],'m':m,'relative_l2':err})
 del w,dense,a
e1=next(e for e in entries if e['name']=='blk.0.ffn_gate_exps.weight');e3=next(e for e in entries if e['name']=='blk.0.ffn_up_exps.weight');e2=next(e for e in entries if e['name']=='blk.0.ffn_down_exps.weight')
ag,au,ad=read(e1,16),read(e3,16),read(e2,16)
w1=torch.from_numpy(np.concatenate([ag,au],axis=1)).cuda();w2=torch.from_numpy(ad).cuda()
dg,du,dd=[torch.from_numpy(gguf.dequantize(a,gguf.GGMLQuantizationType(e['qtype']))).cuda().float() for a,e in [(ag,e1),(au,e3),(ad,e2)]]
for m in [1,2,3,4,64,128]:
 x=torch.randn(m,2560,device='cuda',dtype=torch.bfloat16);ids=torch.randint(0,16,(m,10),device='cuda',dtype=torch.int32);weights=torch.softmax(torch.randn(m,10,device='cuda'),-1)
 print('moe',m,flush=True);y=_fused_moe_gguf(x,w1,w2,weights,ids,e1['qtype'],e2['qtype'],'silu');torch.cuda.synchronize()
 ref=torch.zeros_like(y,dtype=torch.float32)
 for k in range(10):
  ii=ids[:,k].long();g=torch.bmm(dg[ii],x.float().unsqueeze(2)).squeeze(2);u=torch.bmm(du[ii],x.float().unsqueeze(2)).squeeze(2)
  ref+=torch.bmm(dd[ii],(torch.nn.functional.silu(g)*u).unsqueeze(2)).squeeze(2)*weights[:,k,None]
 err=float((y.float()-ref).norm()/ref.norm());assert err<.08,err
 results.append({'op':'moe','m':m,'relative_l2':err})
(output/'gpu-ops-tests.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
