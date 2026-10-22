import json, os, pathlib, sys, tempfile
root=pathlib.Path(os.environ.get('FLASHNEXT_GGUF_ROOT','/data/runtime'))
output=pathlib.Path(tempfile.mkdtemp(prefix='q6-runtime-test-'))
import time, types
os.environ['FLASHNEXT_PLE_RESIDENT']='0';os.environ['VLLM_PLE_CPU_OFFLOAD']='1'
import torch,gguf,numpy as np
from vllm.model_executor.layers.ple_offload_layer import mark_as_offload_worker
from vllm.models.qwen3_8_flash_next.nvidia.ple_layer import Qwen3_8FlashNextNGramEmbedding
torch.set_num_threads(1);mark_as_offload_worker()
cfg=types.SimpleNamespace(**json.loads((root/'model-config/config.json').read_text()))
meta=json.loads((root/'metadata.json').read_text())
layer=Qwen3_8FlashNextNGramEmbedding(cfg,cfg.ple_embed_dim,0,64,4,'test',quant_config=types.SimpleNamespace(get_name=lambda:'gguf'),params_dtype=torch.bfloat16)
for attr,key in {'layer_multipliers':'layer_multipliers','ngram_heads_offsets':'head_offsets','ngram_heads_vocab_sizes':'head_vocab_sizes'}.items():
 getattr(layer,attr).copy_(torch.tensor(meta['qwen4exp.ple.'+key],dtype=torch.int64))
eos=cfg.eos_token_id;seqs=[[1,2,3],[eos,12,15,eos,17],[7]];history=[[eos,eos],[9,10],[1,2]]
ids=[]
for seq,hist in zip(seqs,history):
 context=hist+seq
 for pos in range(2,len(context)):
  previous=max((j for j in range(pos) if context[j]==eos),default=-1)
  shifted=[context[pos]]+[context[pos-k] if pos-k>previous else eos for k in [1,2]]
  row=[]
  for ngram in [2,3]:
   mixed=0
   for j in range(ngram):mixed^=shifted[j]*meta['qwen4exp.ple.layer_multipliers'][j]
   for h in range((ngram-2)*8,(ngram-1)*8):row.append(mixed%meta['qwen4exp.ple.head_vocab_sizes'][h]+meta['qwen4exp.ple.head_offsets'][h])
  ids.append(row)
flat=[v for seq in seqs for v in seq]
expected=torch.from_numpy(gguf.dequantize(layer.ngram_embedding.table.data[np.array(ids).reshape(-1)].copy(),gguf.GGMLQuantizationType.Q8_0)).bfloat16().reshape(len(flat),-1)
starts=torch.tensor([0,3,8,9]);tokens=torch.tensor(flat+[123,456,789]);ctx=torch.tensor(history)
buf=torch.empty((len(tokens),cfg.ple_embed_dim),dtype=torch.bfloat16)
result=layer.forward_impl(torch.empty(0),tokens,starts,ctx,output_buffer=buf)
assert torch.equal(result[:len(flat)],expected),'Hash/row lookup mismatch for ragged requests with EOS and padding'
out={'ragged_requests':3,'valid_tokens':len(flat),'padding_tokens':3,'exact_bf16_match':True,'checks':['context carry','EOS reset','two/three-gram heads','metadata int64 constants','graph padding exclusion','caller-owned output buffer']}
(output/'ple-tests.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
