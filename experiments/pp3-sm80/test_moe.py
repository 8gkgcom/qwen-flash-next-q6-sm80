from pathlib import Path
from kernel_helpers import *
from vllm.model_executor.layers.fused_moe.fused_moe import moe_align_block_size
torch.ops.load_library(str(Path(__file__).resolve().parent/'build/q6_speed_cuda.so'))
new=torch.ops.q6_speed_cuda
results=[]
with torch.inference_mode():
 for layer in [0,17,47]:
  for suffix in ['gate','down']:
   w,e=load(f'blk.{layer}.ffn_{suffix}_exps.weight');E,N,_=w.shape;K=e['shape'][0];q=e['qtype']
   if suffix=='gate':
    up,_=load(f'blk.{layer}.ffn_up_exps.weight');w=torch.cat((w,up),dim=1);del up;N*=2
   for m in [52,128]:
    ids=(torch.arange(m*10,device='cuda')%512).reshape(m,10).int()
    top=10 if suffix=='gate' else 1;tokens=m if top==10 else m*10
    x=torch.randn(tokens,K,device='cuda',dtype=torch.bfloat16)
    base=ops.ggml_moe_a8_vec(x,w,ids,top,q,N,tokens)
    fast=new.vector_moe(x,w,ids,top,q,N,tokens,2)
    aligned=moe_align_block_size(ids,ops.ggml_moe_get_block_size(q),E)
    fixed=new.matrix_moe(x,w,*aligned,q,N,top,tokens)
    rel=float(torch.linalg.vector_norm(fixed.float()-base.float())/torch.linalg.vector_norm(base.float()))
    row=dict(layer=layer,kind=suffix,m=m,all_experts=True,vec_equal=torch.equal(base,fast),matrix_rel_l2=rel,finite=bool(torch.isfinite(fixed).all()))
    print(json.dumps(row),flush=True);results.append(row)
    assert row['vec_equal'] and row['finite'] and rel<0.0005
   del w,x,base,fast,fixed;gc.collect();torch.cuda.empty_cache()
(ROOT/'moe-correctness.json').write_text(json.dumps(results,indent=2))
