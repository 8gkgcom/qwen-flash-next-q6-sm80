"""Q6 PP3 draft-only screening and measured small-shape kernel routing.

Screening follows Qwen-Flash-SM80's approximate-head/original-weight rerank
design. The target model and its verification logits are unchanged.
"""
import json,os,types
from pathlib import Path
import torch
from vllm_gguf_plugin import ops as gguf_ops
import vllm._custom_ops as ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm,marlin_weights
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new,marlin_permute_scales

@torch.no_grad()
def pack_head(weight,bits=4):
    n,packed_k=weight.shape;k=packed_k//34*32
    assert weight.dtype==torch.uint8 and weight.is_contiguous() and packed_k%34==0
    group=128 if bits==4 else 32
    qw=torch.empty((k//16,n*bits//2),device=weight.device,dtype=torch.int32)
    scales=torch.empty((k//group,n),device=weight.device,dtype=torch.bfloat16)
    perm=get_weight_perm(bits)
    for first in range(0,n,4096):
        last=min(first+4096,n);blocks=weight[first:last].view(last-first,k//32,34)
        original_scale=blocks[:,:,:2].contiguous().view(torch.float16).squeeze(-1)
        original_q=blocks[:,:,2:].contiguous().view(torch.int8)
        if bits==4:
            values=(original_q.float()*original_scale.unsqueeze(-1)).reshape(last-first,k//group,group)
            scale=values.abs().amax(-1).div(7).clamp_min(1e-12)
            quant=(values/scale.unsqueeze(-1)).round().clamp(-7,7).reshape(last-first,k).to(torch.int32)+8
        else:
            scale=original_scale;quant=original_q.reshape(last-first,k).to(torch.int32)+128
        qw[:,first*bits//2:last*bits//2].copy_(marlin_weights(quant.T.contiguous(),k,last-first,bits,perm))
        scales[:,first:last].copy_(scale.T)
    return qw,marlin_permute_scales(scales,k,n,group),marlin_make_workspace_new(weight.device,4)

def screened_tokens(hidden,weight,packed,bits=4,top=32):
    qw,scales,workspace=packed;m,k=hidden.shape;n=weight.shape[0]
    approx=ops.marlin_gemm(hidden,None,qw,None,scales,None,None,None,None,None,
        workspace,scalar_types.uint4b8 if bits==4 else scalar_types.uint8b128,m,n,k,
        is_k_full=True,use_atomic_add=False,use_fp32_reduce=True)
    ids=torch.topk(approx,top,dim=-1).indices
    selected=[]
    for row in range(m):
        candidates=ids[row]
        scores=gguf_ops.ggml_mul_mat_vec_a8(weight.index_select(0,candidates),hidden[row:row+1],8,top)[0]
        selected.append(torch.where(scores==scores.max(),candidates,n).min())
    return torch.stack(selected)

def get_top_tokens(self,hidden):
    if not 1<=hidden.shape[0]<=8:return self.compute_logits(hidden).argmax(-1)
    return screened_tokens(hidden,self.lm_head.weight,
        (self.q6_speed_qw,self.q6_speed_scales,self.q6_speed_workspace),self.q6_speed_bits)

def install():
    install_moe()
    if os.environ.get('Q6_SPEED_DRAFT_HEAD','1')!='1':return
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    from vllm.distributed import get_tensor_model_parallel_world_size,get_pp_group
    original=DraftModelSpeculator._validate_local_argmax_reduction
    if getattr(original,'_q6_speed',False):return
    def validate(self):
        bits=int(os.environ.get('Q6_SPEED_HEAD_BITS','4'))
        assert bits in (4,8)
        assert self.speculative_config.num_speculative_tokens==2
        assert self.speculative_config.draft_sample_method!='probabilistic'
        assert get_tensor_model_parallel_world_size()==1
        assert get_pp_group().world_size==3 and get_pp_group().is_last_rank
        model=self.model;weight=model.lm_head.weight
        assert tuple(weight.shape)==(248320,2720) and model.lm_head.weight_type.weight_type==8
        assert model.lm_head.shard_indices.org_vocab_start_index==0
        assert model.lm_head.shard_indices.num_org_vocab_padding==0
        p=model.logits_processor
        assert p.scale==1.0 and p.soft_cap is None and not p.logits_as_input
        packed=pack_head(weight,bits)
        for name,value in zip(('qw','scales','workspace'),packed):model.register_buffer('q6_speed_'+name,value,persistent=False)
        model.q6_speed_bits=bits;model.get_top_tokens=types.MethodType(get_top_tokens,model)
        self.use_local_argmax_reduction=True
        original(self)
        print('Q6_SPEED_DRAFT_HEAD '+json.dumps({'bits':bits,'top':32,'bytes':sum(t.nbytes for t in packed),'target_weights_unchanged':True}),flush=True)
    validate._q6_speed=True
    DraftModelSpeculator._validate_local_argmax_reduction=validate

def install_moe():
    if getattr(gguf_ops.ggml_moe_a8_vec,'_q6_speed',False):return
    torch.ops.load_library(str(Path(__file__).resolve().parent/'build/q6_speed_cuda.so'))
    native_vec=gguf_ops.ggml_moe_a8_vec
    native_mat=gguf_ops.ggml_moe_a8
    def supported(x,w,q):
        return (x.is_cuda and x.dtype==torch.bfloat16 and w.dtype==torch.uint8
                and w.ndim==3 and w.shape[0]==512 and q in (8,14)
                and x.is_contiguous() and w.is_contiguous())
    def vector(x,w,ids,top,q,row,tokens):
        if supported(x,w,q):
            return torch.ops.q6_speed_cuda.vector_moe(x,w,ids,top,q,row,tokens,2)
        return native_vec(x,w,ids,top,q,row,tokens)
    def matrix(x,w,sorted_ids,experts,padded,q,row,top,tokens):
        if supported(x,w,q):
            return torch.ops.q6_speed_cuda.matrix_moe(x,w,sorted_ids,experts,padded,q,row,top,tokens)
        return native_mat(x,w,sorted_ids,experts,padded,q,row,top,tokens)
    vector._q6_speed=True
    gguf_ops.ggml_moe_a8_vec=vector
    gguf_ops.ggml_moe_a8=matrix
    print('Q6_SPEED_MOE experts=512 rows=2 target_quantization=unchanged',flush=True)

