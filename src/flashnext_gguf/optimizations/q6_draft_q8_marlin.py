# SPDX-License-Identifier: Apache-2.0
"""Experimental Q8_0 draft vocabulary screening; target weights stay unchanged."""
import torch
import json, os, time, types
from pathlib import Path
import vllm._custom_ops as ops
from vllm_gguf_plugin import ops as gguf_ops
from vllm.scalar_type import scalar_types
from vllm.model_executor.layers.quantization.utils.marlin_utils_test import get_weight_perm, marlin_weights
from vllm.model_executor.layers.quantization.utils.marlin_utils import marlin_make_workspace_new, marlin_permute_scales

GROUP = 32

@torch.no_grad()
def pack(weight):
    n, packed_k = weight.shape
    k = packed_k // 34 * 32
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    assert packed_k % 34 == 0 and n % 64 == 0 and k % GROUP == 0
    ptr = weight.data_ptr()
    qw = torch.empty((k // 16, n * 4), device=weight.device, dtype=torch.int32)
    scales = torch.empty((k // GROUP, n), device=weight.device, dtype=torch.bfloat16)
    perm = get_weight_perm(8)
    for first in range(0, n, 4096):
        last = min(first + 4096, n)
        blocks = weight[first:last].view(last-first, k//32, 34)
        scale = blocks[:, :, :2].contiguous().view(torch.float16).squeeze(-1)
        quant = blocks[:, :, 2:].contiguous().view(torch.int8).reshape(last-first, k)
        packed = marlin_weights(quant.T.contiguous().to(torch.int32)+128, k, last-first, 8, perm)
        qw[:, first*4:last*4].copy_(packed)
        scales[:, first:last].copy_(scale.T)
    scales = marlin_permute_scales(scales, k, n, GROUP)
    assert weight.data_ptr() == ptr
    return qw, scales, marlin_make_workspace_new(weight.device, 4)

def approximate(hidden, qw, scales, workspace):
    m, k = hidden.shape
    n = scales.shape[1]
    return ops.marlin_gemm(hidden, None, qw, None, scales, None, None, None, None, None,
        workspace, scalar_types.uint8b128, m, n, k,
        is_k_full=True, use_atomic_add=False, use_fp32_reduce=True)

def select(hidden, weight, qw, scales, workspace, top=32):
    n = weight.shape[0]
    ids = torch.topk(approximate(hidden, qw, scales, workspace), top, dim=-1).indices
    chosen = []
    for row in range(hidden.shape[0]):
        candidates = ids[row]
        logits = gguf_ops.ggml_mul_mat_vec_a8(weight.index_select(0, candidates), hidden[row:row+1], 8, top)[0]
        chosen.append(torch.where(logits == logits.max(), candidates, n).min())
    return torch.stack(chosen)

def get_top_tokens(self, hidden):
    if not 1 <= hidden.shape[0] <= 8:
        return self.compute_logits(hidden).argmax(-1)
    return select(hidden, self.lm_head.weight, self.q6_draft_qw,
                  self.q6_draft_scales, self.q6_draft_workspace)

def install():
    if os.environ.get('Q6_DRAFT_Q8_MARLIN') != '1':
        return
    from vllm.distributed import get_tensor_model_parallel_world_size, get_pp_group
    from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator
    if getattr(DraftModelSpeculator, '_q6_draft_q8_marlin_installed', False):
        return
    original = DraftModelSpeculator._validate_local_argmax_reduction
    def validate(self):
        assert self.speculative_config.draft_sample_method != 'probabilistic'
        assert get_tensor_model_parallel_world_size() == 1
        assert get_pp_group().world_size == 2 and get_pp_group().is_last_rank
        model = self.model
        assert 'qwen3_8_flash_next' in type(model).__module__
        weight = model.lm_head.weight
        assert tuple(weight.shape) == (248320, 2720)
        assert model.lm_head.weight_type.weight_type == 8
        assert model.lm_head.shard_indices.org_vocab_start_index == 0
        assert model.lm_head.shard_indices.num_org_vocab_padding == 0
        p = model.logits_processor
        assert p.scale == 1.0 and p.soft_cap is None and not p.logits_as_input
        assert p.head_dtype in (None, torch.bfloat16)
        start = time.perf_counter();ptr = weight.data_ptr()
        with torch.no_grad():
            qw, scales, workspace = pack(weight)
        model.register_buffer('q6_draft_qw', qw, persistent=False)
        model.register_buffer('q6_draft_scales', scales, persistent=False)
        model.register_buffer('q6_draft_workspace', workspace, persistent=False)
        model.get_top_tokens = types.MethodType(get_top_tokens, model)
        self.use_local_argmax_reduction = True
        original(self)
        assert weight.data_ptr() == ptr
        torch.cuda.synchronize(weight.device)
        record = dict(group=32, top=32, packed_bytes=qw.nbytes+scales.nbytes+workspace.nbytes,
                      setup_seconds=time.perf_counter()-start, original_q8_unchanged=True)
        evidence = os.environ.get('Q6_DRAFT_Q8_EVIDENCE')
        if evidence:Path(evidence).write_text(json.dumps(record, indent=2))
        print('Q6_DRAFT_Q8_MARLIN_READY '+json.dumps(record), flush=True)
    DraftModelSpeculator._validate_local_argmax_reduction = validate
    DraftModelSpeculator._q6_draft_q8_marlin_installed = True
