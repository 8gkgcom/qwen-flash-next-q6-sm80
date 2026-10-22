# SPDX-License-Identifier: Apache-2.0
"""Flash Next mapping: inverse of pinned llama.cpp conversion/qwen4exp.py."""
from __future__ import annotations
import json, pathlib, re
import gguf, numpy as np, torch
from vllm_gguf_plugin.weights_adapter.qwen3_5 import QWEN35_ATTN_SUBSTR, QWEN35_MOE_SUBSTR, Qwen35GGUFAdapter, _gdn_value_head_layout
from .q8 import ROOT

HC = {
 'hc_attn_norm': 'attn_hyper_connection.hc_norm',
 'hc_attn_down': 'attn_hyper_connection.input_mix_weight_down',
 'hc_attn_up': 'attn_hyper_connection.input_mix_weight_up',
 'hc_attn_inject': 'attn_hyper_connection.block_inject_weight',
 'hc_ffn_norm': 'mlp_hyper_connection.hc_norm',
 'hc_ffn_down': 'mlp_hyper_connection.input_mix_weight_down',
 'hc_ffn_up': 'mlp_hyper_connection.input_mix_weight_up',
 'hc_ffn_inject': 'mlp_hyper_connection.block_inject_weight',
 'ple_key': 'ple.key_proj', 'ple_value': 'ple.value_proj',
 'ple_norm_key': 'ple.norm_key', 'ple_norm_query':'ple.norm_query',
 'ple_norm_conv':'ple.norm_conv', 'ple_conv1d':'ple.conv1d',
 'indexer.q_norm':'self_attn.indexer.q_layernorm',
 'indexer.k_norm':'self_attn.indexer.k_layernorm',
 'indexer.q_proj':'self_attn.indexer._q_proj',
 'indexer.k_proj':'self_attn.indexer._k_proj',
}

def map_name(raw: str, draft: bool=False):
    roots={'token_embd.weight':'model.embed_tokens.weight', 'output.weight':'lm_head.weight',
      'output_hc_norm.weight':'model.hyper_connection_mixer.hc_norm.weight',
      'output_hc_down.weight':'model.hyper_connection_mixer.input_mix_weight_down.weight',
      'output_hc_up.weight':'model.hyper_connection_mixer.input_mix_weight_up.weight'}
    if raw in roots: return roots[raw]
    if raw=='per_layer_token_embd.weight': return None
    m=re.fullmatch(r'blk\.(\d+)\.(.+)',raw)
    if not m: raise ValueError('Unmapped tensor '+raw)
    i,tail=int(m[1]),m[2]
    if draft:
        if i != 48: raise ValueError('Unexpected MTP block')
        special={'nextn.hc_head_norm.weight':'mtp.hyper_connection_mixer.hc_norm.weight',
          'nextn.hc_head_down.weight':'mtp.hyper_connection_mixer.input_mix_weight_down.weight',
          'nextn.hc_head_up.weight':'mtp.hyper_connection_mixer.input_mix_weight_up.weight',
          'nextn.enorm.weight':'mtp.pre_fc_norm_embedding.weight',
          'nextn.hnorm.weight':'mtp.pre_fc_norm_hidden.weight',
          'nextn.eh_proj.weight':'mtp._eh_proj.weight'}
        if tail in special:return special[tail]
        prefix='mtp.layers.0.'
    else: prefix=f'model.layers.{i}.'
    for old,new in HC.items():
        if tail==old+'.weight':return prefix+new+'.weight'
    for old,new in (QWEN35_ATTN_SUBSTR|QWEN35_MOE_SUBSTR).items():
        if tail.startswith(old):return prefix+new+tail[len(old):]
    raise ValueError('Unmapped tensor '+raw)

def force_dense(name):
    # These runtime modules explicitly ignore quant_config; preserve decoded GGUF values.
    return ('hyper_connection' in name or '.indexer.' in name or name.startswith('mtp.fc_') or 'mtp._eh_proj' in name)

def manifest(draft=False):
    entries=json.loads((ROOT/'gguf-manifest.json').read_text())
    selected=[e for e in entries if ('MTP-draft' in e['file'])==draft and 'mmproj' not in e['file']]
    if len({e['name'] for e in selected})!=len(selected):raise ValueError('Duplicate GGUF tensor names')
    return selected

def iter_weights(model_config, draft=False):
    cfg=model_config.hf_config.get_text_config()
    layout=_gdn_value_head_layout(cfg)
    helper=Qwen35GGUFAdapter();quantized=set(); pending={}
    from vllm.distributed import get_pp_group
    from vllm.distributed.utils import get_pp_indices
    pp=get_pp_group()
    if not draft:start,end=get_pp_indices(cfg.num_hidden_layers,pp.rank_in_group,pp.world_size)
    for entry in manifest(draft):
        name=map_name(entry['name'],draft)
        if name is None:continue
        if not draft:
            match=re.match(r'model.layers\.(\d+)\.',name)
            if match and not start<=int(match[1])<end:continue
            # This Flash Next implementation keeps vocab modules on both PP ranks.
            if name.startswith('model.hyper_connection_mixer.') and not pp.is_last_rank:continue
        qtype=gguf.GGMLQuantizationType(entry['qtype'])
        packed=qtype not in (gguf.GGMLQuantizationType.F32,gguf.GGMLQuantizationType.F16,gguf.GGMLQuantizationType.BF16)
        # Read one tensor at a time, never read or expand the PLE through this iterator.
        if packed:
            shape=tuple(reversed(entry['shape']))
            block,size=gguf.GGML_QUANT_SIZES[qtype]
            array=np.memmap(entry['file'],mode='r',dtype=np.uint8,offset=entry['offset'],shape=(*shape[:-1],shape[-1]//block*size))
            if force_dense(name):
                weight=torch.from_numpy(gguf.dequantize(array,qtype).copy())
                packed=False
            else: weight=torch.from_numpy(array.copy())
        else:
            dtype={gguf.GGMLQuantizationType.F32:np.float32,gguf.GGMLQuantizationType.F16:np.float16,gguf.GGMLQuantizationType.BF16:np.uint16}[qtype]
            array=np.memmap(entry['file'],mode='r',dtype=dtype,offset=entry['offset'],shape=tuple(reversed(entry['shape'])))
            weight=torch.from_numpy(array.copy())
            if qtype==gguf.GGMLQuantizationType.BF16:weight=weight.view(torch.bfloat16)
        if packed:quantized.add(name.removesuffix('.weight'))
        if layout is not None:
            restored=helper._restore_gdn_weight(name,weight,cfg,layout,quantized)
            if restored is not None:weight=restored
        elif name.endswith('.A_log'):weight=torch.log(-weight)
        if name.endswith(('norm.weight','pre_fc_norm_embedding.weight','pre_fc_norm_hidden.weight')) and not name.endswith('linear_attn.norm.weight'):
            weight=weight-1
        elif name.endswith(('.ple.norm_key.weight','.ple.norm_query.weight','.ple.norm_conv.weight','.indexer.q_layernorm.weight','.indexer.k_layernorm.weight')):
            weight=weight-1
        if 'conv1d.weight' in name and weight.ndim==2:weight=weight.unsqueeze(1)
        if name.endswith('mlp.shared_expert_gate.weight') and weight.ndim==1:weight=weight.unsqueeze(0)
        if '._q_proj.' in name or '._k_proj.' in name:
            base=name.rsplit('.',2)[0];pending[name]=weight
            q=pending.get(base+'._q_proj.weight');k=pending.get(base+'._k_proj.weight')
            if q is None or k is None:continue
            yield base+'.index_qk_proj.weight',torch.cat([q,k],dim=0)
            del pending[base+'._q_proj.weight'],pending[base+'._k_proj.weight']
            continue
        if name=='mtp._eh_proj.weight':
            e,h=weight.split(cfg.hidden_size,dim=1)
            yield 'mtp.fc_embedding.weight',e.contiguous()
            yield 'mtp.fc_hidden.weight',h.contiguous()
            continue
        if packed:yield name.removesuffix('.weight')+'.weight_type',torch.tensor(int(qtype))
        if weight.ndim==3 and '.experts.0.' in name:
            for i,w in enumerate(weight.unbind()):yield name.replace('.experts.0.',f'.experts.{i}.'),w
        else:yield name,weight
    if pending:raise RuntimeError('Incomplete indexer pairs: '+str(list(pending)))

def configure(quant_config,model_config,prefix=''):
    cfg=model_config.hf_config.get_text_config();draft='MTP' in model_config.hf_config.architectures[0]
    layout=_gdn_value_head_layout(cfg)
    if draft:
        quant_config.unquantized_modules.extend(['mtp.fc_embedding','mtp.fc_hidden'])
    for e in manifest(draft):
        n=map_name(e['name'],draft)
        if not n:continue
        if draft:n=n.replace('mtp.layers.0.',f'mtp.layers.{cfg.num_hidden_layers}.')
        if e['type'] in ('F32','F16','BF16') or force_dense(n):
            quant_config.unquantized_modules.append(n.removesuffix('.weight'))
            if '._q_proj.' in n or '._k_proj.' in n:
                # The standalone MTP class does not declare this fused mapping.
                quant_config.unquantized_modules.append(n.rsplit('.',2)[0]+'.index_qk_proj')
        if layout is not None and n.endswith('linear_attn.out_proj.weight'):
            quant_config.linear_layouts[n.removesuffix('.weight')]=layout
