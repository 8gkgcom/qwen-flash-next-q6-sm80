# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import torch
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import initialize_model, process_weights_after_loading
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm_gguf_plugin.quantization import recursive_replace_vocab_modules
from .adapter import configure, iter_weights

class FlashNextLoader(BaseModelLoader):
    def download_model(self,model_config): pass

    def load_weights(self,model,model_config):
        draft='MTP' in model_config.hf_config.architectures[0]
        loaded=model.load_weights(iter_weights(model_config,draft))
        # Missing parameter validation is kept explicit: reject a partially loaded network.
        if loaded is not None:
            expected={n.replace('.routed_experts.','.') for n,p in model.named_parameters() if not getattr(p,'is_meta',False)}
            missing=expected-{n.replace('.routed_experts.','.') for n in loaded}
            # Derived packed GGUF weight-type metadata is materialized by the plugin.
            missing={n for n in missing if not n.endswith('weight_type')}
            if missing:raise RuntimeError('Unloaded parameters: '+str(sorted(missing)[:40]))
            for n,p in model.named_parameters():
                if n.endswith(('w13_weight','w2_weight')):
                    seen=getattr(p,'_gguf_loaded_expert_shards',set())
                    expected_count=p.shape[0]*(2 if n.endswith('w13_weight') else 1)
                    if len(seen)!=expected_count:raise RuntimeError(f'Incomplete experts: {n}: {len(seen)}/{expected_count}')
            print(f'FLASHNEXT: verified {len(expected)} materialized parameters including all expert shards',flush=True)

    def load_model(self,vllm_config,model_config,prefix=''):
        if vllm_config.parallel_config.tensor_parallel_size != 1:
            raise ValueError('This experimental Flash Next GGUF adapter is validated for TP=1 only')
        configure(vllm_config.quant_config,model_config,prefix)
        target=torch.device(vllm_config.device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target:
                model=initialize_model(vllm_config=vllm_config,model_config=model_config,prefix=prefix)
                recursive_replace_vocab_modules(model,vllm_config.quant_config,prefix)
            self.load_weights(model,model_config)
            process_weights_after_loading(model,model_config,target)
            if 'MTP' in model_config.hf_config.architectures[0]:
                # The MTP file includes its own Q8 vocab; the target vocab is Q6.
                model.has_own_embed_tokens=True
                model.has_own_lm_head=True
        return model

def load_ple_only(runner,model,offload_layers):
    # Packed table was opened by Q8PLEEmbedding in the CPU-only subtree.
    # Load exact int64 hash constants from GGUF metadata, not from a float conversion.
    import json
    from .q8 import ROOT,Q8PLEEmbedding
    metadata=json.loads((ROOT/'metadata.json').read_text())
    fields={'layer_multipliers':'qwen4exp.ple.layer_multipliers',
            'ngram_heads_offsets':'qwen4exp.ple.head_offsets',
            'ngram_heads_vocab_sizes':'qwen4exp.ple.head_vocab_sizes'}
    for name,layer in offload_layers.items():
        if not isinstance(layer.ngram_embedding,Q8PLEEmbedding):raise TypeError('Unexpected PLE module '+name)
        for attr,key in fields.items():
            original=getattr(layer,attr)
            expected=torch.tensor(metadata[key],dtype=torch.int64,device='cpu')
            if original.shape!=expected.shape:raise ValueError('Hash constant shape mismatch: '+attr)
            original.copy_(expected)
