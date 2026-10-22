# SPDX-License-Identifier: Apache-2.0
"""Keep autoregressive draft steps at one token under Dynamic SD.

The target and draft-prefill managers verify K+1 tokens and need dynamic
capture lengths. The autoregressive draft manager always processes one token;
subtracting the global maximum K from its query length produces zero/negative
capture sizes. Use a local config copy for that manager's candidate setup only.
"""
import copy,functools

def install():
 from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
 original=CudaGraphManager._init_candidates
 if getattr(original,'_q6_dynamic_fixed',False):return
 @functools.wraps(original)
 def init(self):
  config=self.vllm_config;spec=config.speculative_config
  if (type(self).__name__!='SpeculatorCudaGraphManager' or self.decode_query_len!=1
      or spec is None or not spec.uses_dynamic_speculative_decoding()):
   return original(self)
  local=copy.copy(config);local.speculative_config=copy.copy(spec)
  local.speculative_config.num_speculative_tokens_per_batch_size=None
  self.vllm_config=local
  try:return original(self)
  finally:self.vllm_config=config
 init._q6_dynamic_fixed=True
 CudaGraphManager._init_candidates=init
 print('Q6_DYNAMIC_GRAPH_FIX: autoregressive draft captures retain query length 1',flush=True)
