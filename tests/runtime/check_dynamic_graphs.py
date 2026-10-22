import json,pathlib,sys
from types import SimpleNamespace
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.config.compilation import CUDAGraphMode
class Spec:
 num_speculative_tokens_per_batch_size=[[1,1,2],[2,2,1]]
 def uses_dynamic_speculative_decoding(self):return self.num_speculative_tokens_per_batch_size is not None
class SpeculatorCudaGraphManager(CudaGraphManager):pass
def make(cls,length):
 obj=cls.__new__(cls);obj.compilation_config=SimpleNamespace(cudagraph_capture_sizes=list(range(1,7)),max_cudagraph_capture_size=6)
 obj.cudagraph_mode=CUDAGraphMode.FULL_DECODE_ONLY;obj.max_num_reqs=2;obj.decode_query_len=length
 obj.vllm_config=SimpleNamespace(speculative_config=Spec(),num_speculative_tokens=2)
 obj.varlen_decode=False;obj.lora_capture_cases=[0];obj._candidates={};obj._capture_descs={}
 return obj
before=make(SpeculatorCudaGraphManager,1)
try:before._init_candidates()
except ZeroDivisionError:pass
else:raise AssertionError('Original zero-length failure was not reproduced')
from flashnext_gguf.optimizations.q6_dynamic_graphs import install
install();results=[]
for cls,length,want in [(SpeculatorCudaGraphManager,1,{1}),(SpeculatorCudaGraphManager,3,{2,3}),(CudaGraphManager,3,{2,3})]:
 obj=make(cls,length);config=obj.vllm_config;obj._init_candidates()
 descs=[d for group in obj._capture_descs.values() for d in group]
 actual={d.uniform_token_count for d in descs};assert actual==want,(actual,want)
 assert obj.vllm_config is config and config.speculative_config.num_speculative_tokens_per_batch_size==[[1,1,2],[2,2,1]]
 assert all(d.num_tokens>0 and d.num_reqs>0 for d in descs)
 results.append({'manager':cls.__name__,'query_capacity':length,'captured_query_lengths':sorted(actual),'capture_shapes':[(d.num_reqs,d.num_tokens) for d in descs]})
print(json.dumps(results))
