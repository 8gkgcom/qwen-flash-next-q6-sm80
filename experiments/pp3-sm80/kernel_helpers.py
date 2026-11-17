import gc,json,time
from pathlib import Path
import numpy as np
import torch
from vllm_gguf_plugin import ops
from vllm_gguf_plugin.triton.gemm.interface import ggml_mul_mat_a8_triton


import os
ROOT=Path(os.environ.get('Q6_TEST_OUTPUT','./test-output'))
ROOT.mkdir(parents=True,exist_ok=True)
manifest=json.loads(Path(os.environ['Q6_GGUF_MANIFEST']).read_text())
torch.set_num_threads(1);torch.manual_seed(170)
def load(name,draft=False):
    e=next(e for e in manifest if e['name']==name and ('MTP-draft' in e['file'])==draft)
    block,size={14:(256,210),8:(32,34)}[e['qtype']]
    shape=list(reversed(e['shape']));shape[-1]=shape[-1]//block*size
    a=np.memmap(e['file'],mode='r',dtype=np.uint8,offset=e['offset'],shape=tuple(shape))
    return torch.from_numpy(a.copy()).cuda(),e
def bench(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(12):y=fn()
    torch.cuda.current_stream().wait_stream(stream)
    times=[]
    for _ in range(3):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(8):graph.replay()
        b.record();b.synchronize();times.append(a.elapsed_time(b)*1000/96)
    return sorted(times)[1]

