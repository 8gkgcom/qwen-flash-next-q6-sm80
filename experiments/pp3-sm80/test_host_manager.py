import json
from pathlib import Path
from q6_ram_kv import Q6CPUManager
from vllm.v1.kv_offload.base import LookupResult
m=Q6CPUManager(num_blocks=2)
a,b,c=b'a',b'b',b'c'
r=m.prepare_store([a,b],None);assert len(r.keys_to_store)==2
assert m.prepare_store([c],None) is None # no in-flight writes may be evicted
m.complete_store([a,b],None)
m.prepare_load([a],None)
r=m.prepare_store([c],None);assert r.evicted_keys==[b] # pinned read survives LRU
m.complete_store([c],None);assert m.lookup(b,None)==LookupResult.MISS
assert m.lookup(a,None)==LookupResult.HIT
m.complete_load([a],None)
assert m._num_allocated_blocks==2
m.reset_cache();assert m._num_allocated_blocks==0
r=m.prepare_store([a],None);m.complete_store([a],None,success=False)
assert m.lookup(a,None)==LookupResult.MISS
print('PASS: bounded pool, pinned transfers, LRU eviction, reset, failed-write cleanup')
