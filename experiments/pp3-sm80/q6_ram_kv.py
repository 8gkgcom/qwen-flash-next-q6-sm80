"""Q6 PP3 bounded host KV cache, validated with MTP2 and four requests.

Only complete compression-aligned checkpoints are transferable. QSA's raw
8-token scratch ring is reconstructed by the next compression group, as on a
native GPU prefix-cache hit. The engine's original cache layout is untouched.
"""
from dataclasses import replace
import json
import os
from pathlib import Path

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import OffloadingConnector
from vllm.v1.kv_cache_interface import CircularBufferSpec, UniformTypeKVCacheSpecs
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager


def record(kind, **fields):
    destination = os.environ.get('Q6_RAM_KV_EVIDENCE')
    if not destination:
        return
    fields.pop('request', None)
    path = Path(destination)
    path.mkdir(parents=True, exist_ok=True)
    with (path / f'{os.getpid()}.jsonl').open('a') as f:
        f.write(json.dumps(dict(kind=kind, **fields)) + '\n')


def install_prefill_guard():
    """Keep PP3/MTP prefill batches to one request; decode still batches four.

    The Q6 PP loader already separates prefill from decode. Multiple independent
    prefills in one step also corrupt the first token in mixed-length tests.
    Use scheduler eligibility/admission gates without changing tokens or blocks.
    """
    import functools
    from vllm.v1.core.sched.scheduler import Scheduler
    original = Scheduler.schedule
    if getattr(original, '_q6_single_prefill', False):
        return
    original_preempt = Scheduler._preempt_request
    @functools.wraps(original_preempt)
    def preempt(self, request, timestamp, drop_stale_output=False):
        # PP/MTP may still have uncommitted samples in flight. Once this KV
        # generation is invalidated, discard those samples and regenerate them
        # from committed history after restore (native reset uses this path).
        record('preempt_boundary', request=request.request_id,
               committed_output_tokens=request.num_output_tokens,
               in_flight=request.num_in_flight_tokens)
        return original_preempt(self, request, timestamp, drop_stale_output=True)
    Scheduler._preempt_request = preempt
    @functools.wraps(original)
    def schedule(self, throttle_prefills=False):
        prefills = [r for r in self.running if r.is_prefill_chunk]
        saved = [(r, r.next_decode_eligible_step) for r in prefills[1:]]
        capacity, limit = self.prefill_capacity_bound, self.max_num_running_reqs
        blockers = getattr(self, '_q6_pressure_blockers', set())
        if blockers and (not self.running or any(
                rid not in self.requests or self.requests[rid].is_finished()
                for rid in blockers)):
            self._q6_pressure_blockers = blockers = set()
            record('pressure_release')
        for r, step in saved:
            r.next_decode_eligible_step = max(step, self.current_step + 2)
        self.prefill_capacity_bound = False
        self.max_num_running_reqs = min(limit, len(self.running) + int(not prefills and not blockers))
        try:
            result = original(self, throttle_prefills=throttle_prefills)
            if result.preempted_req_ids and not blockers:
                self._q6_pressure_blockers = {r.request_id for r in self.running}
                record('pressure_hold', preempted=len(result.preempted_req_ids),
                       running=len(self.running))
            return result
        finally:
            self.prefill_capacity_bound, self.max_num_running_reqs = capacity, limit
            for r, step in saved:
                r.next_decode_eligible_step = step
    schedule._q6_single_prefill = True
    Scheduler.schedule = schedule
    record('single_prefill_guard_installed')


class Q6CPUManager(CPUOffloadingManager):
    def prepare_store(self, keys, req_context):
        result = super().prepare_store(keys, req_context)
        if result is not None and result.keys_to_store:
            record('host_store', blocks=len(result.keys_to_store),
                   evicted=len(result.evicted_keys), capacity=self._num_blocks,
                   allocated=self._num_allocated_blocks-len(self._free_list))
        return result


def install_pp_state_tables():
    """Use request-slot tables when processing delayed PP sampling results."""
    import copy
    import functools
    import torch
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    original = GPUModelRunner.postprocess_sampled
    if getattr(original, '_q6_pp_state_tables', False):
        return

    @functools.wraps(original)
    def postprocess(self, idx_mapping, *args, **kwargs):
        state = self.model_state
        ctx = getattr(state, '_mamba_ctx', None)
        if self.is_last_pp_rank or ctx is None or not ctx.is_initialized:
            return original(self, idx_mapping, *args, **kwargs)
        if not hasattr(self, '_q6_post_ctx'):
            sources = [self.block_tables.block_tables[g].gpu
                       for g in ctx.mamba_group_ids]
            tables = [torch.empty_like(t) for t in sources]
            post_ctx = copy.copy(ctx)
            pointers = torch.tensor([t.data_ptr() for t in tables],
                                    dtype=torch.int64, pin_memory=True)
            post_ctx.block_table_ptrs = pointers.to(self.device, non_blocking=True)
            strides = {t.stride(0) for t in tables}
            if len(strides) != 1:
                raise RuntimeError('Inconsistent PP postprocess table strides')
            post_ctx.block_table_stride_req = strides.pop()
            self._q6_post_ctx = post_ctx
            self._q6_post_tables = tables
            self._q6_post_sources = sources
            record('pp_state_tables_installed')
        safe_indices = idx_mapping.clamp_min(0).long()
        for source, table in zip(self._q6_post_sources, self._q6_post_tables):
            torch.index_select(source, 0, safe_indices,
                               out=table[:idx_mapping.shape[0]])
        state._mamba_ctx = self._q6_post_ctx
        try:
            return original(self, idx_mapping, *args, **kwargs)
        finally:
            state._mamba_ctx = ctx

    postprocess._q6_pp_state_tables = True
    GPUModelRunner.postprocess_sampled = postprocess


class Q6CPUOffloadingSpec(CPUOffloadingSpec):
    """Identical slot counts across unequal PP stages, separate host buffers."""
    def __init__(self, config):
        super().__init__(config)
        self.num_blocks = int(self.extra_config['cpu_num_blocks'])
        if self.num_blocks < 1 or self.blocks_per_chunk != 1:
            raise ValueError('Q6 host cache needs positive slots and single-block chunks')

    def _uses_shared_region(self):
        # Device ordinal can be zero in all PP processes; PP layer sizes differ.
        # Private buffers avoid aliasing and retain one globally agreed slot map.
        return False

    def get_manager(self):
        if self._manager is None:
            self._manager = Q6CPUManager(num_blocks=self.num_blocks,
                cache_policy='lru', enable_events=self.kv_events_config.enable_kv_cache_events)
        return self._manager

    def create_worker(self, kv_caches):
        allocated = sum(t.page_size_bytes for t in kv_caches.tensors) * self.num_blocks
        limit = int(self.extra_config['cpu_bytes_to_use']) // self.config.parallel.world_size
        if allocated > limit:
            raise ValueError(f'Q6 host KV allocation {allocated} exceeds per-rank budget {limit}')
        record('host_allocation', bytes=allocated, limit=limit, slots=self.num_blocks)
        return super().create_worker(kv_caches)


class Q6RAMKVConnector(OffloadingConnector):
    @property
    def prefer_cross_layer_blocks(self):
        return False

    def __init__(self, vllm_config, role, kv_cache_config):
        self._indices = []
        excluded = []
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            spec = group.kv_cache_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.first_spec
            if isinstance(spec, CircularBufferSpec):
                if spec.block_size != 8 or spec.prefix_cacheable:
                    raise ValueError('Unvalidated QSA scratch-ring geometry')
                excluded.extend(group.layer_names)
            else:
                if spec.block_size != 3456:
                    raise ValueError(f'Unvalidated Q6 checkpoint size: {spec.block_size}')
                self._indices.append(i)
        groups = [kv_cache_config.kv_cache_groups[i] for i in self._indices]
        self._names = {name for group in groups for name in group.layer_names}
        tensors = [replace(t, shared_by=[n for n in t.shared_by if n in self._names])
                   for t in kv_cache_config.kv_cache_tensors
                   if any(n in self._names for n in t.shared_by)]
        projected = replace(kv_cache_config, kv_cache_groups=groups, kv_cache_tensors=tensors)
        record('layout', role=str(role), retained=self._indices, excluded=excluded,
               groups=[dict(names=g.layer_names, spec=type(g.kv_cache_spec).__name__,
                            page=g.kv_cache_spec.page_size_bytes) for g in groups])
        if role == KVConnectorRole.SCHEDULER:
            projected = replace(projected, kv_cache_groups=[
                replace(g, kv_cache_spec=g.kv_cache_spec.first_spec)
                if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else g
                for g in groups])
        super().__init__(vllm_config, role, projected)
        if self.connector_scheduler is not None:
            install_prefill_guard()
            # Partial checkpoints require scratch-ring snapshots, unsupported.
            self.connector_scheduler.config = self.connector_scheduler.config._replace(
                supports_partial_tail=False)

    def _project(self, blocks):
        if blocks is None:
            return None
        return tuple(blocks[i] for i in self._indices)

    def register_kv_caches(self, kv_caches):
        install_pp_state_tables()
        return super().register_kv_caches({k: v for k, v in kv_caches.items() if k in self._names})

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return super().update_state_after_alloc(
            request, replace(blocks, blocks=self._project(blocks.blocks)), num_external_tokens)

    def build_connector_meta(self, scheduler_output):
        cached = scheduler_output.scheduled_cached_reqs
        projected = replace(scheduler_output,
            scheduled_new_reqs=[replace(r, block_ids=self._project(r.block_ids))
                                for r in scheduler_output.scheduled_new_reqs],
            scheduled_cached_reqs=replace(cached,
                new_block_ids=[self._project(b) for b in cached.new_block_ids]),
            partial_tail_offloads={})
        return super().build_connector_meta(projected)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        result = super().get_num_new_matched_tokens(request, num_computed_tokens)
        count, asynchronous = result
        if count and count > 0:
            if (num_computed_tokens + count) % 3456:
                raise RuntimeError('Refusing an unaligned QSA host checkpoint')
            record('host_hit', request=request.request_id, tokens=count,
                   preemptions=request.num_preemptions)
        return result
