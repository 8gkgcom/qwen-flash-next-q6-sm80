# SPDX-License-Identifier: Apache-2.0
"""Separate prefill and decode steps on the pinned PP2 GGUF runtime.

Pure decode still batches both requests and uses native Dynamic SD. Alternate
eligible decode steps with prefill-only steps via native eligibility/deferral
gates. Do not rewrite tokens, accepted counts, or allocated KV blocks.
"""
import functools

def schedule_guard(original, scheduler, throttle_prefills=False):
    if not scheduler.scheduler_config.async_scheduling or not scheduler.num_spec_tokens:
        return original(scheduler, throttle_prefills=throttle_prefills)
    pending = (any(r.is_prefill_chunk for r in scheduler.running)
               or bool(scheduler.waiting) or bool(scheduler.skipped_waiting))
    decoders = [r for r in scheduler.running if not r.is_prefill_chunk]
    if not pending or not decoders:
        scheduler._q6_next_phase = 'prefill'
        return original(scheduler, throttle_prefills=throttle_prefills)
    eligible = any(r.next_decode_eligible_step <= scheduler.current_step + 1 for r in decoders)
    phase = getattr(scheduler, '_q6_next_phase', 'prefill') if eligible else 'prefill'
    scheduler._q6_next_phase = 'decode' if phase == 'prefill' else 'prefill'
    if phase == 'decode':
        capacity, limit = scheduler.prefill_capacity_bound, scheduler.max_num_running_reqs
        scheduler.prefill_capacity_bound = False
        scheduler.max_num_running_reqs = len(scheduler.running)
        try:
            return original(scheduler, throttle_prefills=True)
        finally:
            scheduler.prefill_capacity_bound, scheduler.max_num_running_reqs = capacity, limit
    saved = [(r, r.next_decode_eligible_step) for r in decoders]
    for request, step in saved:
        request.next_decode_eligible_step = max(step, scheduler.current_step + 2)
    try:
        return original(scheduler, throttle_prefills=throttle_prefills)
    finally:
        for request, step in saved:
            request.next_decode_eligible_step = step

def install():
    from vllm.v1.core.sched.scheduler import Scheduler
    original = Scheduler.schedule
    if getattr(original, '_q6_prefill_guard', False):return
    @functools.wraps(original)
    def schedule(self, throttle_prefills=False):
        return schedule_guard(original, self, throttle_prefills)
    schedule._q6_prefill_guard = True
    Scheduler.schedule = schedule
    print('Q6_PREFILL_GUARD: separate prefill and decode steps; retain dual decode Dynamic SD',flush=True)
