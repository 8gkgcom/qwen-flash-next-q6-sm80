# SPDX-License-Identifier: Apache-2.0
"""Q6 shared-memory readers: brief spin, then wait for write notifications."""
import functools
import time


def install():
    from vllm.distributed.device_communicators import shm_broadcast as shm
    if getattr(shm, '_q6_cpu_ipc_installed', False):
        return
    shm._q6_cpu_ipc_installed = True
    # The optional native loop spins for up to 100 ms before using notifications.
    shm.SPINLOOP_EXT_ENABLED = False
    original_init = shm.SpinCondition.__init__
    original_wait = shm.SpinCondition.wait

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if self.is_reader:
            self.busy_loop_s = 0.0002

    @functools.wraps(original_wait)
    def wait(self, timeout_ms=None):
        # Keep the original SHM fences, notifications, cancellation and timeouts.
        # Recheck the authoritative SHM flags even if a PUB notification is lost.
        # Longer waits when idle avoid frequent wakeups without active requests.
        cap = 10 if time.monotonic() - self.last_read > 1.0 else 1
        return original_wait(self, min(timeout_ms, cap) if timeout_ms is not None else cap)

    shm.SpinCondition.__init__ = init
    shm.SpinCondition.wait = wait
    print('Q6_CPU_IPC spin=200us notification recheck=1ms active/10ms idle', flush=True)
