"""Opt-in upstream compute hooks with the original async CPU PLE transport.

The SM80 repository installs these in its GDS initializer. This small adapter
keeps that transport disabled and installs only compute/kernel hooks. It does
not change target weights, n-gram hashing, draft verification or scheduling.
"""
import os
from functools import wraps

_installed = False


def install(runner_cls):
    global _installed
    if _installed or os.environ.get('Q38_RAM_COMPUTE_OPT') != '1':
        return
    if os.environ.get('VLLM_PLE_CPU_OFFLOAD') != '1' or os.environ.get('VLLM_PLE_GDS_RUNNER') == '1':
        raise RuntimeError('RAM compute hooks require CPU PLE and disabled GDS runner')
    from triton_preload import initialize
    previous_load = runner_cls.load_model
    @wraps(previous_load)
    def load_with_preload(self, *args, **kwargs):
        # PP ranks each have TP rank zero. Use the selected CUDA device to keep
        # their startup evidence separate while using the same upstream loader.
        initialize(self.device, rank=self.device.index)
        return previous_load(self, *args, **kwargs)
    runner_cls.load_model = load_with_preload
    if os.environ.get('Q38_RAM_PARALLEL') in ('tp2', 'tep2'):
        from hc_gemv_runtime import install as hc
        from projection_runtime import install as projection
        from draft_int8_runtime import install as draft
        hc(runner_cls)
        projection(runner_cls)
        draft()
    elif os.environ.get('Q38_RAM_PARALLEL') == 'pp2':
        from sm80_ple_local import install as local_ple
        local_ple()
        from sm80_pp_draft_int8 import install as pp_draft
        pp_draft()
        if os.environ.get("Q38_PP_M3_HC") == "1":
            from m3_hc_runtime import install as m3_hc
            m3_hc()
    _installed = True
    print('SM80_RAM_COMPUTE_HOOKS_INSTALLED', os.environ.get('Q38_RAM_PARALLEL'), flush=True)
