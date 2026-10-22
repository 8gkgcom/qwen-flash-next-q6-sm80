# SPDX-License-Identifier: Apache-2.0
def register():
    from vllm_gguf_plugin.plugin import register as register_gguf
    from vllm.model_executor.model_loader import register_model_loader
    from .loader import FlashNextLoader
    register_gguf()
    register_model_loader('flashnext_gguf')(FlashNextLoader)

    import os
    if os.environ.get("Q6_DYNAMIC_GRAPH_FIX")=="1":
        from flashnext_gguf.optimizations.q6_dynamic_graphs import install
        install()

    if os.environ.get("Q6_MIXED_PREFILL_GUARD")=="1":
        from flashnext_gguf.optimizations.q6_prefill_guard import install
        install()

    if os.environ.get("Q6_CPU_IPC")=="1":
        from flashnext_gguf.optimizations.q6_cpu_ipc import install
        install()

    if os.environ.get("Q6_DRAFT_Q8_MARLIN")=="1":
        from flashnext_gguf.optimizations.q6_draft_q8_marlin import install
        install()

    if os.environ.get("Q6_SHORTCONV_ASYNC")=="1":
        from flashnext_gguf.optimizations.q6_shortconv_async import install
        install()
