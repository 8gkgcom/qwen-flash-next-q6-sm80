# SPDX-License-Identifier: Apache-2.0
"""Remove host stream synchronizations from PLE speculative index uploads.

Only scheduling indices change transport; model values, ordering, and math do
not change. PyTorch's pinned allocator retains the upload source until its
copy completes. This module does not patch Tensor.to globally.
"""
import functools
import inspect
import os
import re
import textwrap

REPLACEMENTS = {
    'spec_req_idx_cpu.to(query_start_loc.device)':
        '_q6_upload_index(spec_req_idx_cpu, query_start_loc.device)',
    'non_spec_req_idx_cpu.to(query_start_loc.device)':
        '_q6_upload_index(non_spec_req_idx_cpu, query_start_loc.device)',
    'decode_req_idx_cpu.to(query_start_loc.device)':
        '_q6_upload_index(decode_req_idx_cpu, query_start_loc.device)',
    'spec_req_idx_cpu.to(num_accepted_tokens.device)':
        '_q6_upload_index(spec_req_idx_cpu, num_accepted_tokens.device)',
    'non_spec_req_idx_cpu.to(num_computed_tokens.device)':
        '_q6_upload_index(non_spec_req_idx_cpu, num_computed_tokens.device)',
}


def transform(source):
    for old, new in REPLACEMENTS.items():
        pattern = r'\b' + re.escape(old)
        if len(re.findall(pattern, source)) != 1:
            raise RuntimeError('Unexpected ShortConv metadata source: ' + old)
        source = re.sub(pattern, lambda match: new, source)
    return source


def install():
    if os.environ.get('Q6_SHORTCONV_ASYNC') != '1':
        return
    from vllm.v1.attention.backends import short_conv_attn as module
    from vllm.utils.torch_utils import async_tensor_h2d

    cls = module.PleShortConvAttentionMetadataBuilder
    original = cls.build
    if getattr(original, '_q6_shortconv_async', False):
        return

    def upload_index(tensor, device):
        assert tensor.device.type == 'cpu'
        if device.type == 'cpu':
            return tensor
        return async_tensor_h2d(tensor, device=device)

    source = transform(textwrap.dedent(inspect.getsource(original)))
    namespace = dict(original.__globals__, _q6_upload_index=upload_index)
    exec(compile(source, '<q6-shortconv-async>', 'exec'), namespace)
    patched = functools.update_wrapper(namespace['build'], original)
    patched._q6_shortconv_async = True
    cls.build = patched
    print('Q6_SHORTCONV_ASYNC: five index uploads use pinned asynchronous H2D', flush=True)
