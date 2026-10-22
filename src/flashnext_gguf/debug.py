# SPDX-License-Identifier: Apache-2.0
import os


def check_moe(x, w1, w2, topk_weights, topk_ids):
    if os.environ.get('FLASHNEXT_DEBUG_MOE') != '1':
        return
    import torch
    if torch.cuda.is_current_stream_capturing():
        return
    ids = topk_ids.cpu()
    ok = bool(((ids >= -1) & (ids < w1.shape[0])).all())
    desc = {'x': (list(x.shape), list(x.stride()), str(x.dtype)),
            'w1': (list(w1.shape), list(w1.stride())),
            'w2': (list(w2.shape), list(w2.stride())),
            'ids': (list(ids.shape), list(topk_ids.stride()), str(ids.dtype)),
            'id_min': int(ids.min()), 'id_max': int(ids.max()),
            'x_finite': bool(torch.isfinite(x).all()),
            'weights_finite': bool(torch.isfinite(topk_weights).all())}
    if x.shape[0] <= 3 or not ok:
        print('FLASHNEXT_MOE_DEBUG', desc, flush=True)
    if not ok:
        raise RuntimeError(f'Invalid expert IDs before GGUF kernel: {desc}; first IDs={ids[:3].tolist()}')
