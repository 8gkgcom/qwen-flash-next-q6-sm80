# SPDX-License-Identifier: Apache-2.0
"""Read selected GGUF Q8_0 rows without expanding the table or copying its mapping."""
from __future__ import annotations
import ctypes, json, mmap, os, pathlib
import numpy as np
import torch

ROOT = pathlib.Path(os.environ.get('FLASHNEXT_GGUF_ROOT', '/data/runtime'))

class Q8Rows:
    def __init__(self, entry: dict, resident: bool = False):
        if entry['type'] != 'Q8_0': raise ValueError('PLE must be Q8_0')
        self.width, self.rows = entry['shape']
        if self.width % 32: raise ValueError('Q8_0 row width must be divisible by 32')
        self.stride = self.width // 32 * 34
        self.nbytes = self.rows * self.stride
        if self.nbytes != entry['bytes']: raise ValueError('Incorrect packed table size')
        self.file = open(entry['file'], 'rb')
        start = entry['offset'] // mmap.ALLOCATIONGRANULARITY * mmap.ALLOCATIONGRANULARITY
        delta = entry['offset'] - start
        # NumPy exposes the address of a read-only buffer. Keep file-backed pages:
        # locking a writable private mapping can otherwise force COW copies.
        self.mapping = mmap.mmap(self.file.fileno(), delta + self.nbytes, access=mmap.ACCESS_READ, offset=start)
        self.data = np.ndarray((self.rows, self.stride), np.uint8, buffer=self.mapping, offset=delta)
        self.lib = ctypes.CDLL(str(pathlib.Path(__file__).with_name('q8_rows.so')))
        self.lib.q8_get_rows.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int64, ctypes.c_void_p, ctypes.c_int]
        self.lib.q8_get_rows.restype = ctypes.c_int
        self.pinned = False
        if resident:
            available = int(next(l.split()[1] for l in pathlib.Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
            if available < self.nbytes + 8 * 1024**3:
                raise MemoryError('Not enough available RAM for packed PLE plus 8 GiB reserve')
            libc = ctypes.CDLL(None, use_errno=True)
            libc.mlock.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            if libc.mlock(self.data.ctypes.data, self.nbytes):
                raise OSError(ctypes.get_errno(), 'mlock of packed PLE failed; table was not expanded')
            self.pinned = True

    def gather(self, ids: torch.Tensor, out: torch.Tensor | None = None, dtype=torch.bfloat16):
        if ids.device.type != 'cpu' or ids.dtype != torch.int64: raise TypeError('PLE row IDs must be CPU int64')
        ids = ids.contiguous()
        if out is None: out = torch.empty((*ids.shape, self.width), dtype=dtype, device='cpu')
        if out.device.type != 'cpu' or not out.is_contiguous() or out.numel() != ids.numel() * self.width:
            raise ValueError('Output must be contiguous CPU storage with the exact shape')
        kinds = {torch.float32:0, torch.bfloat16:1}
        if out.dtype not in kinds: raise TypeError('Output must be BF16 or F32')
        rc = self.lib.q8_get_rows(self.data.ctypes.data, self.rows, self.width, ids.data_ptr(), ids.numel(), out.data_ptr(), kinds[out.dtype])
        if rc == -2: raise IndexError('PLE row ID out of bounds')
        if rc: raise RuntimeError(f'Q8 row decoder failed: {rc}')
        return out

class Q8PLEEmbedding(torch.nn.Module):
    def __init__(self, rows: int, width: int, **kwargs):
        super().__init__()
        entries = json.loads((ROOT / 'gguf-manifest.json').read_text())
        entry = next(e for e in entries if e['name'] == 'per_layer_token_embd.weight')
        if entry['shape'] != [width, rows]: raise ValueError('GGUF PLE shape differs from model config')
        self.table = Q8Rows(entry, resident=os.environ.get('FLASHNEXT_PLE_RESIDENT', '1') == '1')
        self.output_dtype = torch.bfloat16

    def forward(self, ids, out=None):
        return self.table.gather(ids, out, self.output_dtype)
