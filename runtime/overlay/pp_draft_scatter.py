"""Restore PP draft rows without CUDA boolean-index compaction and host sync."""
import triton
import triton.language as tl


@triton.jit
def _scatter(Src, Dst, Mapping, S0: tl.constexpr, S1: tl.constexpr,
             D0: tl.constexpr, D1: tl.constexpr, M0: tl.constexpr,
             N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    index = tl.load(Mapping + row * M0)
    col = tl.arange(0, BLOCK)
    active = (index >= 0) & (index < N) & (col < K)
    values = tl.load(Src + row * S0 + col * S1, active, other=0)
    tl.store(Dst + index * D0 + col * D1, values, active)


def restore_drafts(dst, mapping, src):
    # Scheduled valid indices are unique, as required by the original assignment.
    assert dst.ndim == src.ndim == 2 and mapping.ndim == 1
    assert src.shape[0] == mapping.shape[0] and src.shape[1] == dst.shape[1]
    if src.shape[0] and src.shape[1]:
        _scatter[(src.shape[0],)](src, dst, mapping, *src.stride(), *dst.stride(),
            mapping.stride(0), dst.shape[0], src.shape[1],
            triton.next_power_of_2(src.shape[1]), num_warps=1)
