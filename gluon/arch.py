"""
Gluon Architecture Descriptors
===============================
Concrete GPU architecture specifications.  Each arch defines its smem capacity,
warp size, tensor-core shapes, and the set of ops it natively supports.
Gluon is explicitly *not* portable — picking an arch locks you to its ISA.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Tuple

from .types import DType, SwizzleMode


# ---------------------------------------------------------------------------
# Tensor core instruction shape
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TCShape:
    """Shape of a single tensor-core instruction (M×N×K)."""
    m: int
    n: int
    k: int
    input_dtype: DType
    output_dtype: DType

    def __repr__(self) -> str:
        return (f"TC(m{self.m}n{self.n}k{self.k}, "
                f"{self.input_dtype.short}→{self.output_dtype.short})")


# ---------------------------------------------------------------------------
# Architecture base
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Arch:
    """
    GPU architecture descriptor.

    Gluon kernels are *bound* to an Arch at declaration time.
    The compiler validates every op against the arch's capability set.
    """
    name: str
    vendor: str                           # "nvidia" | "amd"
    compute_capability: str               # e.g. "sm_90", "gfx942"
    warp_size: int                        # 32 (NVIDIA) or 64 (AMD)
    max_warps_per_block: int
    max_threads_per_block: int
    smem_per_sm_bytes: int                # total shared memory per SM/CU
    max_smem_per_block_bytes: int         # max usable per block
    register_file_per_sm: int             # total 32-bit registers per SM
    max_registers_per_thread: int

    # Tensor core shapes supported
    tc_shapes: Tuple[TCShape, ...] = ()

    # Native op names this arch supports
    native_ops: FrozenSet[str] = frozenset()

    # Preferred swizzle mode for shared memory
    preferred_swizzle: SwizzleMode = SwizzleMode.NONE

    # Cluster support (Hopper+)
    max_cluster_size: int = 1             # 1 = no cluster support

    # TMA support
    has_tma: bool = False

    # Warp-group MMA support
    has_wgmma: bool = False

    # Async copy support (cp.async)
    has_async_copy: bool = False

    @property
    def max_threads(self) -> int:
        return self.max_threads_per_block

    @property
    def warp_groups_per_block(self) -> int:
        """Number of 128-thread warp groups (NVIDIA convention)."""
        return self.max_threads_per_block // (self.warp_size * 4)

    def supports_op(self, op_name: str) -> bool:
        return op_name in self.native_ops

    def validate_smem(self, total_bytes: int) -> Optional[str]:
        """Return error string if smem exceeds capacity, else None."""
        if total_bytes > self.max_smem_per_block_bytes:
            return (
                f"Shared memory request ({total_bytes}B) exceeds "
                f"{self.name} capacity ({self.max_smem_per_block_bytes}B)"
            )
        return None

    def __repr__(self) -> str:
        return f"Arch({self.name}, {self.compute_capability})"


# ---------------------------------------------------------------------------
# Concrete architectures
# ---------------------------------------------------------------------------

SM80 = Arch(
    name="Ampere",
    vendor="nvidia",
    compute_capability="sm_80",
    warp_size=32,
    max_warps_per_block=32,
    max_threads_per_block=1024,
    smem_per_sm_bytes=164 * 1024,         # 164 KB (A100)
    max_smem_per_block_bytes=164 * 1024,
    register_file_per_sm=65536,
    max_registers_per_thread=255,
    tc_shapes=(
        TCShape(m=16, n=8, k=16, input_dtype=DType.f16, output_dtype=DType.f32),
        TCShape(m=16, n=8, k=16, input_dtype=DType.bf16, output_dtype=DType.f32),
        TCShape(m=16, n=8, k=8,  input_dtype=DType.f32, output_dtype=DType.f32),
    ),
    native_ops=frozenset({
        "hmma",            # half-precision MMA (mma.sync)
        "ldmatrix",        # shared → register matrix load
        "cp_async",        # async global → shared copy
        "barrier_async",   # async barrier
        "redux_sync",      # warp-level reduction
    }),
    preferred_swizzle=SwizzleMode.B128,
    has_async_copy=True,
)

SM90 = Arch(
    name="Hopper",
    vendor="nvidia",
    compute_capability="sm_90",
    warp_size=32,
    max_warps_per_block=32,
    max_threads_per_block=1024,
    smem_per_sm_bytes=228 * 1024,         # 228 KB (H100)
    max_smem_per_block_bytes=228 * 1024,
    register_file_per_sm=65536,
    max_registers_per_thread=255,
    tc_shapes=(
        TCShape(m=64, n=256, k=16, input_dtype=DType.f16, output_dtype=DType.f32),
        TCShape(m=64, n=128, k=16, input_dtype=DType.f16, output_dtype=DType.f32),
        TCShape(m=64, n=256, k=16, input_dtype=DType.bf16, output_dtype=DType.f32),
        TCShape(m=64, n=128, k=16, input_dtype=DType.bf16, output_dtype=DType.f32),
    ),
    native_ops=frozenset({
        "wgmma",           # warp-group MMA
        "tma_load",        # tensor memory accelerator load
        "tma_store",       # TMA store
        "tma_prefetch",    # TMA prefetch
        "mbarrier_init",   # mbarrier init
        "mbarrier_arrive", # mbarrier arrive
        "mbarrier_wait",   # mbarrier wait
        "fence_async",     # fence.proxy.async.shared
        "setmaxnreg",      # dynamic register count
        "hmma",            # backward compat
        "ldmatrix",
        "cp_async",
        "barrier_async",
        "redux_sync",
        "cluster_sync",   # cluster-level sync
    }),
    preferred_swizzle=SwizzleMode.B128,
    max_cluster_size=16,
    has_tma=True,
    has_wgmma=True,
    has_async_copy=True,
)

SM100 = Arch(
    name="Blackwell",
    vendor="nvidia",
    compute_capability="sm_100",
    warp_size=32,
    max_warps_per_block=32,
    max_threads_per_block=1024,
    smem_per_sm_bytes=256 * 1024,         # 256 KB (B200)
    max_smem_per_block_bytes=256 * 1024,
    register_file_per_sm=65536,
    max_registers_per_thread=255,
    tc_shapes=(
        TCShape(m=64, n=256, k=16, input_dtype=DType.f16, output_dtype=DType.f32),
        TCShape(m=64, n=256, k=16, input_dtype=DType.bf16, output_dtype=DType.f32),
        TCShape(m=128, n=256, k=32, input_dtype=DType.f16, output_dtype=DType.f32),
    ),
    native_ops=frozenset({
        "wgmma", "tma_load", "tma_store", "tma_prefetch",
        "mbarrier_init", "mbarrier_arrive", "mbarrier_wait",
        "fence_async", "setmaxnreg", "hmma", "ldmatrix",
        "cp_async", "barrier_async", "redux_sync", "cluster_sync",
        "tcgen05_mma",     # 5th-gen tensor core MMA
        "tcgen05_cp",      # 5th-gen tensor core copy
        "bulk_copy",       # bulk async copy
    }),
    preferred_swizzle=SwizzleMode.B128,
    max_cluster_size=32,
    has_tma=True,
    has_wgmma=True,
    has_async_copy=True,
)

CDNA3 = Arch(
    name="CDNA3",
    vendor="amd",
    compute_capability="gfx942",
    warp_size=64,                         # AMD wavefront = 64
    max_warps_per_block=16,               # 1024 threads / 64
    max_threads_per_block=1024,
    smem_per_sm_bytes=64 * 1024,          # 64 KB LDS per CU (MI300X)
    max_smem_per_block_bytes=64 * 1024,
    register_file_per_sm=65536,
    max_registers_per_thread=255,
    tc_shapes=(
        TCShape(m=16, n=16, k=16, input_dtype=DType.f16, output_dtype=DType.f32),
        TCShape(m=16, n=16, k=16, input_dtype=DType.bf16, output_dtype=DType.f32),
        TCShape(m=32, n=32, k=8,  input_dtype=DType.f16, output_dtype=DType.f32),
    ),
    native_ops=frozenset({
        "mfma",            # matrix fused multiply-add
        "ds_read",         # LDS read
        "ds_write",        # LDS write
        "global_load",     # global memory load
        "global_store",    # global memory store
        "s_barrier",       # barrier
        "s_waitcnt",       # wait count
    }),
    preferred_swizzle=SwizzleMode.NONE,
)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ARCH_REGISTRY: Dict[str, Arch] = {
    "sm_80":   SM80,
    "sm_90":   SM90,
    "sm_100":  SM100,
    "gfx942":  CDNA3,
    # aliases
    "ampere":  SM80,
    "hopper":  SM90,
    "blackwell": SM100,
    "cdna3":   CDNA3,
    "mi300x":  CDNA3,
}


def get_arch(name: str) -> Arch:
    """Look up an architecture by name or compute capability."""
    key = name.lower().replace(" ", "")
    if key not in ARCH_REGISTRY:
        available = ", ".join(sorted(ARCH_REGISTRY.keys()))
        raise ValueError(f"Unknown arch '{name}'. Available: {available}")
    return ARCH_REGISTRY[key]
