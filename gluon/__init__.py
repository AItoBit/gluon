"""
Gluon — Low-Level GPU Kernel DSL
=================================
Explicit layouts. Explicit shared memory. Explicit warp specialization.
Architecture-specific ops.  Not portable, by design.

>>> import gluon
>>> @gluon.kernel(arch=gluon.SM90, num_warps=8)
... def my_gemm(a, b, c):
...     ...
"""
from .types import (
    DType,
    MemorySpace,
    SwizzleMode,
    Layout,
    SharedMemoryDescriptor,
    TensorRef,
    WarpRole,
)

from .arch import (
    Arch,
    TCShape,
    SM80,
    SM90,
    SM100,
    CDNA3,
    get_arch,
    ARCH_REGISTRY,
)

from .ir import (
    GluonModule,
    KernelFunc,
    Region,
    IRNode,
)

from .dsl import kernel, smem, barrier, mbarrier, warp_role, tma_load, wgmma

__version__ = "0.1.0"

__all__ = [
    # Types
    "DType", "MemorySpace", "SwizzleMode", "Layout",
    "SharedMemoryDescriptor", "TensorRef", "WarpRole",
    # Arches
    "Arch", "TCShape", "SM80", "SM90", "SM100", "CDNA3",
    "get_arch", "ARCH_REGISTRY",
    # IR
    "GluonModule", "KernelFunc", "Region", "IRNode",
    # DSL
    "kernel", "smem", "barrier", "mbarrier", "warp_role",
    "tma_load", "wgmma",
]
