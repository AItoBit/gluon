"""
Gluon DSL — Python-Embedded Domain-Specific Language
=====================================================
Decorators and builtins that let you write Gluon kernels as decorated
Python functions.  The DSL traces your code into Gluon IR.
"""
from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .types import (
    DType, Layout, MemorySpace, SharedMemoryDescriptor,
    SwizzleMode, TensorRef, WarpRole,
)
from .arch import Arch, SM90
from .ir import (
    GluonModule, KernelFunc, Region, IRNode,
    SmemAlloc, SmemStore, SmemLoad,
    GlobalLoad, GlobalStore,
    TMALoad, TMAPrefetch,
    WGMMA, HMMA, MFMA, FMA, ElementwiseOp,
    Barrier, MBarrierInit, MBarrierArrive, MBarrierWait, FenceAsyncShared,
    ForLoop, IfElse, WarpSpecialize,
)


# ---------------------------------------------------------------------------
# Trace context — thread-local IR builder state
# ---------------------------------------------------------------------------

class _TraceContext:
    """
    Mutable context used during kernel tracing.
    Records IR nodes as DSL functions are called.
    """

    def __init__(self, kernel_func: KernelFunc):
        self.kernel = kernel_func
        self._region_stack: List[Region] = [
            Region(role=WarpRole.ANY, body=[])
        ]
        self._smem_counter = 0
        self._reg_counter = 0
        self._mbarrier_counter = 0

    @property
    def current_region(self) -> Region:
        return self._region_stack[-1]

    def emit(self, node: IRNode) -> None:
        self.current_region.append(node)

    def push_region(self, role: WarpRole,
                    warp_range: Optional[Tuple[int, int]] = None) -> Region:
        r = Region(role=role, warp_range=warp_range, body=[])
        self._region_stack.append(r)
        return r

    def pop_region(self) -> Region:
        return self._region_stack.pop()

    def fresh_reg(self, prefix: str = "r") -> str:
        self._reg_counter += 1
        return f"%{prefix}{self._reg_counter}"

    def fresh_smem(self, prefix: str = "smem") -> str:
        self._smem_counter += 1
        return f"{prefix}_{self._smem_counter}"

    def fresh_mbarrier(self, prefix: str = "mbar") -> str:
        self._mbarrier_counter += 1
        return f"{prefix}_{self._mbarrier_counter}"


# Global trace context (set during tracing only)
_ctx: Optional[_TraceContext] = None


def _get_ctx() -> _TraceContext:
    if _ctx is None:
        raise RuntimeError(
            "Gluon DSL functions can only be called inside a @gluon.kernel trace"
        )
    return _ctx


# ---------------------------------------------------------------------------
# @gluon.kernel decorator
# ---------------------------------------------------------------------------

def kernel(
    fn: Optional[Callable] = None,
    *,
    arch: Arch = SM90,
    num_warps: int = 8,
    grid: Tuple[Union[int, str], ...] = (1, 1, 1),
) -> Any:
    """
    Declare a Gluon kernel.

    Usage::

        @gluon.kernel(arch=gluon.SM90, num_warps=8)
        def my_gemm(a, b, c):
            ...

    The decorated function is traced: calling it produces a GluonModule
    containing the kernel's IR.
    """
    def decorator(func: Callable) -> '_GluonKernelDef':
        return _GluonKernelDef(func, arch=arch, num_warps=num_warps, grid=grid)

    if fn is not None:
        return decorator(fn)
    return decorator


class _GluonKernelDef:
    """A traced Gluon kernel definition."""

    def __init__(self, fn: Callable, *, arch: Arch, num_warps: int,
                 grid: Tuple[Union[int, str], ...]):
        self._fn = fn
        self._arch = arch
        self._num_warps = num_warps
        self._grid = grid
        self._name = fn.__name__
        functools.update_wrapper(self, fn)

    def trace(self, *tensor_refs: TensorRef) -> GluonModule:
        """
        Trace the kernel function to produce a GluonModule.

        Pass TensorRef objects describing each kernel parameter.
        """
        global _ctx

        kf = KernelFunc(
            name=self._name,
            arch=self._arch,
            params=list(tensor_refs),
            num_warps=self._num_warps,
            grid=self._grid,
        )

        _ctx = _TraceContext(kf)
        try:
            # Execute the function body — DSL calls record IR nodes
            self._fn(*tensor_refs)

            # Flush the top-level region into the kernel body
            top = _ctx.current_region
            kf.body = top.body
            kf.smem_descriptors = [
                n.descriptor for n in top.body if isinstance(n, SmemAlloc)
            ]
            # Also collect smem from nested regions
            _collect_smem(kf.body, kf.smem_descriptors)
        finally:
            _ctx = None

        module = GluonModule(name=f"{self._name}_module")
        module.add_kernel(kf)
        return module

    def __call__(self, *args, **kwargs):
        """Direct call returns trace for inspection."""
        return self.trace(*args, **kwargs)

    def __repr__(self) -> str:
        return (f"GluonKernel('{self._name}', arch={self._arch.name}, "
                f"warps={self._num_warps})")


def _collect_smem(nodes: List[IRNode], descriptors: List[SharedMemoryDescriptor]):
    """Recursively collect SmemAlloc descriptors."""
    for n in nodes:
        if isinstance(n, WarpSpecialize):
            for region in n.regions.values():
                for sub in region.body:
                    if isinstance(sub, SmemAlloc):
                        if sub.descriptor not in descriptors:
                            descriptors.append(sub.descriptor)
                _collect_smem(region.body, descriptors)
        elif isinstance(n, ForLoop):
            _collect_smem(n.body, descriptors)
        elif isinstance(n, IfElse):
            _collect_smem(n.then_body, descriptors)
            _collect_smem(n.else_body, descriptors)


# ---------------------------------------------------------------------------
# DSL builtins — shared memory
# ---------------------------------------------------------------------------

def smem(dtype: DType, layout: Layout, *,
         name: Optional[str] = None,
         double_buffer: bool = False,
         alignment: int = 128) -> str:
    """
    Allocate typed shared memory with an explicit layout.

    Returns the smem region name (used in subsequent load/store calls).
    """
    ctx = _get_ctx()
    region_name = name or ctx.fresh_smem()
    desc = SharedMemoryDescriptor(
        name=region_name,
        dtype=dtype,
        layout=layout,
        alignment=alignment,
        double_buffer=double_buffer,
    )
    ctx.emit(SmemAlloc(descriptor=desc))
    return region_name


# ---------------------------------------------------------------------------
# DSL builtins — synchronization
# ---------------------------------------------------------------------------

def barrier() -> None:
    """Emit a __syncthreads() barrier."""
    _get_ctx().emit(Barrier())


def mbarrier(count: int = 1, *, name: Optional[str] = None) -> str:
    """
    Initialize an mbarrier (Hopper+).
    Returns the mbarrier name for arrive/wait calls.
    """
    ctx = _get_ctx()
    arch = ctx.kernel.arch
    if not arch.supports_op("mbarrier_init"):
        raise RuntimeError(
            f"mbarrier not supported on {arch.name} ({arch.compute_capability})"
        )
    mbar_name = name or ctx.fresh_mbarrier()
    ctx.emit(MBarrierInit(name=mbar_name, count=count))
    return mbar_name


def mbarrier_arrive(name: str, bytes_arrived: int = 0) -> None:
    """Signal arrival at a named mbarrier."""
    _get_ctx().emit(MBarrierArrive(name=name, bytes_arrived=bytes_arrived))


def mbarrier_wait(name: str, phase: int = 0) -> None:
    """Wait on a named mbarrier phase."""
    _get_ctx().emit(MBarrierWait(name=name, phase=phase))


def fence_async_shared() -> None:
    """Emit an async proxy fence for shared memory."""
    _get_ctx().emit(FenceAsyncShared())


# ---------------------------------------------------------------------------
# DSL builtins — warp specialization
# ---------------------------------------------------------------------------

@contextmanager
def warp_role(role: WarpRole,
              warp_range: Optional[Tuple[int, int]] = None):
    """
    Context manager for warp-specialized regions.

    Usage::

        with gluon.warp_role(WarpRole.PRODUCER, warp_range=(0, 3)):
            gluon.tma_load(...)
        with gluon.warp_role(WarpRole.CONSUMER, warp_range=(4, 7)):
            gluon.wgmma(...)
    """
    ctx = _get_ctx()

    # Check if we're inside a WarpSpecialize node already
    region = ctx.push_region(role, warp_range)
    try:
        yield region
    finally:
        completed_region = ctx.pop_region()
        # Wrap in a WarpSpecialize node or add to existing one
        parent = ctx.current_region
        # Find or create WarpSpecialize in parent
        ws_node = None
        if parent.body and isinstance(parent.body[-1], WarpSpecialize):
            ws_node = parent.body[-1]
        if ws_node is None:
            ws_node = WarpSpecialize(regions={})
            parent.append(ws_node)
        ws_node.regions[role] = completed_region


# ---------------------------------------------------------------------------
# DSL builtins — data movement
# ---------------------------------------------------------------------------

def tma_load(dst_smem: str, src_global: str, *,
             coords: Tuple[str, ...] = (),
             barrier_name: str = "") -> None:
    """
    TMA load from global to shared memory (Hopper+).
    """
    ctx = _get_ctx()
    arch = ctx.kernel.arch
    if not arch.has_tma:
        raise RuntimeError(
            f"TMA not supported on {arch.name} ({arch.compute_capability}). "
            f"Requires sm_90+."
        )
    ctx.emit(TMALoad(
        dst_smem=dst_smem,
        src_global=src_global,
        coords=coords,
        barrier=barrier_name,
    ))


def tma_prefetch(src_global: str, *,
                 coords: Tuple[str, ...] = ()) -> None:
    """TMA prefetch hint (Hopper+)."""
    ctx = _get_ctx()
    if not ctx.kernel.arch.has_tma:
        raise RuntimeError("TMA prefetch requires sm_90+")
    ctx.emit(TMAPrefetch(src_global=src_global, coords=coords))


def global_load(dst: str, src: str, *,
                indices: Tuple[str, ...] = ()) -> str:
    """Load from global memory to registers."""
    ctx = _get_ctx()
    reg = dst or ctx.fresh_reg("gld")
    ctx.emit(GlobalLoad(dst=reg, src=src, indices=indices))
    return reg


def global_store(dst: str, src: str, *,
                 indices: Tuple[str, ...] = ()) -> None:
    """Store from registers to global memory."""
    _get_ctx().emit(GlobalStore(dst=dst, src=src, indices=indices))


def smem_load(dst: str, src_smem: str, *,
              indices: Tuple[str, ...] = ()) -> str:
    """Load from shared memory to a register."""
    ctx = _get_ctx()
    reg = dst or ctx.fresh_reg("sld")
    ctx.emit(SmemLoad(dst=reg, src=src_smem, indices=indices))
    return reg


def smem_store(dst_smem: str, src: str, *,
               indices: Tuple[str, ...] = ()) -> None:
    """Store from a register into shared memory."""
    _get_ctx().emit(SmemStore(dst=dst_smem, src=src, indices=indices))


# ---------------------------------------------------------------------------
# DSL builtins — compute (architecture-specific)
# ---------------------------------------------------------------------------

def wgmma(a_smem: str, b_smem: str, c_reg: str, *,
          shape: Optional[Any] = None) -> str:
    """
    Warp-Group Matrix Multiply-Accumulate (Hopper sm_90).
    Operates on tiles in shared memory, accumulates in registers.
    """
    ctx = _get_ctx()
    arch = ctx.kernel.arch
    if not arch.has_wgmma:
        raise RuntimeError(
            f"WGMMA not supported on {arch.name} ({arch.compute_capability}). "
            f"Requires sm_90+."
        )
    tc = shape or (arch.tc_shapes[0] if arch.tc_shapes else None)
    ctx.emit(WGMMA(a_smem=a_smem, b_smem=b_smem, c_reg=c_reg, shape=tc))
    return c_reg


def hmma(a: str, b: str, c: str, *,
         shape: Optional[Any] = None) -> str:
    """Half-precision MMA (Ampere mma.sync)."""
    ctx = _get_ctx()
    arch = ctx.kernel.arch
    if not arch.supports_op("hmma"):
        raise RuntimeError(f"HMMA not supported on {arch.name}")
    tc = shape or (arch.tc_shapes[0] if arch.tc_shapes else None)
    ctx.emit(HMMA(a=a, b=b, c=c, shape=tc))
    return c


def mfma(a: str, b: str, c: str, *,
         shape: Optional[Any] = None) -> str:
    """Matrix Fused Multiply-Add (AMD CDNA)."""
    ctx = _get_ctx()
    arch = ctx.kernel.arch
    if not arch.supports_op("mfma"):
        raise RuntimeError(f"MFMA not supported on {arch.name}")
    tc = shape or (arch.tc_shapes[0] if arch.tc_shapes else None)
    ctx.emit(MFMA(a=a, b=b, c=c, shape=tc))
    return c


def fma(a: str, b: str, c: str, *,
        dtype: DType = DType.f32,
        dst: Optional[str] = None) -> str:
    """Scalar fused multiply-add: d = a * b + c."""
    ctx = _get_ctx()
    reg = dst or ctx.fresh_reg("fma")
    ctx.emit(FMA(dst=reg, a=a, b=b, c=c, dtype=dtype))
    return reg


def elementwise(op: str, *operands: str,
                dtype: DType = DType.f32,
                dst: Optional[str] = None) -> str:
    """Generic elementwise operation (add, mul, silu, exp, max, ...)."""
    ctx = _get_ctx()
    reg = dst or ctx.fresh_reg(op)
    ctx.emit(ElementwiseOp(dst=reg, op=op, operands=operands, dtype=dtype))
    return reg


# ---------------------------------------------------------------------------
# DSL builtins — control flow helpers
# ---------------------------------------------------------------------------

@contextmanager
def for_range(var: str, start: Union[int, str],
              end: Union[int, str], step: Union[int, str] = 1):
    """
    Emits a ForLoop in the IR.

    Usage::

        with gluon.for_range("k", 0, K, BLOCK_K) as loop:
            ...
    """
    ctx = _get_ctx()
    loop = ForLoop(var=var, start=start, end=end, step=step, body=[])
    # Push a temporary region for the loop body
    region = ctx.push_region(ctx.current_region.role)
    try:
        yield loop
    finally:
        completed = ctx.pop_region()
        loop.body = completed.body
        ctx.emit(loop)


# ---------------------------------------------------------------------------
# DSL builtins — thread indexing
# ---------------------------------------------------------------------------

def program_id(axis: int = 0) -> str:
    """Return a symbolic name for the block/program index along an axis."""
    return f"%pid_{axis}"


def warp_id() -> str:
    """Return a symbolic name for the warp index within a block."""
    return "%warp_id"


def lane_id() -> str:
    """Return a symbolic name for the lane index within a warp."""
    return "%lane_id"
