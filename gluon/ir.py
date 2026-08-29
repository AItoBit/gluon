"""
Gluon IR — Region-Based Intermediate Representation
=====================================================
Structured IR for GPU kernel programs with explicit warp roles,
memory ops, compute ops, synchronization, and control flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .types import (
    DType, Layout, MemorySpace, SharedMemoryDescriptor,
    SwizzleMode, TensorRef, WarpRole,
)
from .arch import Arch, TCShape


# ---------------------------------------------------------------------------
# Base IR node
# ---------------------------------------------------------------------------

@dataclass
class IRNode:
    """Base class for all Gluon IR nodes."""
    _id_counter: int = field(default=0, init=False, repr=False)

    def __post_init__(self):
        IRNode._id_counter += 1
        self._node_id: int = IRNode._id_counter

    @property
    def node_id(self) -> int:
        return self._node_id

    def pretty(self, indent: int = 0) -> str:
        """Pretty-print this node (override in subclasses)."""
        prefix = "  " * indent
        return f"{prefix}{self.__class__.__name__}(id={self._node_id})"


# ===================================================================
#  MEMORY OPS
# ===================================================================

@dataclass
class SmemAlloc(IRNode):
    """Allocate a typed shared-memory region."""
    descriptor: SharedMemoryDescriptor = field(default_factory=lambda: SharedMemoryDescriptor(
        name="unnamed", dtype=DType.f32, layout=Layout.row_major(1)
    ))
    offset: int = 0  # assigned by smem_alloc pass

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        return (f"{p}smem.alloc '{self.descriptor.name}' "
                f"{self.descriptor.dtype.short}[{self.descriptor.layout.shape}] "
                f"@ offset={self.offset}")


@dataclass
class SmemStore(IRNode):
    """Store data into shared memory."""
    dst: str = ""       # smem region name
    src: str = ""       # source register / value name
    indices: Tuple[str, ...] = ()

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        idx = ", ".join(self.indices)
        return f"{p}smem.store {self.dst}[{idx}] <- {self.src}"


@dataclass
class SmemLoad(IRNode):
    """Load data from shared memory to registers."""
    dst: str = ""
    src: str = ""       # smem region name
    indices: Tuple[str, ...] = ()

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        idx = ", ".join(self.indices)
        return f"{p}{self.dst} = smem.load {self.src}[{idx}]"


@dataclass
class GlobalLoad(IRNode):
    """Load from global memory."""
    dst: str = ""
    src: str = ""       # global tensor name
    indices: Tuple[str, ...] = ()

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        idx = ", ".join(self.indices)
        return f"{p}{self.dst} = global.load {self.src}[{idx}]"


@dataclass
class GlobalStore(IRNode):
    """Store to global memory."""
    dst: str = ""
    src: str = ""
    indices: Tuple[str, ...] = ()

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        idx = ", ".join(self.indices)
        return f"{p}global.store {self.dst}[{idx}] <- {self.src}"


@dataclass
class TMALoad(IRNode):
    """Tensor Memory Accelerator load (Hopper+)."""
    dst_smem: str = ""
    src_global: str = ""
    coords: Tuple[str, ...] = ()
    barrier: str = ""   # mbarrier name for completion tracking

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        c = ", ".join(self.coords)
        return (f"{p}tma.load {self.dst_smem} <- {self.src_global}[{c}] "
                f"(barrier={self.barrier})")


@dataclass
class TMAPrefetch(IRNode):
    """TMA prefetch hint."""
    src_global: str = ""
    coords: Tuple[str, ...] = ()

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        c = ", ".join(self.coords)
        return f"{p}tma.prefetch {self.src_global}[{c}]"


# ===================================================================
#  COMPUTE OPS
# ===================================================================

@dataclass
class WGMMA(IRNode):
    """Warp-Group Matrix Multiply-Accumulate (Hopper sm_90)."""
    a_smem: str = ""    # operand A in shared memory
    b_smem: str = ""    # operand B in shared memory
    c_reg: str = ""     # accumulator in registers
    shape: Optional[TCShape] = None

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        s = f"m{self.shape.m}n{self.shape.n}k{self.shape.k}" if self.shape else "?"
        return f"{p}{self.c_reg} = wgmma({self.a_smem}, {self.b_smem}) [{s}]"


@dataclass
class HMMA(IRNode):
    """Half-precision MMA (Ampere mma.sync)."""
    a: str = ""
    b: str = ""
    c: str = ""
    shape: Optional[TCShape] = None

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        s = f"m{self.shape.m}n{self.shape.n}k{self.shape.k}" if self.shape else "?"
        return f"{p}{self.c} = hmma({self.a}, {self.b}) [{s}]"


@dataclass
class MFMA(IRNode):
    """Matrix Fused Multiply-Add (AMD CDNA)."""
    a: str = ""
    b: str = ""
    c: str = ""
    shape: Optional[TCShape] = None

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        s = f"m{self.shape.m}n{self.shape.n}k{self.shape.k}" if self.shape else "?"
        return f"{p}{self.c} = mfma({self.a}, {self.b}) [{s}]"


@dataclass
class FMA(IRNode):
    """Scalar fused multiply-add: d = a * b + c."""
    dst: str = ""
    a: str = ""
    b: str = ""
    c: str = ""
    dtype: DType = DType.f32

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        return f"{p}{self.dst} = fma({self.a}, {self.b}, {self.c}) [{self.dtype.short}]"


@dataclass
class ElementwiseOp(IRNode):
    """Generic elementwise operation."""
    dst: str = ""
    op: str = ""        # "add", "mul", "silu", "exp", "max", etc.
    operands: Tuple[str, ...] = ()
    dtype: DType = DType.f32

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        args = ", ".join(self.operands)
        return f"{p}{self.dst} = {self.op}({args}) [{self.dtype.short}]"


# ===================================================================
#  SYNCHRONIZATION OPS
# ===================================================================

@dataclass
class Barrier(IRNode):
    """Classic __syncthreads() barrier."""
    def pretty(self, indent: int = 0) -> str:
        return f"{'  ' * indent}barrier.sync"


@dataclass
class MBarrierInit(IRNode):
    """Initialize an mbarrier object (Hopper+)."""
    name: str = ""
    count: int = 1      # expected arrival count

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        return f"{p}mbarrier.init '{self.name}' (count={self.count})"


@dataclass
class MBarrierArrive(IRNode):
    """Signal arrival at an mbarrier."""
    name: str = ""
    bytes_arrived: int = 0  # for TMA tracking

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        return f"{p}mbarrier.arrive '{self.name}' (bytes={self.bytes_arrived})"


@dataclass
class MBarrierWait(IRNode):
    """Wait on an mbarrier phase."""
    name: str = ""
    phase: int = 0

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        return f"{p}mbarrier.wait '{self.name}' (phase={self.phase})"


@dataclass
class FenceAsyncShared(IRNode):
    """Async proxy fence for shared memory."""
    def pretty(self, indent: int = 0) -> str:
        return f"{'  ' * indent}fence.async.shared"


# ===================================================================
#  CONTROL FLOW
# ===================================================================

@dataclass
class ForLoop(IRNode):
    """Counted for-loop."""
    var: str = "i"
    start: Union[int, str] = 0
    end: Union[int, str] = 0
    step: Union[int, str] = 1
    body: List[IRNode] = field(default_factory=list)

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        lines = [f"{p}for {self.var} in [{self.start}, {self.end}) step {self.step}:"]
        for node in self.body:
            lines.append(node.pretty(indent + 1))
        return "\n".join(lines)


@dataclass
class IfElse(IRNode):
    """Conditional branch."""
    condition: str = ""
    then_body: List[IRNode] = field(default_factory=list)
    else_body: List[IRNode] = field(default_factory=list)

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        lines = [f"{p}if ({self.condition}):"]
        for n in self.then_body:
            lines.append(n.pretty(indent + 1))
        if self.else_body:
            lines.append(f"{p}else:")
            for n in self.else_body:
                lines.append(n.pretty(indent + 1))
        return "\n".join(lines)


@dataclass
class WarpSpecialize(IRNode):
    """
    Warp specialization region.
    Splits the thread block into role-specific regions.
    """
    regions: Dict[WarpRole, 'Region'] = field(default_factory=dict)

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        lines = [f"{p}warp_specialize:"]
        for role, region in self.regions.items():
            lines.append(region.pretty(indent + 1))
        return "\n".join(lines)


# ===================================================================
#  REGION — scoped code block with a warp role
# ===================================================================

@dataclass
class Region:
    """
    A code region scoped to a warp role.
    Contains a sequence of IR nodes and optional warp range.
    """
    role: WarpRole = WarpRole.ANY
    warp_range: Optional[Tuple[int, int]] = None  # (start_warp, end_warp)
    body: List[IRNode] = field(default_factory=list)

    def append(self, node: IRNode) -> None:
        self.body.append(node)

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        wr = f" warps=[{self.warp_range[0]}..{self.warp_range[1]}]" if self.warp_range else ""
        lines = [f"{p}region @{self.role.name}{wr}:"]
        for node in self.body:
            lines.append(node.pretty(indent + 1))
        return "\n".join(lines)


# ===================================================================
#  KERNEL FUNCTION
# ===================================================================

@dataclass
class KernelFunc:
    """
    A Gluon kernel function.
    Bound to a specific architecture with explicit grid/block config.
    """
    name: str
    arch: Arch
    params: List[TensorRef] = field(default_factory=list)
    num_warps: int = 8
    smem_descriptors: List[SharedMemoryDescriptor] = field(default_factory=list)
    body: List[IRNode] = field(default_factory=list)
    grid: Tuple[Union[int, str], ...] = (1, 1, 1)

    @property
    def num_threads(self) -> int:
        return self.num_warps * self.arch.warp_size

    @property
    def total_smem_bytes(self) -> int:
        return sum(d.total_bytes for d in self.smem_descriptors)

    def append(self, node: IRNode) -> None:
        self.body.append(node)

    def pretty(self, indent: int = 0) -> str:
        p = "  " * indent
        params_str = ", ".join(f"{r.name}: {r.dtype.short}[{r.shape}]"
                               for r in self.params)
        lines = [
            f"{p}kernel @{self.name}({params_str})",
            f"{p}  arch = {self.arch.name} ({self.arch.compute_capability})",
            f"{p}  grid = {self.grid}",
            f"{p}  warps = {self.num_warps} ({self.num_threads} threads)",
            f"{p}  smem = {self.total_smem_bytes} bytes",
            f"{p}  body:",
        ]
        for node in self.body:
            lines.append(node.pretty(indent + 2))
        return "\n".join(lines)


# ===================================================================
#  MODULE — top-level container
# ===================================================================

@dataclass
class GluonModule:
    """Top-level container for a Gluon program."""
    name: str = "gluon_module"
    kernels: List[KernelFunc] = field(default_factory=list)

    def add_kernel(self, kernel: KernelFunc) -> None:
        self.kernels.append(kernel)

    def pretty(self) -> str:
        lines = [f"module @{self.name}:"]
        for k in self.kernels:
            lines.append(k.pretty(indent=1))
            lines.append("")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.pretty()
