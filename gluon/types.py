"""
Gluon Type System
=================
Core data types, memory spaces, layouts, and tensor references for the Gluon DSL.
Explicit layouts and shared memory descriptors are first-class — no implicit defaults.
"""
from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class DType(enum.Enum):
    """Supported scalar data types."""
    f16  = ("float16",  2, "f16")
    bf16 = ("bfloat16", 2, "bf16")
    f32  = ("float32",  4, "f32")
    f64  = ("float64",  8, "f64")
    i8   = ("int8",     1, "i8")
    i16  = ("int16",    2, "i16")
    i32  = ("int32",    4, "i32")
    i64  = ("int64",    8, "i64")
    u8   = ("uint8",    1, "u8")
    u16  = ("uint16",   2, "u16")
    u32  = ("uint32",   4, "u32")
    u64  = ("uint64",   8, "u64")

    def __init__(self, label: str, nbytes: int, short: str):
        self.label = label
        self.nbytes = nbytes
        self.short = short

    @property
    def bits(self) -> int:
        return self.nbytes * 8

    def __repr__(self) -> str:
        return f"gluon.{self.short}"


# ---------------------------------------------------------------------------
# Memory spaces
# ---------------------------------------------------------------------------

class MemorySpace(enum.Enum):
    """GPU memory hierarchy levels."""
    GLOBAL   = "global"
    SHARED   = "shared"
    LOCAL    = "local"
    REGISTER = "register"

    def __repr__(self) -> str:
        return f"MemorySpace.{self.name}"


# ---------------------------------------------------------------------------
# Swizzle patterns (for shared-memory bank-conflict avoidance)
# ---------------------------------------------------------------------------

class SwizzleMode(enum.Enum):
    """Swizzle modes for shared memory layouts."""
    NONE     = "none"
    B32      = "32B"      # 32-byte swizzle  (Ampere+)
    B64      = "64B"      # 64-byte swizzle
    B128     = "128B"     # 128-byte swizzle (Hopper TMA)

    @property
    def bytes(self) -> int:
        return {"none": 0, "32B": 32, "64B": 64, "128B": 128}[self.value]


# ---------------------------------------------------------------------------
# Layout — explicit tensor data layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Layout:
    """
    Explicit N-dimensional tensor layout.

    Stores shape and strides in elements (not bytes).
    Swizzle mode controls shared-memory banking strategy.
    """
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]
    swizzle: SwizzleMode = SwizzleMode.NONE

    # -- Factories ----------------------------------------------------------

    @classmethod
    def row_major(cls, *dims: int, swizzle: SwizzleMode = SwizzleMode.NONE) -> Layout:
        """C-contiguous (row-major) layout."""
        strides = cls._row_major_strides(dims)
        return cls(shape=dims, strides=strides, swizzle=swizzle)

    @classmethod
    def col_major(cls, *dims: int, swizzle: SwizzleMode = SwizzleMode.NONE) -> Layout:
        """Fortran-contiguous (column-major) layout."""
        strides = cls._col_major_strides(dims)
        return cls(shape=dims, strides=strides, swizzle=swizzle)

    @classmethod
    def swizzled(cls, *dims: int, swizzle: SwizzleMode = SwizzleMode.B128) -> Layout:
        """Row-major layout with a bank-conflict-avoidance swizzle."""
        strides = cls._row_major_strides(dims)
        return cls(shape=dims, strides=strides, swizzle=swizzle)

    @classmethod
    def custom(cls, shape: Tuple[int, ...], strides: Tuple[int, ...],
               swizzle: SwizzleMode = SwizzleMode.NONE) -> Layout:
        """Fully custom layout."""
        if len(shape) != len(strides):
            raise ValueError("shape and strides must have the same rank")
        return cls(shape=shape, strides=strides, swizzle=swizzle)

    # -- Properties ---------------------------------------------------------

    @property
    def rank(self) -> int:
        return len(self.shape)

    @property
    def numel(self) -> int:
        """Total number of elements."""
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def is_contiguous(self) -> bool:
        """Check if layout is row-major contiguous."""
        return self.strides == self._row_major_strides(self.shape)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _row_major_strides(dims: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = []
        stride = 1
        for d in reversed(dims):
            strides.append(stride)
            stride *= d
        return tuple(reversed(strides))

    @staticmethod
    def _col_major_strides(dims: Tuple[int, ...]) -> Tuple[int, ...]:
        strides = []
        stride = 1
        for d in dims:
            strides.append(stride)
            stride *= d
        return tuple(strides)

    def byte_size(self, dtype: DType) -> int:
        """Total contiguous storage in bytes (ignoring swizzle padding)."""
        return self.numel * dtype.nbytes

    def linear_index(self, *indices: int) -> int:
        """Compute flat element index from multi-dimensional indices."""
        if len(indices) != self.rank:
            raise ValueError(f"Expected {self.rank} indices, got {len(indices)}")
        idx = 0
        for i, s in zip(indices, self.strides):
            idx += i * s
        return idx

    def __repr__(self) -> str:
        s = f"Layout(shape={self.shape}, strides={self.strides}"
        if self.swizzle != SwizzleMode.NONE:
            s += f", swizzle={self.swizzle.value}"
        s += ")"
        return s


# ---------------------------------------------------------------------------
# Shared Memory Descriptor
# ---------------------------------------------------------------------------

@dataclass
class SharedMemoryDescriptor:
    """
    Typed shared-memory region with explicit layout and alignment.

    This is the central abstraction distinguishing Gluon from Triton:
    the programmer explicitly declares how tiles live in smem.
    """
    name: str
    dtype: DType
    layout: Layout
    alignment: int = 128          # bytes, default = cache-line
    double_buffer: bool = False   # allocate 2× for pipelining

    @property
    def single_buffer_bytes(self) -> int:
        raw = self.layout.byte_size(self.dtype)
        return int(math.ceil(raw / self.alignment) * self.alignment)

    @property
    def total_bytes(self) -> int:
        base = self.single_buffer_bytes
        return base * 2 if self.double_buffer else base

    @property
    def num_banks(self) -> int:
        """Shared memory has 32 banks on all modern NVIDIA/AMD GPUs."""
        return 32

    def check_bank_conflicts(self) -> List[str]:
        """
        Heuristic bank-conflict analysis for 2-D tiles.
        Returns a list of warning strings (empty = no issues detected).
        """
        warnings: List[str] = []
        if self.layout.rank < 2:
            return warnings

        inner_stride_bytes = self.layout.strides[-1] * self.dtype.nbytes
        if inner_stride_bytes == 0:
            return warnings

        outer_stride_bytes = self.layout.strides[-2] * self.dtype.nbytes

        # Check if consecutive rows map to the same bank
        if self.layout.swizzle == SwizzleMode.NONE:
            row_width_bytes = self.layout.shape[-1] * self.dtype.nbytes
            bank_width = 4  # 4 bytes per bank
            if row_width_bytes % (self.num_banks * bank_width) == 0:
                warnings.append(
                    f"Potential {self.num_banks}-way bank conflict: "
                    f"row width ({row_width_bytes}B) is a multiple of "
                    f"{self.num_banks * bank_width}B. Consider adding swizzle."
                )
        return warnings

    def __repr__(self) -> str:
        buf = "double" if self.double_buffer else "single"
        return (
            f"SharedMem('{self.name}', {self.dtype!r}, {self.layout!r}, "
            f"align={self.alignment}, {buf}-buffered, {self.total_bytes}B)"
        )


# ---------------------------------------------------------------------------
# Tensor Reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TensorRef:
    """
    A typed reference to a tensor in a specific memory space.
    Connects dtype + shape/layout + memory space.
    """
    name: str
    dtype: DType
    layout: Layout
    space: MemorySpace = MemorySpace.GLOBAL

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.layout.shape

    @property
    def byte_size(self) -> int:
        return self.layout.byte_size(self.dtype)

    def to_space(self, space: MemorySpace) -> TensorRef:
        """Derive a new TensorRef in a different memory space."""
        return TensorRef(name=self.name, dtype=self.dtype,
                         layout=self.layout, space=space)

    def __repr__(self) -> str:
        return (
            f"TensorRef('{self.name}', {self.dtype!r}, "
            f"{self.space!r}, shape={self.shape})"
        )


# ---------------------------------------------------------------------------
# Warp roles (for warp specialization)
# ---------------------------------------------------------------------------

class WarpRole(enum.Enum):
    """Role assigned to a warp group in a warp-specialized kernel."""
    PRODUCER = "producer"    # data movement (TMA/async copy)
    CONSUMER = "consumer"    # computation (MMA)
    COMPUTE  = "compute"     # generic compute (epilogue, reductions)
    ANY      = "any"         # not specialized

    def __repr__(self) -> str:
        return f"WarpRole.{self.name}"
