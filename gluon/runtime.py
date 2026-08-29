"""
Gluon Runtime — Kernel Launcher & Argument Marshalling
========================================================
Wraps compiled kernels with launch configuration and argument binding.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

from .jit import CompiledKernel, compile as jit_compile
from .ir import GluonModule


@dataclass
class LaunchConfig:
    """Grid and block dimensions for kernel launch."""
    grid: Tuple[int, ...] = (1, 1, 1)
    block: Tuple[int, ...] = (256, 1, 1)
    smem_bytes: int = 0
    stream: Any = None           # GPU stream handle (opaque)

    def __repr__(self) -> str:
        return (
            f"Launch(grid={self.grid}, block={self.block}, "
            f"smem={self.smem_bytes}B)"
        )


class GluonKernel:
    """
    A launchable Gluon kernel.
    Combines a compiled kernel with launch configuration.
    """

    def __init__(self, compiled: CompiledKernel, *,
                 grid: Optional[Tuple[int, ...]] = None):
        self.compiled = compiled
        self.grid = grid or (1, 1, 1)
        self._config = LaunchConfig(
            grid=self.grid,
            block=(compiled.num_warps * compiled.arch.warp_size, 1, 1),
            smem_bytes=compiled.smem_bytes,
        )

    @property
    def config(self) -> LaunchConfig:
        return self._config

    @property
    def name(self) -> str:
        return self.compiled.name

    def launch(self, *args, **kwargs) -> None:
        """
        Launch the kernel (stub — actual GPU launch requires CUDA/HIP runtime).

        On real hardware:
        1. Marshal tensor arguments to device pointers
        2. Set dynamic shared memory size
        3. Launch via cuLaunchKernel or hipLaunchKernel

        In this implementation, prints launch info for verification.
        """
        print(f"[gluon.launch] Kernel '{self.name}'")
        print(f"  Grid:  {self._config.grid}")
        print(f"  Block: {self._config.block}")
        print(f"  Smem:  {self._config.smem_bytes} bytes")
        print(f"  Arch:  {self.compiled.arch.name} "
              f"({self.compiled.arch.compute_capability})")
        print(f"  Args:  {len(args)} tensors")

    def __repr__(self) -> str:
        return (
            f"GluonKernel('{self.name}', "
            f"arch={self.compiled.arch.name}, "
            f"{self._config})"
        )


def build_and_launch(module: GluonModule, *args,
                     grid: Optional[Tuple[int, ...]] = None,
                     verbose: bool = False, **kwargs) -> GluonKernel:
    """
    Convenience: compile a module and immediately launch it.
    """
    compiled = jit_compile(module, verbose=verbose)
    kernel = GluonKernel(compiled, grid=grid)
    kernel.launch(*args, **kwargs)
    return kernel


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return (a + b - 1) // b


def compute_gemm_grid(M: int, N: int, BLOCK_M: int, BLOCK_N: int) -> Tuple[int, int, int]:
    """Compute grid dimensions for a GEMM kernel."""
    return (cdiv(M, BLOCK_M), cdiv(N, BLOCK_N), 1)


def compute_attention_grid(batch: int, heads: int, seq_len: int,
                           BLOCK_M: int) -> Tuple[int, int, int]:
    """Compute grid dimensions for a FlashAttention-style kernel."""
    return (cdiv(seq_len, BLOCK_M), batch * heads, 1)
