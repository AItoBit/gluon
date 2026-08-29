"""
Gluon JIT Compilation Driver
==============================
Takes a GluonModule, runs all passes, lowers to backend IR,
and returns a callable launcher.
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from .ir import GluonModule, KernelFunc
from .frontend import validate_module
from .passes import run_all_passes
from .backend.triton_lowering import TritonLowering


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------

_compilation_cache: Dict[str, 'CompiledKernel'] = {}


class CompiledKernel:
    """A compiled Gluon kernel ready for launch or inspection."""

    def __init__(self, name: str, module: GluonModule,
                 lowered_ir: str, kernel_func: KernelFunc):
        self.name = name
        self.module = module
        self.lowered_ir = lowered_ir
        self.kernel_func = kernel_func

    @property
    def smem_bytes(self) -> int:
        return self.kernel_func.total_smem_bytes

    @property
    def num_warps(self) -> int:
        return self.kernel_func.num_warps

    @property
    def arch(self):
        return self.kernel_func.arch

    def __repr__(self) -> str:
        return (
            f"CompiledKernel('{self.name}', arch={self.arch.name}, "
            f"warps={self.num_warps}, smem={self.smem_bytes}B)"
        )


# ---------------------------------------------------------------------------
# JIT compile function
# ---------------------------------------------------------------------------

def compile(module: GluonModule, *,
            cache: bool = True,
            verbose: bool = False) -> CompiledKernel:
    """
    Compile a GluonModule through the full pipeline.

    Steps:
    1. Validate the IR structure
    2. Run compiler passes (layout check, warp spec, smem alloc, pipeline)
    3. Lower to Triton-compatible IR
    4. Return a CompiledKernel

    Args:
        module: The GluonModule to compile.
        cache: Whether to cache compiled kernels.
        verbose: Print compilation progress.

    Returns:
        CompiledKernel with lowered IR.

    Raises:
        RuntimeError: If validation or pass errors are found.
    """
    if not module.kernels:
        raise RuntimeError("GluonModule has no kernels to compile")

    # Cache key
    ir_text = module.pretty()
    cache_key = hashlib.sha256(ir_text.encode()).hexdigest()[:16]

    if cache and cache_key in _compilation_cache:
        if verbose:
            print(f"[gluon] Cache hit for '{module.name}' ({cache_key})")
        return _compilation_cache[cache_key]

    if verbose:
        print(f"[gluon] Compiling '{module.name}'...")

    # Step 1: Validate IR structure
    if verbose:
        print(f"  [1/3] Validating IR...")
    validation_errors = validate_module(module)
    if validation_errors:
        msg = "\n".join(f"  - {e}" for e in validation_errors)
        raise RuntimeError(f"Gluon IR validation failed:\n{msg}")

    # Step 2: Run compiler passes
    if verbose:
        print(f"  [2/3] Running compiler passes...")
    pass_errors = run_all_passes(module)
    if pass_errors:
        msg = "\n".join(f"  - {e}" for e in pass_errors)
        raise RuntimeError(f"Gluon pass errors:\n{msg}")

    # Step 3: Lower to TTIR
    if verbose:
        print(f"  [3/3] Lowering to TTIR...")
    lowering = TritonLowering()
    lowered = lowering.lower(module)

    kernel_func = module.kernels[0]
    result = CompiledKernel(
        name=kernel_func.name,
        module=module,
        lowered_ir=lowered,
        kernel_func=kernel_func,
    )

    if cache:
        _compilation_cache[cache_key] = result

    if verbose:
        print(f"[gluon] Compilation complete: {result}")

    return result


def clear_cache() -> None:
    """Clear the compilation cache."""
    _compilation_cache.clear()
