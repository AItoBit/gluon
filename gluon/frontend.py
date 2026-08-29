"""
Gluon Frontend — AST-Based Kernel Tracer
==========================================
Higher-level frontend that can introspect decorated kernel functions
and produce validated Gluon IR.  For now, the primary entry point is
the tracing-based approach in dsl.py; this module provides additional
validation and IR utilities.
"""
from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Callable, Dict, List, Optional

from .types import DType, Layout, TensorRef, WarpRole
from .arch import Arch
from .ir import GluonModule, KernelFunc, IRNode, WarpSpecialize


# ---------------------------------------------------------------------------
# Kernel source inspector
# ---------------------------------------------------------------------------

class KernelInspector:
    """
    Inspects a kernel function's source code and validates structural
    properties before / after tracing.
    """

    def __init__(self, fn: Callable, arch: Arch):
        self.fn = fn
        self.arch = arch
        self.source = textwrap.dedent(inspect.getsource(fn))
        self.tree = ast.parse(self.source)
        self._warnings: List[str] = []
        self._errors: List[str] = []

    @property
    def warnings(self) -> List[str]:
        return list(self._warnings)

    @property
    def errors(self) -> List[str]:
        return list(self._errors)

    def validate_arch_ops(self) -> bool:
        """
        Walk the AST and check that any Gluon DSL call is compatible
        with the target architecture.
        """
        arch_gated = {
            "tma_load":  lambda a: a.has_tma,
            "tma_prefetch": lambda a: a.has_tma,
            "wgmma":     lambda a: a.has_wgmma,
            "mbarrier":  lambda a: a.supports_op("mbarrier_init"),
        }

        class _Visitor(ast.NodeVisitor):
            def __init__(self, inspector: KernelInspector):
                self.inspector = inspector

            def visit_Call(self, node: ast.Call):
                # Check for gluon.xxx() or just xxx() calls
                func_name = None
                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id

                if func_name and func_name in arch_gated:
                    checker = arch_gated[func_name]
                    if not checker(self.inspector.arch):
                        self.inspector._errors.append(
                            f"'{func_name}' is not supported on "
                            f"{self.inspector.arch.name} "
                            f"({self.inspector.arch.compute_capability})"
                        )
                self.generic_visit(node)

        _Visitor(self).visit(self.tree)
        return len(self._errors) == 0


# ---------------------------------------------------------------------------
# IR validation
# ---------------------------------------------------------------------------

def validate_module(module: GluonModule) -> List[str]:
    """
    Perform structural validation on a traced GluonModule.
    Returns a list of error messages (empty = valid).
    """
    errors: List[str] = []

    for kernel in module.kernels:
        # 1. Check thread count
        if kernel.num_threads > kernel.arch.max_threads_per_block:
            errors.append(
                f"Kernel '{kernel.name}': {kernel.num_threads} threads exceeds "
                f"{kernel.arch.name} max of {kernel.arch.max_threads_per_block}"
            )

        # 2. Check smem capacity
        smem_err = kernel.arch.validate_smem(kernel.total_smem_bytes)
        if smem_err:
            errors.append(f"Kernel '{kernel.name}': {smem_err}")

        # 3. Check warp specialization structure
        errors.extend(_validate_warp_spec(kernel))

    return errors


def _validate_warp_spec(kernel: KernelFunc) -> List[str]:
    """Validate warp specialization regions."""
    errors: List[str] = []

    for node in kernel.body:
        if isinstance(node, WarpSpecialize):
            roles_seen = set()
            for role, region in node.regions.items():
                if role in roles_seen:
                    errors.append(
                        f"Kernel '{kernel.name}': duplicate warp role "
                        f"{role.name} in same WarpSpecialize block"
                    )
                roles_seen.add(role)

                # Check warp range bounds
                if region.warp_range:
                    lo, hi = region.warp_range
                    max_warps = kernel.num_warps
                    if hi >= max_warps:
                        errors.append(
                            f"Kernel '{kernel.name}': warp range "
                            f"[{lo}..{hi}] exceeds num_warps={max_warps}"
                        )
    return errors


# ---------------------------------------------------------------------------
# Pretty-print utility
# ---------------------------------------------------------------------------

def dump_ir(module: GluonModule) -> str:
    """Return a formatted string representation of the entire module."""
    return module.pretty()
