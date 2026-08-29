"""
Layout Check Pass
==================
Validates that all tensor references and shared memory descriptors
have explicit layouts, and checks layout compatibility with the
target architecture's tensor-core requirements.
"""
from __future__ import annotations

from typing import List

from ..ir import (
    GluonModule, KernelFunc, IRNode, SmemAlloc, SmemLoad, SmemStore,
    TMALoad, WGMMA, HMMA, MFMA, ForLoop, IfElse, WarpSpecialize,
)
from ..types import Layout, SharedMemoryDescriptor, SwizzleMode


class LayoutCheckPass:
    """
    Validates layout correctness across a GluonModule.

    Checks:
    1. All SmemAlloc nodes have non-degenerate layouts
    2. Shared memory layouts match tensor-core alignment requirements
    3. Bank conflict warnings for un-swizzled layouts
    """

    def run(self, module: GluonModule) -> List[str]:
        errors: List[str] = []
        for kernel in module.kernels:
            errors.extend(self._check_kernel(kernel))
        return errors

    def _check_kernel(self, kernel: KernelFunc) -> List[str]:
        errors: List[str] = []

        # Check all kernel parameters have layouts
        for param in kernel.params:
            if param.layout.numel == 0:
                errors.append(
                    f"Kernel '{kernel.name}': parameter '{param.name}' "
                    f"has zero-element layout"
                )

        # Walk IR nodes
        errors.extend(self._walk_nodes(kernel.name, kernel.body, kernel))
        return errors

    def _walk_nodes(self, kernel_name: str, nodes: List[IRNode],
                    kernel: KernelFunc) -> List[str]:
        errors: List[str] = []

        for node in nodes:
            if isinstance(node, SmemAlloc):
                errors.extend(self._check_smem_alloc(kernel_name, node, kernel))
            elif isinstance(node, (WGMMA, HMMA, MFMA)):
                errors.extend(self._check_tc_layout(kernel_name, node, kernel))
            elif isinstance(node, ForLoop):
                errors.extend(self._walk_nodes(kernel_name, node.body, kernel))
            elif isinstance(node, IfElse):
                errors.extend(self._walk_nodes(kernel_name, node.then_body, kernel))
                errors.extend(self._walk_nodes(kernel_name, node.else_body, kernel))
            elif isinstance(node, WarpSpecialize):
                for region in node.regions.values():
                    errors.extend(self._walk_nodes(kernel_name, region.body, kernel))

        return errors

    def _check_smem_alloc(self, kernel_name: str, node: SmemAlloc,
                          kernel: KernelFunc) -> List[str]:
        errors: List[str] = []
        desc = node.descriptor

        # Check non-zero layout
        if desc.layout.numel == 0:
            errors.append(
                f"Kernel '{kernel_name}': smem '{desc.name}' has zero elements"
            )

        # Alignment check
        if desc.alignment <= 0 or (desc.alignment & (desc.alignment - 1)) != 0:
            errors.append(
                f"Kernel '{kernel_name}': smem '{desc.name}' alignment "
                f"({desc.alignment}) must be a power of 2"
            )

        # Bank conflict detection
        warnings = desc.check_bank_conflicts()
        for w in warnings:
            errors.append(
                f"Kernel '{kernel_name}': smem '{desc.name}': {w}"
            )

        return errors

    def _check_tc_layout(self, kernel_name: str, node: IRNode,
                         kernel: KernelFunc) -> List[str]:
        """
        Verify that tensor-core ops reference smem regions with
        compatible layouts for the target architecture.
        """
        errors: List[str] = []
        arch = kernel.arch

        if isinstance(node, WGMMA):
            # WGMMA on Hopper prefers B128 swizzle for smem operands
            if arch.preferred_swizzle != SwizzleMode.NONE:
                for smem_name in [node.a_smem, node.b_smem]:
                    desc = self._find_smem(smem_name, kernel)
                    if desc and desc.layout.swizzle == SwizzleMode.NONE:
                        errors.append(
                            f"Kernel '{kernel_name}': WGMMA operand "
                            f"'{smem_name}' has no swizzle but {arch.name} "
                            f"prefers {arch.preferred_swizzle.value} for "
                            f"bank-conflict-free access"
                        )
        return errors

    def _find_smem(self, name: str,
                   kernel: KernelFunc) -> SharedMemoryDescriptor | None:
        """Find a shared memory descriptor by name in the kernel."""
        for desc in kernel.smem_descriptors:
            if desc.name == name:
                return desc
        return None
