"""
Shared Memory Allocation Pass
===============================
Computes total shared memory footprint, validates against the target
architecture's capacity, and assigns byte offsets with alignment padding.
"""
from __future__ import annotations

import math
from typing import List

from ..ir import (
    GluonModule, KernelFunc, IRNode, SmemAlloc,
    ForLoop, IfElse, WarpSpecialize,
)


class SmemAllocPass:
    """
    Assigns shared-memory byte offsets and validates total usage.

    Each SmemAlloc node gets an `offset` field set to its starting
    byte address in the shared memory block.  Offsets are padded
    to each descriptor's alignment requirement.
    """

    def run(self, module: GluonModule) -> List[str]:
        errors: List[str] = []
        for kernel in module.kernels:
            errors.extend(self._allocate(kernel))
        return errors

    def _allocate(self, kernel: KernelFunc) -> List[str]:
        errors: List[str] = []
        allocs = self._collect_allocs(kernel.body)

        current_offset = 0
        for alloc in allocs:
            desc = alloc.descriptor
            align = desc.alignment

            # Pad to alignment boundary
            if current_offset % align != 0:
                current_offset = int(math.ceil(current_offset / align) * align)

            alloc.offset = current_offset
            current_offset += desc.total_bytes

        # Update kernel's smem descriptors with offsets
        kernel.smem_descriptors = [a.descriptor for a in allocs]

        # Validate against arch capacity
        total = current_offset
        err = kernel.arch.validate_smem(total)
        if err:
            errors.append(f"Kernel '{kernel.name}': {err}")
        else:
            # Store total for runtime launch config
            kernel._total_smem_bytes = total

        return errors

    def _collect_allocs(self, nodes: List[IRNode]) -> List[SmemAlloc]:
        """Collect all SmemAlloc nodes in traversal order."""
        result: List[SmemAlloc] = []
        for node in nodes:
            if isinstance(node, SmemAlloc):
                result.append(node)
            elif isinstance(node, WarpSpecialize):
                for region in node.regions.values():
                    result.extend(self._collect_allocs(region.body))
            elif isinstance(node, ForLoop):
                result.extend(self._collect_allocs(node.body))
            elif isinstance(node, IfElse):
                result.extend(self._collect_allocs(node.then_body))
                result.extend(self._collect_allocs(node.else_body))
        return result
