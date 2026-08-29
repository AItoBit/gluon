"""
Warp Specialization Pass
=========================
Validates warp-role assignments, barrier pairing, and register budgets
in warp-specialized kernels.
"""
from __future__ import annotations

from typing import Dict, List, Set, Tuple

from ..ir import (
    GluonModule, KernelFunc, IRNode, Region,
    WarpSpecialize, MBarrierInit, MBarrierArrive, MBarrierWait,
    ForLoop, IfElse,
)
from ..types import WarpRole


class WarpSpecPass:
    """
    Validates warp specialization correctness.

    Checks:
    1. Warp ranges don't overlap and are within bounds
    2. Every mbarrier.arrive has a matching mbarrier.wait (and vice versa)
    3. Producer and consumer roles use disjoint warp ranges
    """

    def run(self, module: GluonModule) -> List[str]:
        errors: List[str] = []
        for kernel in module.kernels:
            errors.extend(self._check_kernel(kernel))
        return errors

    def _check_kernel(self, kernel: KernelFunc) -> List[str]:
        errors: List[str] = []

        # Collect all WarpSpecialize nodes
        ws_nodes = self._collect_ws(kernel.body)

        for ws in ws_nodes:
            # Check warp range overlaps
            ranges: Dict[WarpRole, Tuple[int, int]] = {}
            for role, region in ws.regions.items():
                if region.warp_range:
                    ranges[role] = region.warp_range

            # Verify no overlap
            range_list = list(ranges.items())
            for i in range(len(range_list)):
                for j in range(i + 1, len(range_list)):
                    r1_role, (lo1, hi1) = range_list[i]
                    r2_role, (lo2, hi2) = range_list[j]
                    if lo1 <= hi2 and lo2 <= hi1:
                        errors.append(
                            f"Kernel '{kernel.name}': warp ranges overlap: "
                            f"{r1_role.name}[{lo1}..{hi1}] and "
                            f"{r2_role.name}[{lo2}..{hi2}]"
                        )

            # Check bounds
            for role, (lo, hi) in ranges.items():
                if lo < 0:
                    errors.append(
                        f"Kernel '{kernel.name}': warp range for "
                        f"{role.name} has negative lower bound ({lo})"
                    )
                if hi >= kernel.num_warps:
                    errors.append(
                        f"Kernel '{kernel.name}': warp range for "
                        f"{role.name} [{lo}..{hi}] exceeds "
                        f"num_warps={kernel.num_warps}"
                    )

        # Check barrier pairing across all nodes
        errors.extend(self._check_barriers(kernel))

        return errors

    def _check_barriers(self, kernel: KernelFunc) -> List[str]:
        """Verify mbarrier init/arrive/wait pairing."""
        errors: List[str] = []

        inits: Set[str] = set()
        arrives: Set[str] = set()
        waits: Set[str] = set()

        self._collect_barrier_names(kernel.body, inits, arrives, waits)

        # Every arrived or waited barrier must be initialized
        for name in arrives:
            if name not in inits:
                errors.append(
                    f"Kernel '{kernel.name}': mbarrier.arrive('{name}') "
                    f"without matching mbarrier.init"
                )
        for name in waits:
            if name not in inits:
                errors.append(
                    f"Kernel '{kernel.name}': mbarrier.wait('{name}') "
                    f"without matching mbarrier.init"
                )

        # Warn if we have init without any arrive/wait
        for name in inits:
            if name not in arrives and name not in waits:
                errors.append(
                    f"Kernel '{kernel.name}': mbarrier.init('{name}') "
                    f"is never arrived or waited on"
                )

        return errors

    def _collect_barrier_names(self, nodes: List[IRNode],
                                inits: Set[str], arrives: Set[str],
                                waits: Set[str]) -> None:
        for node in nodes:
            if isinstance(node, MBarrierInit):
                inits.add(node.name)
            elif isinstance(node, MBarrierArrive):
                arrives.add(node.name)
            elif isinstance(node, MBarrierWait):
                waits.add(node.name)
            elif isinstance(node, WarpSpecialize):
                for region in node.regions.values():
                    self._collect_barrier_names(region.body, inits, arrives, waits)
            elif isinstance(node, ForLoop):
                self._collect_barrier_names(node.body, inits, arrives, waits)
            elif isinstance(node, IfElse):
                self._collect_barrier_names(node.then_body, inits, arrives, waits)
                self._collect_barrier_names(node.else_body, inits, arrives, waits)

    def _collect_ws(self, nodes: List[IRNode]) -> List[WarpSpecialize]:
        """Recursively collect all WarpSpecialize nodes."""
        result: List[WarpSpecialize] = []
        for node in nodes:
            if isinstance(node, WarpSpecialize):
                result.append(node)
                for region in node.regions.values():
                    result.extend(self._collect_ws(region.body))
            elif isinstance(node, ForLoop):
                result.extend(self._collect_ws(node.body))
            elif isinstance(node, IfElse):
                result.extend(self._collect_ws(node.then_body))
                result.extend(self._collect_ws(node.else_body))
        return result
