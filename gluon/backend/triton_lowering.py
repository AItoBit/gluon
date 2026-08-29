"""
Triton-Compatible Lowering
===========================
Lowers Gluon IR to a pseudo-TTIR (Triton IR) representation.
For ops that Triton doesn't natively support, emits inline PTX stubs.
"""
from __future__ import annotations

from typing import List

from ..ir import (
    GluonModule, KernelFunc, IRNode,
    SmemAlloc, SmemStore, SmemLoad,
    GlobalLoad, GlobalStore,
    TMALoad, TMAPrefetch,
    WGMMA, HMMA, MFMA, FMA, ElementwiseOp,
    Barrier, MBarrierInit, MBarrierArrive, MBarrierWait, FenceAsyncShared,
    ForLoop, IfElse, WarpSpecialize, Region,
)
from ..types import WarpRole
from .ptx_emit import get_ptx_template


class TritonLowering:
    """
    Lowers Gluon IR to a structured textual representation that
    mirrors Triton's TTIR dialect.

    - Standard ops (loads, stores, barriers) → Triton-style ops
    - Architecture-specific ops (WGMMA, TMA) → inline_asm blocks
    """

    def __init__(self):
        self._output: List[str] = []
        self._indent = 0

    def lower(self, module: GluonModule) -> str:
        """Lower an entire GluonModule and return the TTIR text."""
        self._output = []
        self._indent = 0
        self._emit(f"// Gluon → TTIR lowering")
        self._emit(f'module @{module.name} {{')
        self._indent += 1
        for kernel in module.kernels:
            self._lower_kernel(kernel)
        self._indent -= 1
        self._emit("}")
        return "\n".join(self._output)

    def _emit(self, line: str) -> None:
        self._output.append("  " * self._indent + line)

    def _lower_kernel(self, kernel: KernelFunc) -> None:
        params = ", ".join(
            f"%{p.name}: tensor<{p.dtype.short}>"
            for p in kernel.params
        )
        self._emit(f"tt.func @{kernel.name}({params}) {{")
        self._emit(f"  // arch: {kernel.arch.name} "
                   f"({kernel.arch.compute_capability})")
        self._emit(f"  // warps: {kernel.num_warps}, "
                   f"smem: {kernel.total_smem_bytes}B")
        self._indent += 1

        for node in kernel.body:
            self._lower_node(node, kernel)

        self._emit("tt.return")
        self._indent -= 1
        self._emit("}")
        self._emit("")

    def _lower_node(self, node: IRNode, kernel: KernelFunc) -> None:
        cc = kernel.arch.compute_capability

        if isinstance(node, SmemAlloc):
            self._emit(
                f"%{node.descriptor.name} = tt.alloc_shared "
                f"<{node.descriptor.dtype.short}x"
                f"{'x'.join(str(s) for s in node.descriptor.layout.shape)}> "
                f": !smem  // {node.descriptor.total_bytes}B"
            )

        elif isinstance(node, SmemLoad):
            self._emit(f"%{node.dst} = tt.load %{node.src} : !smem → !reg")

        elif isinstance(node, SmemStore):
            self._emit(f"tt.store %{node.dst}, %{node.src} : !reg → !smem")

        elif isinstance(node, GlobalLoad):
            self._emit(f"%{node.dst} = tt.load %{node.src} : !global → !reg")

        elif isinstance(node, GlobalStore):
            self._emit(f"tt.store %{node.dst}, %{node.src} : !reg → !global")

        elif isinstance(node, TMALoad):
            # TMA has no Triton equivalent — emit inline PTX
            tmpl = get_ptx_template(cc, "tma_load_2d")
            if tmpl:
                self._emit(f"// TMA load: {node.src_global} → {node.dst_smem}")
                self._emit(f'tt.inline_asm "{tmpl.asm[:60]}..."')
            else:
                self._emit(f"// WARNING: no TMA template for {cc}")

        elif isinstance(node, TMAPrefetch):
            tmpl = get_ptx_template(cc, "tma_prefetch_2d")
            if tmpl:
                self._emit(f'tt.inline_asm "{tmpl.comment}"')

        elif isinstance(node, WGMMA):
            tmpl = get_ptx_template(cc, "wgmma_m64n256k16_f16")
            shape_str = ""
            if node.shape:
                shape_str = f" m{node.shape.m}n{node.shape.n}k{node.shape.k}"
            self._emit(
                f"%{node.c_reg} = tt.inline_asm \"wgmma\"{shape_str} "
                f"(%{node.a_smem}, %{node.b_smem})"
            )

        elif isinstance(node, HMMA):
            self._emit(
                f"%{node.c} = tt.dot %{node.a}, %{node.b} "
                f": !f16 → !f32  // hmma"
            )

        elif isinstance(node, MFMA):
            tmpl = get_ptx_template(cc, "mfma_f32_16x16x16_f16")
            self._emit(
                f"%{node.c} = tt.inline_asm \"mfma\" "
                f"(%{node.a}, %{node.b})"
            )

        elif isinstance(node, FMA):
            self._emit(
                f"%{node.dst} = arith.fma %{node.a}, %{node.b}, %{node.c} "
                f": {node.dtype.short}"
            )

        elif isinstance(node, ElementwiseOp):
            args = ", ".join(f"%{o}" for o in node.operands)
            self._emit(
                f"%{node.dst} = tt.{node.op} ({args}) : {node.dtype.short}"
            )

        elif isinstance(node, Barrier):
            self._emit("gpu.barrier")

        elif isinstance(node, MBarrierInit):
            self._emit(
                f'tt.inline_asm "mbarrier.init" '
                f'// {node.name}, count={node.count}'
            )

        elif isinstance(node, MBarrierArrive):
            self._emit(
                f'tt.inline_asm "mbarrier.arrive" '
                f'// {node.name}'
            )

        elif isinstance(node, MBarrierWait):
            self._emit(
                f'tt.inline_asm "mbarrier.wait" '
                f'// {node.name}, phase={node.phase}'
            )

        elif isinstance(node, FenceAsyncShared):
            self._emit('tt.inline_asm "fence.proxy.async.shared::cta"')

        elif isinstance(node, ForLoop):
            self._emit(
                f"scf.for %{node.var} = {node.start} to {node.end} "
                f"step {node.step} {{"
            )
            self._indent += 1
            for child in node.body:
                self._lower_node(child, kernel)
            self._indent -= 1
            self._emit("}")

        elif isinstance(node, IfElse):
            self._emit(f"scf.if %{node.condition} {{")
            self._indent += 1
            for child in node.then_body:
                self._lower_node(child, kernel)
            self._indent -= 1
            if node.else_body:
                self._emit("} else {")
                self._indent += 1
                for child in node.else_body:
                    self._lower_node(child, kernel)
                self._indent -= 1
            self._emit("}")

        elif isinstance(node, WarpSpecialize):
            self._emit("// --- warp_specialize ---")
            for role, region in node.regions.items():
                wr = ""
                if region.warp_range:
                    wr = f" warps=[{region.warp_range[0]}..{region.warp_range[1]}]"
                self._emit(f"// region @{role.name}{wr}")
                self._indent += 1
                for child in region.body:
                    self._lower_node(child, kernel)
                self._indent -= 1
            self._emit("// --- end warp_specialize ---")

        else:
            self._emit(f"// UNKNOWN: {node.__class__.__name__}")
