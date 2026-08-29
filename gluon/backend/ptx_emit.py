"""
PTX Emission Helpers
=====================
Architecture-specific inline PTX assembly templates for Gluon ops
that have no high-level LLVM/Triton equivalent.
"""
from __future__ import annotations

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# PTX instruction templates
# ---------------------------------------------------------------------------

class PTXTemplate:
    """An inline PTX asm template with input/output constraints."""

    def __init__(self, asm: str, *, comment: str = ""):
        self.asm = asm
        self.comment = comment

    def __repr__(self) -> str:
        return f"PTX({self.comment or self.asm[:40]})"


# ===================================================================
#  HOPPER (sm_90) — WGMMA, TMA, MBarrier
# ===================================================================

WGMMA_M64N256K16_F16_F32 = PTXTemplate(
    asm="""\
// Warp Group MMA m64n256k16  f16×f16 → f32
// Operand A: shared memory (swizzled), Operand B: shared memory (swizzled)
// Accumulator D: registers (256 f32 values per warp group)
wgmma.mma_async.sync.aligned.m64n256k16.f32.f16.f16
    {{%d0,  %d1,  %d2,  %d3,  %d4,  %d5,  %d6,  %d7,
      %d8,  %d9,  %d10, %d11, %d12, %d13, %d14, %d15,
      %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23,
      %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31,
      %d32, %d33, %d34, %d35, %d36, %d37, %d38, %d39,
      %d40, %d41, %d42, %d43, %d44, %d45, %d46, %d47,
      %d48, %d49, %d50, %d51, %d52, %d53, %d54, %d55,
      %d56, %d57, %d58, %d59, %d60, %d61, %d62, %d63,
      %d64, %d65, %d66, %d67, %d68, %d69, %d70, %d71,
      %d72, %d73, %d74, %d75, %d76, %d77, %d78, %d79,
      %d80, %d81, %d82, %d83, %d84, %d85, %d86, %d87,
      %d88, %d89, %d90, %d91, %d92, %d93, %d94, %d95,
      %d96, %d97, %d98, %d99, %d100,%d101,%d102,%d103,
      %d104,%d105,%d106,%d107,%d108,%d109,%d110,%d111,
      %d112,%d113,%d114,%d115,%d116,%d117,%d118,%d119,
      %d120,%d121,%d122,%d123,%d124,%d125,%d126,%d127}},
    desc_a, desc_b, scale_d, imm_scale_a, imm_scale_b;""",
    comment="wgmma m64n256k16 f16→f32",
)

WGMMA_M64N128K16_F16_F32 = PTXTemplate(
    asm="""\
wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
    {{%d0,  %d1,  %d2,  %d3,  %d4,  %d5,  %d6,  %d7,
      %d8,  %d9,  %d10, %d11, %d12, %d13, %d14, %d15,
      %d16, %d17, %d18, %d19, %d20, %d21, %d22, %d23,
      %d24, %d25, %d26, %d27, %d28, %d29, %d30, %d31,
      %d32, %d33, %d34, %d35, %d36, %d37, %d38, %d39,
      %d40, %d41, %d42, %d43, %d44, %d45, %d46, %d47,
      %d48, %d49, %d50, %d51, %d52, %d53, %d54, %d55,
      %d56, %d57, %d58, %d59, %d60, %d61, %d62, %d63}},
    desc_a, desc_b, scale_d, imm_scale_a, imm_scale_b;""",
    comment="wgmma m64n128k16 f16→f32",
)


# --- TMA ---

TMA_LOAD_2D = PTXTemplate(
    asm="""\
// TMA 2D load: global → shared via tensor descriptor
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_ptr], [tensor_desc, {{coord_x, coord_y}}], [mbar_ptr];""",
    comment="TMA 2D load",
)

TMA_LOAD_MULTICAST_2D = PTXTemplate(
    asm="""\
// TMA 2D load with multicast (cluster)
cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster
    [smem_ptr], [tensor_desc, {{coord_x, coord_y}}], [mbar_ptr], multicast_mask;""",
    comment="TMA 2D multicast load",
)

TMA_PREFETCH_2D = PTXTemplate(
    asm="""\
cp.async.bulk.prefetch.tensor.2d.L2.global
    [tensor_desc, {{coord_x, coord_y}}];""",
    comment="TMA 2D L2 prefetch",
)


# --- MBarrier ---

MBARRIER_INIT = PTXTemplate(
    asm="""\
mbarrier.init.shared::cta.b64 [mbar_ptr], expected_count;""",
    comment="mbarrier init",
)

MBARRIER_ARRIVE = PTXTemplate(
    asm="""\
mbarrier.arrive.shared::cta.b64 state, [mbar_ptr];""",
    comment="mbarrier arrive",
)

MBARRIER_ARRIVE_EXPECT_TX = PTXTemplate(
    asm="""\
mbarrier.arrive.expect_tx.shared::cta.b64 [mbar_ptr], tx_count;""",
    comment="mbarrier arrive with expected tx bytes",
)

MBARRIER_WAIT_PARITY = PTXTemplate(
    asm="""\
// Spin-wait on mbarrier phase parity
{{
    .reg .pred  p;
    LAB_WAIT:
    mbarrier.try_wait.parity.shared::cta.b64  p, [mbar_ptr], phase_bit;
    @!p bra     LAB_WAIT;
}}""",
    comment="mbarrier wait parity",
)


# --- Fences ---

FENCE_ASYNC_SHARED = PTXTemplate(
    asm="fence.proxy.async.shared::cta;",
    comment="fence async shared",
)

FENCE_MMA_ASYNC = PTXTemplate(
    asm="wgmma.fence.sync.aligned;",
    comment="wgmma fence",
)

WGMMA_COMMIT_GROUP = PTXTemplate(
    asm="wgmma.commit_group.sync.aligned;",
    comment="wgmma commit group",
)

WGMMA_WAIT_GROUP = PTXTemplate(
    asm="wgmma.wait_group.sync.aligned {n_prior};",
    comment="wgmma wait group",
)


# --- Register control ---

SETMAXNREG_INC = PTXTemplate(
    asm="setmaxnreg.inc.sync.aligned.u32 {delta};",
    comment="increase max register count",
)

SETMAXNREG_DEC = PTXTemplate(
    asm="setmaxnreg.dec.sync.aligned.u32 {delta};",
    comment="decrease max register count",
)


# ===================================================================
#  AMPERE (sm_80) — HMMA, ldmatrix, cp.async
# ===================================================================

HMMA_M16N8K16_F16_F32 = PTXTemplate(
    asm="""\
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
    {{%d0, %d1, %d2, %d3}},
    {{%a0, %a1, %a2, %a3}},
    {{%b0, %b1}},
    {{%c0, %c1, %c2, %c3}};""",
    comment="hmma m16n8k16 f16→f32",
)

LDMATRIX_X4 = PTXTemplate(
    asm="""\
ldmatrix.sync.aligned.m8n8.x4.shared.b16
    {{%r0, %r1, %r2, %r3}}, [smem_ptr];""",
    comment="ldmatrix x4",
)

CP_ASYNC_CG = PTXTemplate(
    asm="""\
cp.async.cg.shared.global [dst_smem], [src_global], 16;""",
    comment="cp.async 16B cache-global",
)

CP_ASYNC_COMMIT_GROUP = PTXTemplate(
    asm="cp.async.commit_group;",
    comment="cp.async commit group",
)

CP_ASYNC_WAIT_GROUP = PTXTemplate(
    asm="cp.async.wait_group {n_prior};",
    comment="cp.async wait group",
)


# ===================================================================
#  AMD CDNA3 (gfx942) — MFMA
# ===================================================================

MFMA_F32_16X16X16_F16 = PTXTemplate(
    asm="""\
// AMD Matrix Fused Multiply-Add
v_mfma_f32_16x16x16_f16 v[vd:vd+3], v[va:va+1], v[vb:vb+1], v[vc:vc+3];""",
    comment="mfma f32_16x16x16_f16",
)

MFMA_F32_32X32X8_F16 = PTXTemplate(
    asm="""\
v_mfma_f32_32x32x8_f16 v[vd:vd+15], v[va:va+1], v[vb:vb+1], v[vc:vc+15];""",
    comment="mfma f32_32x32x8_f16",
)


# ===================================================================
#  Template registry
# ===================================================================

PTX_TEMPLATES: Dict[str, Dict[str, PTXTemplate]] = {
    "sm_90": {
        "wgmma_m64n256k16_f16":   WGMMA_M64N256K16_F16_F32,
        "wgmma_m64n128k16_f16":   WGMMA_M64N128K16_F16_F32,
        "tma_load_2d":            TMA_LOAD_2D,
        "tma_load_multicast_2d":  TMA_LOAD_MULTICAST_2D,
        "tma_prefetch_2d":        TMA_PREFETCH_2D,
        "mbarrier_init":          MBARRIER_INIT,
        "mbarrier_arrive":        MBARRIER_ARRIVE,
        "mbarrier_arrive_tx":     MBARRIER_ARRIVE_EXPECT_TX,
        "mbarrier_wait":          MBARRIER_WAIT_PARITY,
        "fence_async_shared":     FENCE_ASYNC_SHARED,
        "fence_mma":              FENCE_MMA_ASYNC,
        "wgmma_commit":           WGMMA_COMMIT_GROUP,
        "wgmma_wait":             WGMMA_WAIT_GROUP,
        "setmaxnreg_inc":         SETMAXNREG_INC,
        "setmaxnreg_dec":         SETMAXNREG_DEC,
    },
    "sm_100": {
        # Blackwell inherits Hopper templates + adds new ones
        **{k: v for k, v in {
            "wgmma_m64n256k16_f16":   WGMMA_M64N256K16_F16_F32,
            "wgmma_m64n128k16_f16":   WGMMA_M64N128K16_F16_F32,
            "tma_load_2d":            TMA_LOAD_2D,
            "tma_load_multicast_2d":  TMA_LOAD_MULTICAST_2D,
            "tma_prefetch_2d":        TMA_PREFETCH_2D,
            "mbarrier_init":          MBARRIER_INIT,
            "mbarrier_arrive":        MBARRIER_ARRIVE,
            "mbarrier_arrive_tx":     MBARRIER_ARRIVE_EXPECT_TX,
            "mbarrier_wait":          MBARRIER_WAIT_PARITY,
            "fence_async_shared":     FENCE_ASYNC_SHARED,
            "fence_mma":              FENCE_MMA_ASYNC,
            "wgmma_commit":           WGMMA_COMMIT_GROUP,
            "wgmma_wait":             WGMMA_WAIT_GROUP,
            "setmaxnreg_inc":         SETMAXNREG_INC,
            "setmaxnreg_dec":         SETMAXNREG_DEC,
        }.items()},
    },
    "sm_80": {
        "hmma_m16n8k16_f16":      HMMA_M16N8K16_F16_F32,
        "ldmatrix_x4":            LDMATRIX_X4,
        "cp_async_cg":            CP_ASYNC_CG,
        "cp_async_commit":        CP_ASYNC_COMMIT_GROUP,
        "cp_async_wait":          CP_ASYNC_WAIT_GROUP,
    },
    "gfx942": {
        "mfma_f32_16x16x16_f16":  MFMA_F32_16X16X16_F16,
        "mfma_f32_32x32x8_f16":   MFMA_F32_32X32X8_F16,
    },
}


def get_ptx_template(arch_cc: str, op_name: str) -> Optional[PTXTemplate]:
    """Look up a PTX template by architecture and operation name."""
    templates = PTX_TEMPLATES.get(arch_cc, {})
    return templates.get(op_name)


def list_templates(arch_cc: str) -> list[str]:
    """List available template names for an architecture."""
    return sorted(PTX_TEMPLATES.get(arch_cc, {}).keys())
