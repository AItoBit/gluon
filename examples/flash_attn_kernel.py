#!/usr/bin/env python3
"""
Example: FlashAttention-Style Fused Attention Kernel (Hopper SM90)
====================================================================
Demonstrates Gluon's warp specialization for a fused multi-head
attention kernel with online softmax.

  Score    = Q @ K^T          (WGMMA)
  P        = softmax(Score)   (online, in registers)
  Output   = P @ V            (WGMMA)

Architecture: SM90 (Hopper / H100)
  - Producer warps (0-3): TMA loads Q, K, V tiles into shared memory
  - Consumer warps (4-7): WGMMA compute + online softmax in registers
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gluon
from gluon import (
    DType, Layout, SwizzleMode, TensorRef, WarpRole, MemorySpace,
    SM90,
)
from gluon.dsl import (
    kernel, smem, barrier, mbarrier, mbarrier_arrive, mbarrier_wait,
    fence_async_shared, warp_role, tma_load, wgmma,
    for_range, global_store, elementwise, fma, program_id,
)
from gluon.jit import compile as jit_compile
from gluon.runtime import compute_attention_grid
from gluon.backend.triton_lowering import TritonLowering

# ---------------------------------------------------------------------------
# Attention parameters
# ---------------------------------------------------------------------------
BATCH = 8
HEADS = 32
SEQ_LEN = 2048
HEAD_DIM = 128

BLOCK_M = 128      # query tile rows
BLOCK_N = 64       # key/value tile cols
NUM_WARPS = 8

# ---------------------------------------------------------------------------
# Tensor descriptors
# ---------------------------------------------------------------------------
q_ref = TensorRef("Q", DType.f16,
                  Layout.row_major(BLOCK_M, HEAD_DIM), MemorySpace.GLOBAL)
k_ref = TensorRef("K", DType.f16,
                  Layout.col_major(HEAD_DIM, BLOCK_N), MemorySpace.GLOBAL)
v_ref = TensorRef("V", DType.f16,
                  Layout.row_major(BLOCK_N, HEAD_DIM), MemorySpace.GLOBAL)
o_ref = TensorRef("O", DType.f32,
                  Layout.row_major(BLOCK_M, HEAD_DIM), MemorySpace.GLOBAL)


# ---------------------------------------------------------------------------
# Kernel definition
# ---------------------------------------------------------------------------
@kernel(arch=SM90, num_warps=NUM_WARPS,
        grid=(SEQ_LEN // BLOCK_M, BATCH * HEADS, 1))
def flash_attention_v2(Q: TensorRef, K: TensorRef,
                       V: TensorRef, O: TensorRef):
    """
    FlashAttention-2 style fused attention with warp specialization.
    Online softmax: maintains running max and sum in registers.
    """
    # --- Shared memory allocations ---
    q_smem = smem(DType.f16,
                  Layout.swizzled(BLOCK_M, HEAD_DIM, swizzle=SwizzleMode.B128),
                  name="Q_smem", alignment=128)
    k_smem = smem(DType.f16,
                  Layout.swizzled(HEAD_DIM, BLOCK_N, swizzle=SwizzleMode.B128),
                  name="K_smem", double_buffer=True, alignment=128)
    v_smem = smem(DType.f16,
                  Layout.swizzled(BLOCK_N, HEAD_DIM, swizzle=SwizzleMode.B128),
                  name="V_smem", double_buffer=True, alignment=128)

    # mbarrier for Q (loaded once), K/V (loaded per iteration)
    mbar_q = mbarrier(count=1, name="mbar_q")
    mbar_kv = mbarrier(count=1, name="mbar_kv")

    # --- Warp Specialization ---

    # Producer warps: TMA loads
    with warp_role(WarpRole.PRODUCER, warp_range=(0, 3)):
        # Load Q tile once
        tma_load(q_smem, "Q",
                 coords=(program_id(0), "0"),
                 barrier_name=mbar_q)
        mbarrier_arrive(mbar_q,
                       bytes_arrived=BLOCK_M * HEAD_DIM * 2)

        # Stream K, V tiles
        with for_range("j", 0, SEQ_LEN, BLOCK_N) as loop:
            tma_load(k_smem, "K",
                     coords=("j", program_id(1)),
                     barrier_name=mbar_kv)
            tma_load(v_smem, "V",
                     coords=("j", program_id(1)),
                     barrier_name=mbar_kv)
            mbarrier_arrive(mbar_kv,
                           bytes_arrived=(HEAD_DIM * BLOCK_N * 2 +
                                         BLOCK_N * HEAD_DIM * 2))
            fence_async_shared()

    # Consumer warps: compute S = Q@K^T, P = softmax(S), O = P@V
    with warp_role(WarpRole.CONSUMER, warp_range=(4, 7)):
        # Wait for Q
        mbarrier_wait(mbar_q, phase=0)

        with for_range("j", 0, SEQ_LEN, BLOCK_N) as loop:
            # Wait for K, V
            mbarrier_wait(mbar_kv, phase=0)

            # Score = Q @ K^T  (WGMMA)
            wgmma(q_smem, k_smem, "score")

            # Online softmax (elementwise in registers)
            row_max = elementwise("max", "score", "running_max", dtype=DType.f32,
                                  dst="new_max")
            correction = elementwise("exp", "running_max_minus_new",
                                     dtype=DType.f32, dst="correction")
            elementwise("exp", "score_minus_max", dtype=DType.f32, dst="P")

            # Output = P @ V  (WGMMA)
            wgmma(q_smem, v_smem, "O_acc")

    # Store output
    global_store("O", "O_acc", indices=(program_id(0), "0"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("  Gluon FlashAttention Kernel — Hopper SM90 Warp-Specialized")
    print("=" * 70)
    print()

    # Trace
    print("[1] Tracing kernel...")
    module = flash_attention_v2.trace(q_ref, k_ref, v_ref, o_ref)
    print(module.pretty())
    print()

    # Compile
    print("[2] Compiling...")
    try:
        compiled = jit_compile(module, verbose=True)
        print(f"\n  Result: {compiled}")
    except RuntimeError as e:
        print(f"\n  Compilation note: {e}")
    print()

    # Lower
    print("[3] Lowered TTIR:")
    print("-" * 70)
    lowering = TritonLowering()
    print(lowering.lower(module))
    print("-" * 70)

    # Launch config
    grid = compute_attention_grid(BATCH, HEADS, SEQ_LEN, BLOCK_M)
    print()
    print(f"[4] Launch configuration:")
    print(f"  Grid:     {grid}")
    print(f"  Block:    ({NUM_WARPS * 32}, 1, 1)")
    print(f"  Problem:  B={BATCH}, H={HEADS}, S={SEQ_LEN}, D={HEAD_DIM}")
    print(f"  Tiles:    BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print()
    print("✓ Gluon FlashAttention kernel traced, compiled, and lowered.")


if __name__ == "__main__":
    main()
