# Gluon DSL

**Low-level GPU kernel DSL with explicit layouts, shared memory, and warp specialization.**

Gluon shares Triton's compiler stack but deliberately trades portability for peak performance. Where Triton abstracts away hardware details, Gluon makes them first-class constructs that the programmer controls directly.

## Key Features

- **Explicit data layouts** — row-major, column-major, swizzled (B32/B64/B128)
- **Explicit shared memory descriptors** — typed regions with bank-conflict analysis
- **Explicit warp specialization** — producer/consumer roles with `warp_role()` context manager
- **Architecture-specific ops** — WGMMA (Hopper), HMMA (Ampere), MFMA (CDNA3)
- **Python-embedded DSL** — `@gluon.kernel` decorator, no custom syntax
- **4 compiler passes** — layout validation, warp-spec checking, smem allocation, software pipelining
- **JIT compilation** — validate → passes → lower → cache

## Supported Architectures

| Arch | GPU | Key Ops |
|---|---|---|
| SM80 (Ampere) | A100 | HMMA, ldmatrix, cp.async |
| SM90 (Hopper) | H100/H200 | WGMMA, TMA, mbarrier, cluster |
| SM100 (Blackwell) | B200 | 5th-gen TC, bulk copy |
| CDNA3 (gfx942) | MI300X | MFMA |

## Quick Start

```python
import gluon
from gluon import DType, Layout, SwizzleMode, TensorRef, WarpRole, MemorySpace, SM90
from gluon.dsl import kernel, smem, mbarrier, warp_role, tma_load, wgmma, for_range

@gluon.kernel(arch=SM90, num_warps=8)
def my_gemm(A, B, C):
    # Explicit swizzled shared memory
    a_smem = smem(DType.f16, Layout.swizzled(128, 64, swizzle=SwizzleMode.B128),
                  name="A_tile", double_buffer=True)
    b_smem = smem(DType.f16, Layout.swizzled(64, 256, swizzle=SwizzleMode.B128),
                  name="B_tile", double_buffer=True)

    mbar = mbarrier(count=1, name="sync")

    # Producer warps: TMA data movement
    with warp_role(WarpRole.PRODUCER, warp_range=(0, 3)):
        with for_range("k", 0, "K", 64):
            tma_load(a_smem, "A", barrier_name=mbar)
            tma_load(b_smem, "B", barrier_name=mbar)

    # Consumer warps: WGMMA compute
    with warp_role(WarpRole.CONSUMER, warp_range=(4, 7)):
        with for_range("k", 0, "K", 64):
            wgmma(a_smem, b_smem, "acc")

# Trace → compile → inspect
a = TensorRef("A", DType.f16, Layout.row_major(128, 64), MemorySpace.GLOBAL)
b = TensorRef("B", DType.f16, Layout.col_major(64, 256), MemorySpace.GLOBAL)
c = TensorRef("C", DType.f32, Layout.row_major(128, 256), MemorySpace.GLOBAL)

module = my_gemm.trace(a, b, c)
print(module.pretty())
```

## Examples

- **[GEMM Kernel](examples/gemm_kernel.py)** — Warp-specialized GEMM with TMA + WGMMA, double-buffered smem
- **[FlashAttention](examples/flash_attn_kernel.py)** — Fused attention with online softmax, TMA streaming

```bash
python examples/gemm_kernel.py
python examples/flash_attn_kernel.py
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```
gluon-dsl/
├── gluon/
│   ├── __init__.py          # Public API
│   ├── types.py             # DType, Layout, SharedMemoryDescriptor, TensorRef
│   ├── arch.py              # SM80, SM90, SM100, CDNA3
│   ├── ir.py                # Region-based IR (20+ ops)
│   ├── dsl.py               # @gluon.kernel + DSL builtins
│   ├── frontend.py          # AST inspector + IR validator
│   ├── jit.py               # JIT compilation driver
│   ├── runtime.py           # Kernel launcher
│   ├── passes/
│   │   ├── layout_check.py  # Layout + bank conflict validation
│   │   ├── warp_spec.py     # Warp role + barrier pairing
│   │   ├── smem_alloc.py    # Offset assignment + capacity check
│   │   └── pipeline.py      # Software pipelining transform
│   └── backend/
│       ├── ptx_emit.py      # Inline PTX templates
│       └── triton_lowering.py  # Gluon IR → TTIR
├── examples/
│   ├── gemm_kernel.py
│   └── flash_attn_kernel.py
├── tests/
│   ├── test_types.py
│   ├── test_ir.py
│   ├── test_passes.py
│   └── test_frontend.py
└── pyproject.toml
```

## Design Philosophy

Gluon is not Triton. It is explicitly **not portable**:

- Kernels are **bound to an arch** at declaration time
- Using `wgmma()` on SM80 raises `RuntimeError` at trace time
- Using `mfma()` on any NVIDIA arch raises `RuntimeError`
- Shared memory layouts are **never implicit** — you choose the swizzle
- Warp roles are **never auto-assigned** — you pick producer vs consumer

This is by design. Gluon is the vehicle for **peak-performance kernels** when Triton's auto-tuning isn't enough.

## License

MIT
