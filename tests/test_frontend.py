"""
Tests for Gluon DSL frontend — tracing @gluon.kernel functions into IR.
"""
import pytest
from gluon import (
    DType, Layout, SwizzleMode, TensorRef, WarpRole, MemorySpace,
    SM90, SM80, CDNA3,
)
from gluon.dsl import (
    kernel, smem, barrier, mbarrier, mbarrier_arrive, mbarrier_wait,
    fence_async_shared, warp_role, tma_load, wgmma, hmma, mfma,
    for_range, global_store, elementwise, program_id,
)
from gluon.ir import (
    GluonModule, SmemAlloc, TMALoad, WGMMA, HMMA,
    Barrier, MBarrierInit, WarpSpecialize, ForLoop, GlobalStore,
)
from gluon.frontend import validate_module, KernelInspector


class TestKernelTracing:
    def test_simple_kernel(self):
        @kernel(arch=SM90, num_warps=4)
        def simple(A: TensorRef):
            s = smem(DType.f16, Layout.row_major(64, 64), name="tile")
            barrier()

        ref = TensorRef("A", DType.f16, Layout.row_major(64, 64))
        module = simple.trace(ref)

        assert isinstance(module, GluonModule)
        assert len(module.kernels) == 1
        kf = module.kernels[0]
        assert kf.name == "simple"
        assert kf.arch == SM90
        assert any(isinstance(n, SmemAlloc) for n in kf.body)
        assert any(isinstance(n, Barrier) for n in kf.body)

    def test_warp_specialized_kernel(self):
        @kernel(arch=SM90, num_warps=8)
        def ws_kernel(A: TensorRef, B: TensorRef):
            a_smem = smem(DType.f16,
                          Layout.swizzled(128, 64, swizzle=SwizzleMode.B128),
                          name="A_smem")
            mbar = mbarrier(count=1, name="sync")

            with warp_role(WarpRole.PRODUCER, warp_range=(0, 3)):
                tma_load(a_smem, "A", coords=("i",), barrier_name=mbar)
                mbarrier_arrive(mbar)

            with warp_role(WarpRole.CONSUMER, warp_range=(4, 7)):
                mbarrier_wait(mbar)
                wgmma(a_smem, "B_smem", "acc")

        a = TensorRef("A", DType.f16, Layout.row_major(128, 64))
        b = TensorRef("B", DType.f16, Layout.col_major(64, 128))
        module = ws_kernel.trace(a, b)

        kf = module.kernels[0]
        ws_nodes = [n for n in kf.body if isinstance(n, WarpSpecialize)]
        assert len(ws_nodes) >= 1

    def test_for_loop(self):
        @kernel(arch=SM90, num_warps=4)
        def loop_kernel(A: TensorRef):
            with for_range("i", 0, 16, 1):
                barrier()

        ref = TensorRef("A", DType.f16, Layout.row_major(16, 16))
        module = loop_kernel.trace(ref)
        kf = module.kernels[0]
        loops = [n for n in kf.body if isinstance(n, ForLoop)]
        assert len(loops) == 1
        assert loops[0].var == "i"


class TestArchGating:
    def test_tma_rejected_on_ampere(self):
        @kernel(arch=SM80, num_warps=4)
        def bad_kernel(A: TensorRef):
            s = smem(DType.f16, Layout.row_major(64, 64), name="tile")
            tma_load(s, "A", coords=("i",))

        ref = TensorRef("A", DType.f16, Layout.row_major(64, 64))
        with pytest.raises(RuntimeError, match="TMA not supported"):
            bad_kernel.trace(ref)

    def test_wgmma_rejected_on_ampere(self):
        @kernel(arch=SM80, num_warps=4)
        def bad_kernel(A: TensorRef):
            wgmma("a", "b", "c")

        ref = TensorRef("A", DType.f16, Layout.row_major(64, 64))
        with pytest.raises(RuntimeError, match="WGMMA not supported"):
            bad_kernel.trace(ref)

    def test_mfma_rejected_on_nvidia(self):
        @kernel(arch=SM90, num_warps=4)
        def bad_kernel(A: TensorRef):
            mfma("a", "b", "c")

        ref = TensorRef("A", DType.f16, Layout.row_major(64, 64))
        with pytest.raises(RuntimeError, match="MFMA not supported"):
            bad_kernel.trace(ref)

    def test_mbarrier_rejected_on_ampere(self):
        @kernel(arch=SM80, num_warps=4)
        def bad_kernel(A: TensorRef):
            mbarrier(count=1)

        ref = TensorRef("A", DType.f16, Layout.row_major(64, 64))
        with pytest.raises(RuntimeError, match="mbarrier not supported"):
            bad_kernel.trace(ref)

    def test_hmma_on_ampere(self):
        """HMMA should work on Ampere."""
        @kernel(arch=SM80, num_warps=4)
        def hmma_kernel(A: TensorRef):
            hmma("a", "b", "c")

        ref = TensorRef("A", DType.f16, Layout.row_major(64, 64))
        module = hmma_kernel.trace(ref)
        assert len(module.kernels) == 1


class TestModuleValidation:
    def test_thread_count_exceeded(self):
        kf = __import__("gluon.ir", fromlist=["KernelFunc"]).KernelFunc
        from gluon.arch import SM90
        k = kf(name="big", arch=SM90, num_warps=64)  # 64*32=2048 > 1024
        m = GluonModule(name="test")
        m.add_kernel(k)
        errors = validate_module(m)
        assert any("exceeds" in e for e in errors)


class TestKernelInspector:
    def test_valid_function(self):
        @kernel(arch=SM90, num_warps=4)
        def valid(A: TensorRef):
            barrier()

        inspector = KernelInspector(valid._fn, SM90)
        assert inspector.validate_arch_ops()

    def test_invalid_tma_on_ampere(self):
        def bad_func(A):
            tma_load("s", "A")

        inspector = KernelInspector(bad_func, SM80)
        result = inspector.validate_arch_ops()
        assert not result
        assert len(inspector.errors) > 0
