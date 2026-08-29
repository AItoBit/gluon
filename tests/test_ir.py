"""
Tests for Gluon IR — node construction, pretty-printing, module structure.
"""
import pytest
from gluon.ir import (
    GluonModule, KernelFunc, Region, IRNode,
    SmemAlloc, SmemLoad, SmemStore, GlobalLoad, GlobalStore,
    TMALoad, WGMMA, HMMA, MFMA, FMA, ElementwiseOp,
    Barrier, MBarrierInit, MBarrierArrive, MBarrierWait, FenceAsyncShared,
    ForLoop, IfElse, WarpSpecialize,
)
from gluon.types import (
    DType, Layout, MemorySpace, SharedMemoryDescriptor,
    SwizzleMode, TensorRef, WarpRole,
)
from gluon.arch import SM90, SM80


class TestIRNodes:
    def test_smem_alloc(self):
        desc = SharedMemoryDescriptor("tile", DType.f16, Layout.row_major(64, 64))
        node = SmemAlloc(descriptor=desc)
        assert "tile" in node.pretty()
        assert node.node_id > 0

    def test_tma_load(self):
        node = TMALoad(dst_smem="a_smem", src_global="A",
                       coords=("i", "j"), barrier="mbar")
        text = node.pretty()
        assert "tma.load" in text
        assert "a_smem" in text

    def test_wgmma(self):
        from gluon.arch import SM90
        tc = SM90.tc_shapes[0]
        node = WGMMA(a_smem="a", b_smem="b", c_reg="acc", shape=tc)
        text = node.pretty()
        assert "wgmma" in text
        assert "m64" in text

    def test_barrier(self):
        node = Barrier()
        assert "barrier" in node.pretty()

    def test_mbarrier_init(self):
        node = MBarrierInit(name="mbar_0", count=4)
        text = node.pretty()
        assert "mbar_0" in text
        assert "count=4" in text

    def test_fence(self):
        node = FenceAsyncShared()
        assert "fence" in node.pretty()

    def test_for_loop(self):
        body = [Barrier(), FenceAsyncShared()]
        loop = ForLoop(var="i", start=0, end=16, step=1, body=body)
        text = loop.pretty()
        assert "for i" in text
        assert "barrier" in text

    def test_if_else(self):
        node = IfElse(condition="is_producer",
                      then_body=[Barrier()],
                      else_body=[FenceAsyncShared()])
        text = node.pretty()
        assert "if" in text
        assert "else" in text

    def test_elementwise(self):
        node = ElementwiseOp(dst="r1", op="silu", operands=("x",), dtype=DType.f32)
        text = node.pretty()
        assert "silu" in text

    def test_fma(self):
        node = FMA(dst="d", a="a", b="b", c="c", dtype=DType.f32)
        assert "fma" in node.pretty()


class TestRegion:
    def test_basic(self):
        region = Region(role=WarpRole.PRODUCER, warp_range=(0, 3))
        region.append(Barrier())
        region.append(FenceAsyncShared())
        text = region.pretty()
        assert "PRODUCER" in text
        assert "warps=" in text
        assert len(region.body) == 2


class TestKernelFunc:
    def test_basic(self):
        params = [
            TensorRef("A", DType.f16, Layout.row_major(128, 64)),
            TensorRef("B", DType.f16, Layout.col_major(64, 128)),
        ]
        kf = KernelFunc(name="test_gemm", arch=SM90, params=params,
                        num_warps=8)
        assert kf.num_threads == 256
        text = kf.pretty()
        assert "test_gemm" in text
        assert "Hopper" in text

    def test_smem_tracking(self):
        desc = SharedMemoryDescriptor("tile", DType.f16,
                                       Layout.row_major(128, 64))
        kf = KernelFunc(name="test", arch=SM90, num_warps=8,
                        smem_descriptors=[desc])
        assert kf.total_smem_bytes > 0


class TestGluonModule:
    def test_basic(self):
        kf = KernelFunc(name="my_kernel", arch=SM90, num_warps=4)
        module = GluonModule(name="test_module")
        module.add_kernel(kf)
        assert len(module.kernels) == 1
        text = module.pretty()
        assert "test_module" in text
        assert "my_kernel" in text

    def test_repr(self):
        module = GluonModule(name="m")
        assert "module" in repr(module)
