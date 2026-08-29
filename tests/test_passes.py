"""
Tests for Gluon compiler passes.
"""
import pytest
from gluon.ir import (
    GluonModule, KernelFunc, Region, SmemAlloc, WGMMA, TMALoad,
    ForLoop, WarpSpecialize, MBarrierInit, MBarrierArrive, MBarrierWait,
    FenceAsyncShared,
)
from gluon.types import (
    DType, Layout, SharedMemoryDescriptor, SwizzleMode, TensorRef, WarpRole,
)
from gluon.arch import SM90, SM80
from gluon.passes.layout_check import LayoutCheckPass
from gluon.passes.warp_spec import WarpSpecPass
from gluon.passes.smem_alloc import SmemAllocPass
from gluon.passes.pipeline import PipelinePass


def _make_module(kernel: KernelFunc) -> GluonModule:
    m = GluonModule(name="test")
    m.add_kernel(kernel)
    return m


class TestLayoutCheckPass:
    def test_valid_layout(self):
        desc = SharedMemoryDescriptor("tile", DType.f16,
                                       Layout.swizzled(64, 64, swizzle=SwizzleMode.B128))
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        smem_descriptors=[desc],
                        body=[SmemAlloc(descriptor=desc)])
        errors = LayoutCheckPass().run(_make_module(kf))
        # Should have no errors for properly swizzled layout
        bank_errors = [e for e in errors if "bank conflict" in e.lower()]
        assert len(bank_errors) == 0

    def test_zero_element_layout(self):
        desc = SharedMemoryDescriptor("bad", DType.f16, Layout.row_major(0, 64))
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        body=[SmemAlloc(descriptor=desc)])
        errors = LayoutCheckPass().run(_make_module(kf))
        assert any("zero" in e.lower() for e in errors)

    def test_bad_alignment(self):
        desc = SharedMemoryDescriptor("bad", DType.f16,
                                       Layout.row_major(64, 64), alignment=3)
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        body=[SmemAlloc(descriptor=desc)])
        errors = LayoutCheckPass().run(_make_module(kf))
        assert any("power of 2" in e.lower() for e in errors)

    def test_wgmma_no_swizzle_warning(self):
        desc_a = SharedMemoryDescriptor("a", DType.f16,
                                         Layout.row_major(64, 64))
        desc_b = SharedMemoryDescriptor("b", DType.f16,
                                         Layout.col_major(64, 64))
        tc = SM90.tc_shapes[0]
        wgmma_node = WGMMA(a_smem="a", b_smem="b", c_reg="acc", shape=tc)
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        smem_descriptors=[desc_a, desc_b],
                        body=[SmemAlloc(descriptor=desc_a),
                              SmemAlloc(descriptor=desc_b),
                              wgmma_node])
        errors = LayoutCheckPass().run(_make_module(kf))
        assert any("swizzle" in e.lower() for e in errors)


class TestWarpSpecPass:
    def test_valid_warp_spec(self):
        prod = Region(role=WarpRole.PRODUCER, warp_range=(0, 3))
        cons = Region(role=WarpRole.CONSUMER, warp_range=(4, 7))
        ws = WarpSpecialize(regions={WarpRole.PRODUCER: prod,
                                      WarpRole.CONSUMER: cons})
        mbar_init = MBarrierInit(name="m", count=1)
        mbar_arr = MBarrierArrive(name="m")
        mbar_wait = MBarrierWait(name="m")
        prod.body.append(mbar_arr)
        cons.body.append(mbar_wait)

        kf = KernelFunc(name="k", arch=SM90, num_warps=8,
                        body=[mbar_init, ws])
        errors = WarpSpecPass().run(_make_module(kf))
        assert len(errors) == 0

    def test_overlapping_warp_ranges(self):
        prod = Region(role=WarpRole.PRODUCER, warp_range=(0, 5))
        cons = Region(role=WarpRole.CONSUMER, warp_range=(3, 7))
        ws = WarpSpecialize(regions={WarpRole.PRODUCER: prod,
                                      WarpRole.CONSUMER: cons})
        kf = KernelFunc(name="k", arch=SM90, num_warps=8, body=[ws])
        errors = WarpSpecPass().run(_make_module(kf))
        assert any("overlap" in e.lower() for e in errors)

    def test_warp_range_out_of_bounds(self):
        prod = Region(role=WarpRole.PRODUCER, warp_range=(0, 3))
        cons = Region(role=WarpRole.CONSUMER, warp_range=(4, 9))
        ws = WarpSpecialize(regions={WarpRole.PRODUCER: prod,
                                      WarpRole.CONSUMER: cons})
        kf = KernelFunc(name="k", arch=SM90, num_warps=8, body=[ws])
        errors = WarpSpecPass().run(_make_module(kf))
        assert any("exceeds" in e.lower() for e in errors)

    def test_unpaired_barrier(self):
        mbar_arr = MBarrierArrive(name="orphan")
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        body=[mbar_arr])
        errors = WarpSpecPass().run(_make_module(kf))
        assert any("without matching" in e.lower() for e in errors)

    def test_unused_barrier(self):
        mbar_init = MBarrierInit(name="unused", count=1)
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        body=[mbar_init])
        errors = WarpSpecPass().run(_make_module(kf))
        assert any("never arrived" in e.lower() for e in errors)


class TestSmemAllocPass:
    def test_basic_allocation(self):
        desc1 = SharedMemoryDescriptor("a", DType.f16,
                                        Layout.row_major(128, 64), alignment=128)
        desc2 = SharedMemoryDescriptor("b", DType.f16,
                                        Layout.row_major(64, 128), alignment=128)
        a1 = SmemAlloc(descriptor=desc1)
        a2 = SmemAlloc(descriptor=desc2)
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        body=[a1, a2])
        errors = SmemAllocPass().run(_make_module(kf))
        assert len(errors) == 0
        assert a1.offset == 0
        assert a2.offset >= desc1.total_bytes  # must be after first alloc

    def test_smem_overflow(self):
        # Create a huge smem alloc that exceeds SM90's 228KB
        layout = Layout.row_major(1024, 1024)
        desc = SharedMemoryDescriptor("huge", DType.f32, layout)
        assert desc.total_bytes > SM90.max_smem_per_block_bytes
        a = SmemAlloc(descriptor=desc)
        kf = KernelFunc(name="k", arch=SM90, num_warps=4, body=[a])
        errors = SmemAllocPass().run(_make_module(kf))
        assert any("exceeds" in e.lower() for e in errors)

    def test_alignment_padding(self):
        desc1 = SharedMemoryDescriptor("small", DType.i8,
                                        Layout.row_major(3), alignment=128)
        desc2 = SharedMemoryDescriptor("next", DType.f32,
                                        Layout.row_major(32), alignment=128)
        a1 = SmemAlloc(descriptor=desc1)
        a2 = SmemAlloc(descriptor=desc2)
        kf = KernelFunc(name="k", arch=SM90, num_warps=4,
                        body=[a1, a2])
        SmemAllocPass().run(_make_module(kf))
        # Second alloc must be aligned to 128
        assert a2.offset % 128 == 0


class TestPipelinePass:
    def test_pipeline_insertion(self):
        # Build a warp-specialized structure with producer/consumer loops
        tma = TMALoad(dst_smem="a", src_global="A", coords=("i",), barrier="m")
        prod_loop = ForLoop(var="k", start=0, end=16, step=1, body=[tma])
        prod = Region(role=WarpRole.PRODUCER, warp_range=(0, 3),
                      body=[prod_loop])

        tc = SM90.tc_shapes[0]
        wgmma = WGMMA(a_smem="a", b_smem="b", c_reg="acc", shape=tc)
        cons_loop = ForLoop(var="k", start=0, end=16, step=1, body=[wgmma])
        cons = Region(role=WarpRole.CONSUMER, warp_range=(4, 7),
                      body=[cons_loop])

        ws = WarpSpecialize(regions={WarpRole.PRODUCER: prod,
                                      WarpRole.CONSUMER: cons})
        kf = KernelFunc(name="k", arch=SM90, num_warps=8, body=[ws])
        module = _make_module(kf)

        info = PipelinePass(num_stages=2).run(module)
        # Should report pipelining was applied
        assert any("pipeline" in i.lower() for i in info)
        # Producer loop should now have a fence at the end
        assert any(isinstance(n, FenceAsyncShared)
                   for n in prod_loop.body)
