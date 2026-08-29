"""
Tests for Gluon type system — Layout, SharedMemoryDescriptor, DType, TensorRef.
"""
import pytest
from gluon.types import (
    DType, Layout, MemorySpace, SharedMemoryDescriptor,
    SwizzleMode, TensorRef, WarpRole,
)


class TestDType:
    def test_sizes(self):
        assert DType.f16.nbytes == 2
        assert DType.f32.nbytes == 4
        assert DType.f64.nbytes == 8
        assert DType.i8.nbytes == 1
        assert DType.i32.nbytes == 4

    def test_bits(self):
        assert DType.f16.bits == 16
        assert DType.f32.bits == 32

    def test_repr(self):
        assert "f16" in repr(DType.f16)
        assert "i32" in repr(DType.i32)


class TestLayout:
    def test_row_major_2d(self):
        layout = Layout.row_major(4, 8)
        assert layout.shape == (4, 8)
        assert layout.strides == (8, 1)
        assert layout.numel == 32
        assert layout.is_contiguous

    def test_col_major_2d(self):
        layout = Layout.col_major(4, 8)
        assert layout.shape == (4, 8)
        assert layout.strides == (1, 4)
        assert layout.numel == 32
        assert not layout.is_contiguous

    def test_row_major_3d(self):
        layout = Layout.row_major(2, 3, 4)
        assert layout.shape == (2, 3, 4)
        assert layout.strides == (12, 4, 1)
        assert layout.numel == 24

    def test_swizzled(self):
        layout = Layout.swizzled(128, 64, swizzle=SwizzleMode.B128)
        assert layout.swizzle == SwizzleMode.B128
        assert layout.numel == 128 * 64

    def test_custom(self):
        layout = Layout.custom(shape=(8, 16), strides=(32, 1))
        assert layout.strides == (32, 1)
        assert not layout.is_contiguous

    def test_custom_rank_mismatch(self):
        with pytest.raises(ValueError):
            Layout.custom(shape=(8,), strides=(1, 2))

    def test_byte_size(self):
        layout = Layout.row_major(128, 64)
        assert layout.byte_size(DType.f16) == 128 * 64 * 2
        assert layout.byte_size(DType.f32) == 128 * 64 * 4

    def test_linear_index(self):
        layout = Layout.row_major(4, 8)
        assert layout.linear_index(0, 0) == 0
        assert layout.linear_index(0, 1) == 1
        assert layout.linear_index(1, 0) == 8
        assert layout.linear_index(3, 7) == 31

    def test_linear_index_wrong_rank(self):
        layout = Layout.row_major(4, 8)
        with pytest.raises(ValueError):
            layout.linear_index(0, 0, 0)

    def test_rank(self):
        assert Layout.row_major(4, 8).rank == 2
        assert Layout.row_major(2, 3, 4).rank == 3


class TestSharedMemoryDescriptor:
    def test_basic(self):
        layout = Layout.row_major(128, 64)
        desc = SharedMemoryDescriptor("tile_a", DType.f16, layout)
        assert desc.single_buffer_bytes >= 128 * 64 * 2
        assert desc.total_bytes == desc.single_buffer_bytes

    def test_double_buffer(self):
        layout = Layout.row_major(128, 64)
        desc = SharedMemoryDescriptor("tile_a", DType.f16, layout,
                                       double_buffer=True)
        assert desc.total_bytes == desc.single_buffer_bytes * 2

    def test_bank_conflict_detection(self):
        # 128 columns of f32 = 512 bytes per row = 32*16, multiple of 128
        layout = Layout.row_major(64, 128)
        desc = SharedMemoryDescriptor("bad_layout", DType.f32, layout)
        warnings = desc.check_bank_conflicts()
        assert len(warnings) > 0

    def test_no_bank_conflict_with_swizzle(self):
        layout = Layout.swizzled(64, 128, swizzle=SwizzleMode.B128)
        desc = SharedMemoryDescriptor("good_layout", DType.f32, layout)
        warnings = desc.check_bank_conflicts()
        assert len(warnings) == 0


class TestTensorRef:
    def test_basic(self):
        ref = TensorRef("A", DType.f16, Layout.row_major(128, 64))
        assert ref.shape == (128, 64)
        assert ref.byte_size == 128 * 64 * 2
        assert ref.space == MemorySpace.GLOBAL

    def test_to_space(self):
        ref = TensorRef("A", DType.f16, Layout.row_major(128, 64))
        shared_ref = ref.to_space(MemorySpace.SHARED)
        assert shared_ref.space == MemorySpace.SHARED
        assert shared_ref.dtype == ref.dtype


class TestWarpRole:
    def test_values(self):
        assert WarpRole.PRODUCER.value == "producer"
        assert WarpRole.CONSUMER.value == "consumer"
        assert WarpRole.ANY.value == "any"
