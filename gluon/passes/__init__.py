"""Gluon compiler passes package."""
from .layout_check import LayoutCheckPass
from .warp_spec import WarpSpecPass
from .smem_alloc import SmemAllocPass
from .pipeline import PipelinePass

__all__ = ["LayoutCheckPass", "WarpSpecPass", "SmemAllocPass", "PipelinePass"]


def run_all_passes(module, arch=None):
    """Run the standard Gluon pass pipeline on a module."""
    errors = []
    errors.extend(LayoutCheckPass().run(module))
    errors.extend(WarpSpecPass().run(module))
    errors.extend(SmemAllocPass().run(module))
    # Pipeline pass is a transform, not a validation
    PipelinePass().run(module)
    return errors
