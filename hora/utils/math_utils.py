# --------------------------------------------------------
# In-Hand Object Rotation via Rapid Motor Adaptation
# https://arxiv.org/abs/2210.04887
# Copyright (c) 2022 Haozhi Qi
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
# Replaces the handful of helpers hora used from `isaacgym.torch_utils`.
# See docs/isaaclab_migration.md.
# --------------------------------------------------------

"""Small tensor helpers that IsaacLab does not provide.

Anything with an IsaacLab equivalent should be imported from ``isaaclab.utils.math``
rather than reimplemented here -- in particular ``quat_mul``, ``quat_conjugate``,
``quat_apply``, ``quat_from_angle_axis`` and ``axis_angle_from_quat``.

**Quaternion convention.** IsaacGym used xyzw (real part last); IsaacLab and USD use
**wxyz** (real part first). Everything in this codebase is wxyz now. The one place raw
xyzw survives is the published grasp caches in ``cache/*.npy``, which were written by the
old IsaacGym code -- :func:`quat_xyzw_to_wxyz` converts them on load.
"""

from __future__ import annotations

import torch


def to_torch(x, dtype=torch.float, device='cuda:0', requires_grad=False) -> torch.Tensor:
    """`isaacgym.torch_utils.to_torch` equivalent."""
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def unscale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map [lower, upper] onto [-1, 1]."""
    return (2.0 * x - upper - lower) / (upper - lower)


def scale(x: torch.Tensor, lower: torch.Tensor, upper: torch.Tensor) -> torch.Tensor:
    """Map [-1, 1] onto [lower, upper]."""
    return 0.5 * (x + 1.0) * (upper - lower) + lower


def tensor_clamp(t: torch.Tensor, min_t: torch.Tensor, max_t: torch.Tensor) -> torch.Tensor:
    """Elementwise clamp against broadcastable tensor bounds."""
    return torch.max(torch.min(t, max_t), min_t)


def torch_rand_float(lower: float, upper: float, shape: tuple[int, ...], device) -> torch.Tensor:
    """Uniform sample in [lower, upper) with the given shape."""
    return (upper - lower) * torch.rand(*shape, device=device) + lower


def quat_xyzw_to_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Convert IsaacGym-convention quaternions (real part last) to IsaacLab's (first).

    Only needed for data produced by the old IsaacGym code -- i.e. the grasp caches.
    """
    return torch.roll(q, shifts=1, dims=-1)


def quat_wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quat_xyzw_to_wxyz`."""
    return torch.roll(q, shifts=-1, dims=-1)
