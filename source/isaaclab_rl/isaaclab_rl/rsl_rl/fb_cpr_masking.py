# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Backward masking for FB-CPR (BFM-0.7).

The backward map ``B`` encodes a state into ``z`` from a concatenation of obs
keys. Backward masking partitions that flat input into named body-part groups
and lets ``B`` see only a random subset of them: ``B(m * s, m)`` where ``m`` is
the per-group active flag (1 = visible, 0 = zeroed) and is also appended to the
input. The mask is a VIEW SELECTOR for the encoder only: once a ``z`` is
produced, no mask travels with it (F, actor, critics, discriminator and the
replay never see it).

Groups (G1 29-DoF, BFM keypoint layout): left_arm, right_arm, torso, left_leg,
right_leg, pelvis, contacts. See :func:`build_backward_mask_groups`.
"""

from __future__ import annotations

import torch

MASK_GROUP_NAMES: tuple[str, ...] = (
    "left_arm", "right_arm", "torso", "left_leg", "right_leg", "pelvis", "contacts",
)

# --- G1 29-DoF canonical joint order -> group ------------------------------
# legs 0-5 / 6-11, waist 12-14, arms 15-21 / 22-28.
_JOINT_GROUP: tuple[str, ...] = (
    *(["left_leg"] * 6), *(["right_leg"] * 6), *(["torso"] * 3),
    *(["left_arm"] * 7), *(["right_arm"] * 7),
)
# --- BFM 31-keypoint order -> group -----------------------------------------
# 0 pelvis; 1-6 left leg; 7-12 right leg; 13 waist_yaw, 14 waist_roll, 15 torso;
# 16-22 left arm; 23-29 right arm; 30 head (virtual, rigid on torso).
_KEYPOINT_GROUP: tuple[str, ...] = (
    "pelvis", *(["left_leg"] * 6), *(["right_leg"] * 6), "torso", "torso", "torso",
    *(["left_arm"] * 7), *(["right_arm"] * 7), "torso",
)
_NUM_JOINTS = 29
_NUM_KEYPOINTS = 31
# BFM ``state`` block: [dof_pos_dev(29) | dof_vel(29) | gravity(3) | root_ang_vel(3)]
_STATE_DIM = 2 * _NUM_JOINTS + 6
# BFM ``max_local_self``: root_h(1) | pos (K-1)*3 | rot6d K*6 | lin_vel K*3 | ang_vel K*3
_PRIV_DIM = 1 + (_NUM_KEYPOINTS - 1) * 3 + _NUM_KEYPOINTS * (6 + 3 + 3)
# BFM-0.7 ``contact_labels``: L hand, R hand, L foot, R foot.
_CONTACT_DIM = 4


def _state_groups(offset: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {g: [] for g in MASK_GROUP_NAMES}
    for j in range(_NUM_JOINTS):
        out[_JOINT_GROUP[j]].append(offset + j)                    # dof_pos_dev
        out[_JOINT_GROUP[j]].append(offset + _NUM_JOINTS + j)      # dof_vel
    out["pelvis"] += [offset + 2 * _NUM_JOINTS + i for i in range(6)]  # gravity + base ang vel
    return out


def _priv_groups(offset: int) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {g: [] for g in MASK_GROUP_NAMES}
    K = _NUM_KEYPOINTS
    out["pelvis"].append(offset)                                   # root_h
    pos0, rot0, lin0, ang0 = 1, 1 + (K - 1) * 3, 1 + (K - 1) * 3 + K * 6, 1 + (K - 1) * 3 + K * 9
    for k in range(1, K):
        out[_KEYPOINT_GROUP[k]] += [offset + pos0 + (k - 1) * 3 + i for i in range(3)]
    for k in range(K):
        g = _KEYPOINT_GROUP[k]
        out[g] += [offset + rot0 + k * 6 + i for i in range(6)]
        out[g] += [offset + lin0 + k * 3 + i for i in range(3)]
        out[g] += [offset + ang0 + k * 3 + i for i in range(3)]
    return out


def build_backward_mask_groups(
    input_keys: tuple[str, ...] | list[str],
    key_dims: dict[str, int],
) -> tuple[tuple[str, ...], list[list[int]]]:
    """Map B's flat input (keys concatenated in ``input_keys`` order) to groups.

    Supported keys: ``state`` (64), ``privileged_state`` (463), ``contact_labels``
    (4). Any other key, or a key with an unexpected width, raises. Returns
    ``(group_names, groups)`` where ``groups[i]`` lists the flat indices of
    group ``i``; the groups partition ``range(sum(dims))``.
    """
    groups: dict[str, list[int]] = {g: [] for g in MASK_GROUP_NAMES}
    offset = 0
    for key in input_keys:
        dim = int(key_dims[key])
        if key == "state":
            if dim != _STATE_DIM:
                raise ValueError(f"backward masking expects state dim {_STATE_DIM}, got {dim}")
            part = _state_groups(offset)
        elif key == "privileged_state":
            if dim != _PRIV_DIM:
                raise ValueError(f"backward masking expects privileged_state dim {_PRIV_DIM}, got {dim}")
            part = _priv_groups(offset)
        elif key == "contact_labels":
            if dim != _CONTACT_DIM:
                raise ValueError(f"backward masking expects contact_labels dim {_CONTACT_DIM}, got {dim}")
            part = {g: [] for g in MASK_GROUP_NAMES}
            part["contacts"] = [offset + i for i in range(dim)]
        else:
            raise ValueError(f"backward masking has no group layout for obs key {key!r}")
        for g, ix in part.items():
            groups[g] += ix
        offset += dim
    flat = sorted(i for ix in groups.values() for i in ix)
    if flat != list(range(offset)):
        raise AssertionError("backward mask groups do not partition the input")
    return MASK_GROUP_NAMES, [groups[g] for g in MASK_GROUP_NAMES]


def group_expand_matrix(groups: list[list[int]], in_dim: int) -> torch.Tensor:
    """``[G, in_dim]`` 0/1 matrix so that ``mask @ M`` is the per-feature mask."""
    M = torch.zeros(len(groups), in_dim)
    for g, ix in enumerate(groups):
        M[g, ix] = 1.0
    return M


def sample_group_mask(
    num: int,
    num_groups: int,
    mask_prob: float,
    fallback_group: int,
    forced_off_groups: tuple[int, ...] = (),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Sample ``[num, G]`` active flags (1 = visible).

    Each group is masked independently with ``mask_prob``; ``forced_off_groups``
    are always masked (e.g. contacts for expert states that carry no labels).
    Rows with every group masked get ``fallback_group`` re-enabled so B always
    sees something.
    """
    if not 0.0 <= mask_prob < 1.0:
        raise ValueError(f"mask_prob must be in [0, 1), got {mask_prob}")
    if not 0 <= fallback_group < num_groups:
        raise ValueError("fallback_group out of range")
    active = torch.rand(num, num_groups, device=device) >= mask_prob
    for g in forced_off_groups:
        active[:, g] = False
    none_active = ~active.any(dim=1)
    active[none_active, fallback_group] = True
    return active.to(torch.float32)


def full_mask(
    num: int,
    num_groups: int,
    forced_off_groups: tuple[int, ...] = (),
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """All groups visible except ``forced_off_groups`` (the canonical view)."""
    m = torch.ones(num, num_groups, device=device)
    for g in forced_off_groups:
        m[:, g] = 0.0
    return m
