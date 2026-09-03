# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared import stubs so the FB-CPR pure-torch tests can load
``isaaclab_rl.rsl_rl`` modules without Isaac Lab / gymnasium / rsl_rl.

All test modules must install the SAME stubs (pytest shares ``sys.modules``
across files), so they import from here instead of defining their own.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import types

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "isaaclab_rl"))


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        low = np.asarray(low, dtype=dtype)
        high = np.asarray(high, dtype=dtype)
        self.low = low
        self.high = high
        self.shape = tuple(shape) if shape is not None else low.shape
        self.dtype = dtype


class Dict:
    def __init__(self, spaces):
        self.spaces = dict(spaces)

    def __getitem__(self, key):
        return self.spaces[key]

    def keys(self):
        return self.spaces.keys()


def _stub(name: str, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


def install() -> None:
    if "isaaclab" not in sys.modules:
        _stub("isaaclab")
        _stub("isaaclab.utils", configclass=dataclasses.dataclass)
    if "gymnasium" not in sys.modules:
        spaces = _stub("gymnasium.spaces", Box=Box, Dict=Dict, Space=object)
        _stub("gymnasium", spaces=spaces)
    for name, path in (
        ("isaaclab_rl", _ROOT),
        ("isaaclab_rl.rsl_rl", f"{_ROOT}/rsl_rl"),
        ("isaaclab_rl.rsl_rl.modules", f"{_ROOT}/rsl_rl/modules"),
        ("isaaclab_rl.rsl_rl.algorithms", f"{_ROOT}/rsl_rl/algorithms"),
    ):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = [path]
            sys.modules[name] = m


def load(module: str):
    """Import ``isaaclab_rl.rsl_rl.<module>`` with the stubs installed."""
    install()
    return importlib.import_module(f"isaaclab_rl.rsl_rl.{module}")
