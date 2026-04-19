# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Expert motion buffer for sparse-successor style discrimination.

Reads a precomputed dataset produced by
``scripts/precompute_expert_dataset.py`` in the Latent-Control repo, and
samples random contiguous snippets together with the corresponding keypoint
positions at the snippet's start frame. These are used by the
``SparseSuccessor`` algorithm to train the style discriminator against real
expert motions instead of self-referential rollouts.
"""

from __future__ import annotations

import os
import torch


class ExpertMotionBuffer:
    """Flat on-device buffer of per-frame expert features.

    All motions are concatenated along the time axis into a single
    ``[N_total, feature_dim]`` tensor. Motion boundaries are tracked so that
    snippets never cross clip boundaries.
    """

    def __init__(
        self,
        dataset_path: str,
        snippet_length: int,
        device: str = "cpu",
        keypoint_names: list[str] | None = None,
    ):
        """Load a precomputed expert dataset and pack it into contiguous buffers.

        Args:
            dataset_path: Path to the ``.pt`` file written by
                ``precompute_expert_dataset.py``.
            snippet_length: Length of the style snippet (in motion frames) used
                by the discriminator. Snippets are sampled as
                ``[start, start + snippet_length)``.
            device: Device to hold the flat buffer on. ``cpu`` is usually fine
                because sampled batches are tiny relative to total frames.
            keypoint_names: Optional override for the keypoint order used at
                training time. If provided, must match the dataset's
                ``keypoint_names``; used purely as a sanity check.
        """
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Expert motion dataset not found at {dataset_path}. "
                "Run scripts/precompute_expert_dataset.py first."
            )

        dataset = torch.load(dataset_path, map_location=device, weights_only=False)

        self.dataset_path = dataset_path
        self.snippet_length = snippet_length
        self.device = device
        self.keypoint_names: list[str] = dataset["keypoint_names"]
        self.joint_names: list[str] = dataset["joint_names"]
        self.style_state_dim: int = int(dataset["style_state_dim"])
        self.style_priv_dim: int = int(dataset["style_priv_dim"])
        self.num_keypoints = len(self.keypoint_names)

        if keypoint_names is not None:
            if list(keypoint_names) != list(self.keypoint_names):
                raise ValueError(
                    f"Keypoint set mismatch. Dataset has {self.keypoint_names} "
                    f"but algorithm config expects {keypoint_names}. "
                    "Re-run precompute_expert_dataset.py with the matching list."
                )

        motions = dataset["motions"]
        self.motion_names: list[str] = list(motions.keys())
        if len(self.motion_names) == 0:
            raise ValueError(f"Expert dataset at {dataset_path} has no motions.")

        # Concatenate all per-frame features along the time axis.
        state_chunks = []
        priv_chunks = []
        kp_chunks = []
        # Reset-state fields — present when the dataset was built with the
        # updated precompute script (root velocities added). Older datasets
        # that lack these fields make RSI impossible but still allow
        # discriminator training.
        joint_pos_chunks: list[torch.Tensor] = []
        joint_vel_chunks: list[torch.Tensor] = []
        root_pos_chunks: list[torch.Tensor] = []
        root_quat_chunks: list[torch.Tensor] = []
        root_lin_vel_chunks: list[torch.Tensor] = []
        root_ang_vel_chunks: list[torch.Tensor] = []

        motion_starts: list[int] = []
        motion_ends: list[int] = []  # exclusive
        cursor = 0
        reset_fields_complete = True
        for name in self.motion_names:
            m = motions[name]
            T = int(m["num_frames"])
            state_chunks.append(m["style_state"].float())
            priv_chunks.append(m["style_priv"].float())
            kp_chunks.append(m["keypoint_pos"].float())
            # All the RSI fields must be present on every motion for the
            # buffer to advertise support. If any motion is missing a field
            # (legacy datasets), disable RSI globally.
            for key, store in (
                ("joint_pos", joint_pos_chunks),
                ("joint_vel", joint_vel_chunks),
                ("root_pos", root_pos_chunks),
                ("root_quat", root_quat_chunks),
                ("root_lin_vel", root_lin_vel_chunks),
                ("root_ang_vel", root_ang_vel_chunks),
            ):
                if key in m:
                    store.append(m[key].float())
                else:
                    reset_fields_complete = False
            motion_starts.append(cursor)
            motion_ends.append(cursor + T)
            cursor += T

        self.state_buffer = torch.cat(state_chunks, dim=0).to(device).contiguous()        # [N, 64]
        self.priv_buffer = torch.cat(priv_chunks, dim=0).to(device).contiguous()          # [N, priv_dim]
        self.kp_buffer = torch.cat(kp_chunks, dim=0).to(device).contiguous()              # [N, K, 3]
        self.total_frames = self.state_buffer.shape[0]

        # Optional reset-state buffers
        self.supports_reset_states = reset_fields_complete and len(joint_pos_chunks) == len(self.motion_names)
        if self.supports_reset_states:
            self.joint_pos_buffer = torch.cat(joint_pos_chunks, dim=0).to(device).contiguous()      # [N, J]
            self.joint_vel_buffer = torch.cat(joint_vel_chunks, dim=0).to(device).contiguous()      # [N, J]
            self.root_pos_buffer = torch.cat(root_pos_chunks, dim=0).to(device).contiguous()        # [N, 3]
            self.root_quat_buffer = torch.cat(root_quat_chunks, dim=0).to(device).contiguous()      # [N, 4] wxyz
            self.root_lin_vel_buffer = torch.cat(root_lin_vel_chunks, dim=0).to(device).contiguous()  # [N, 3]
            self.root_ang_vel_buffer = torch.cat(root_ang_vel_chunks, dim=0).to(device).contiguous()  # [N, 3]
            self.num_joints = self.joint_pos_buffer.shape[1]
        else:
            self.joint_pos_buffer = None
            self.joint_vel_buffer = None
            self.root_pos_buffer = None
            self.root_quat_buffer = None
            self.root_lin_vel_buffer = None
            self.root_ang_vel_buffer = None
            self.num_joints = 0

        # Per-frame style feature = state | priv
        self.style_feature_dim = self.style_state_dim + self.style_priv_dim

        # Valid start indices: frames where [start, start + snippet_length)
        # lies fully within one motion. Build a flat index tensor for O(1) sampling.
        valid_starts: list[int] = []
        for s, e in zip(motion_starts, motion_ends):
            last_valid = e - snippet_length
            if last_valid > s:
                valid_starts.extend(range(s, last_valid + 1))
        if len(valid_starts) == 0:
            raise ValueError(
                f"No motion is long enough for snippet_length={snippet_length}. "
                "Either reduce snippet_length or use a different dataset."
            )
        self.valid_starts = torch.as_tensor(valid_starts, dtype=torch.long, device=device)
        self.num_valid_starts = self.valid_starts.shape[0]

        rsi_msg = f"rsi={'on' if self.supports_reset_states else 'off'}"
        print(
            f"[ExpertMotionBuffer] loaded {len(self.motion_names)} motions, "
            f"{self.total_frames} frames, {self.num_valid_starts} valid starts, "
            f"style_dim={self.style_feature_dim}, snippet_dim={self.snippet_length * self.style_feature_dim}, "
            f"{rsi_msg}"
        )

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _stack_snippet(self, start_idx: torch.Tensor) -> torch.Tensor:
        """Gather ``snippet_length`` consecutive frames starting at each index.

        Args:
            start_idx: [B] indices into ``state_buffer``.
        Returns:
            [B, snippet_length * style_feature_dim]
        """
        L = self.snippet_length
        B = start_idx.shape[0]
        # Build absolute frame indices via broadcasting — [B, L]
        offsets = torch.arange(L, device=self.device).unsqueeze(0)
        indices = start_idx.unsqueeze(1) + offsets                # [B, L]
        flat = indices.reshape(-1)
        state = self.state_buffer[flat].reshape(B, L, -1)
        priv = self.priv_buffer[flat].reshape(B, L, -1)
        frame = torch.cat([state, priv], dim=-1)                  # [B, L, style_dim]
        return frame.reshape(B, -1).contiguous()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample a batch of expert snippets with start-frame keypoint positions.

        Returns:
            dict with keys:
                snippet:      [B, snippet_length * style_feature_dim]
                keypoint_pos: [B, num_keypoints, 3]  (world-frame, at start frame)
        """
        idx = torch.randint(0, self.num_valid_starts, (batch_size,), device=self.device)
        start_frames = self.valid_starts[idx]  # [B]
        snippet = self._stack_snippet(start_frames)
        keypoint_pos = self.kp_buffer[start_frames]   # [B, K, 3]
        return {"snippet": snippet, "keypoint_pos": keypoint_pos}

    # ------------------------------------------------------------------
    # Future-grounded sampling (per-atom sparse constraint construction)
    # ------------------------------------------------------------------

    def _future_valid_starts(self, horizon: int) -> torch.Tensor:
        """Valid start frames s.t. both snippet (L frames) *and* the future
        window (H+1 frames from the anchor) stay inside one motion.

        We cache the result per-``horizon`` so repeated sampler calls are O(1).
        """
        if not hasattr(self, "_future_valid_cache"):
            self._future_valid_cache: dict[int, torch.Tensor] = {}
        if horizon in self._future_valid_cache:
            return self._future_valid_cache[horizon]

        # ``valid_starts`` was computed in __init__ to guarantee the snippet
        # window fits inside one motion. For future-grounded sampling we
        # also need ``start + horizon < motion_end``. We recover motion
        # boundaries by splitting ``valid_starts`` at the jumps (within a
        # motion the starts are contiguous; a gap of >1 signals the next
        # motion).
        vs = self.valid_starts.tolist()
        if len(vs) == 0:
            self._future_valid_cache[horizon] = torch.as_tensor(
                [], dtype=torch.long, device=self.device,
            )
            return self._future_valid_cache[horizon]

        runs: list[tuple[int, int]] = []
        run_start = vs[0]
        run_prev = vs[0]
        for s in vs[1:]:
            if s == run_prev + 1:
                run_prev = s
                continue
            runs.append((run_start, run_prev))
            run_start = s
            run_prev = s
        runs.append((run_start, run_prev))

        # Each run's last valid start implies ``motion_end = last + snippet_length``.
        # For the future-window constraint we need ``start + horizon < motion_end``.
        future_starts: list[int] = []
        for (lo, hi) in runs:
            motion_end = hi + self.snippet_length
            last_future_valid = motion_end - (horizon + 1)
            if last_future_valid >= lo:
                future_starts.extend(range(lo, min(hi, last_future_valid) + 1))

        tensor = torch.as_tensor(future_starts, dtype=torch.long, device=self.device)
        self._future_valid_cache[horizon] = tensor
        return tensor

    def sample_with_future_window(
        self, batch_size: int, horizon: int,
    ) -> dict[str, torch.Tensor]:
        """Sample a batch of snippets plus a per-sample future keypoint window.

        Used by the per-atom future-grounded constraint constructor: each
        atomic query samples its own ``τ_i ∈ [1, horizon]`` and pulls its
        target from ``kp_window[b, τ_i, k_i]``. The whole ``z_C`` is then
        a pooled encoding of a multi-time sparse constraint set.

        Args:
            batch_size: B
            horizon: H — maximum τ (inclusive) the caller will sample.

        Returns:
            snippet:     [B, snippet_length * style_feature_dim]
            kp_window:   [B, H+1, K, 3]  — index 0 is the anchor frame,
                         index h>=1 is the keypoint pos at ``start + h``.
        """
        valid = self._future_valid_starts(horizon)
        if valid.numel() == 0:
            raise ValueError(
                f"No expert motion is long enough for horizon={horizon} + "
                f"snippet_length={self.snippet_length}."
            )
        idx = torch.randint(0, valid.shape[0], (batch_size,), device=self.device)
        start_frames = valid[idx]   # [B]
        snippet = self._stack_snippet(start_frames)

        H = int(horizon)
        offsets = torch.arange(H + 1, device=self.device).unsqueeze(0)   # [1, H+1]
        frame_idx = start_frames.unsqueeze(1) + offsets                   # [B, H+1]
        # Bounds: ``frame_idx`` is guaranteed valid by construction of
        # future_valid_starts, so no clamp needed.
        kp_window = self.kp_buffer[frame_idx.reshape(-1)].reshape(
            batch_size, H + 1, self.num_keypoints, 3,
        )                                                                # [B, H+1, K, 3]
        return {"snippet": snippet, "kp_window": kp_window}

    # ------------------------------------------------------------------
    # Reference-state initialization (BFM-style RSI)
    # ------------------------------------------------------------------

    def sample_reset_states(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Sample expert frames for reference-state initialization.

        Samples uniformly over every frame in the flat buffer (not restricted
        to ``valid_starts`` — RSI doesn't need a trailing snippet window).
        Returns world-frame root pose + velocities and joint pose + velocities
        so the env can write them into the simulator.

        Returns:
            dict with keys:
                joint_pos:    [B, num_joints]
                joint_vel:    [B, num_joints]
                root_pos:     [B, 3]        world frame
                root_quat:    [B, 4]        world frame, wxyz convention
                root_lin_vel: [B, 3]        world frame
                root_ang_vel: [B, 3]        world frame
        """
        if not self.supports_reset_states:
            raise RuntimeError(
                "Expert dataset at {} does not contain RSI fields. Re-run "
                "scripts/precompute_expert_dataset.py to regenerate it.".format(self.dataset_path)
            )
        frame = torch.randint(0, self.total_frames, (batch_size,), device=self.device)
        return {
            "joint_pos": self.joint_pos_buffer[frame],
            "joint_vel": self.joint_vel_buffer[frame],
            "root_pos": self.root_pos_buffer[frame],
            "root_quat": self.root_quat_buffer[frame],
            "root_lin_vel": self.root_lin_vel_buffer[frame],
            "root_ang_vel": self.root_ang_vel_buffer[frame],
        }
