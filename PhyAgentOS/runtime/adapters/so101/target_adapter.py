"""SO-101 Isaac Sim target adapter.

Translates between canonical PhyAgentOS runtime observations and the
SO-101 Isaac target server wire format (see runtime/targets/remote/so101_isaac/server.py).

Key contract details:
- Sensors: front_rgb + wrist_rgb (uint8 HWC) + proprio (float32 [6])
- Action: float32 [T, 6] joint position targets (NOT delta EEF)
- Normalization happens inside the IPU ACT server, so this adapter is
  normalized=False end-to-end.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PhyAgentOS.runtime.adapters.base import BaseTargetAdapter
from PhyAgentOS.runtime.watchdog.errors import AdapterError


class SO101TargetAdapter(BaseTargetAdapter):
    """SO-101 (6-DoF) Isaac Sim target adapter."""

    def output_observation_contract(self) -> dict[str, Any]:
        return {
            "sensors": {
                "front_rgb": {"kind": "image", "dtype": "uint8", "layout": "HWC"},
                "wrist_rgb": {"kind": "image", "dtype": "uint8", "layout": "HWC"},
                "proprio": {"kind": "vector", "dtype": "float32", "shape": [6]},
            }
        }

    def input_action_contract(self) -> dict[str, Any]:
        return {"actions": {"dtype": "float32", "shape": ["T", 6]}}

    def to_runtime_observation(
        self,
        raw_obs: dict[str, Any],
        target_info: dict[str, Any],
    ) -> dict[str, Any]:
        """The SO-101 target server already returns canonical sensors; pass-through.

        Supports two wire formats:
        1. ``sensors`` block: {front_rgb, wrist_rgb, proprio} (SO101IsaacRuntime)
        2. ``images`` + ``state`` block: {images: {front, wrist}, state: [6]}
           (So101IsaacSimRuntime)
        """
        sensors_in = raw_obs.get("sensors") or {}
        if not sensors_in:
            # Fall back to the images/state wire format used by So101IsaacSimRuntime.
            images = raw_obs.get("images") or {}
            state = raw_obs.get("state")
            if images and state is not None:
                sensors_in = {
                    "front_rgb": images.get("front"),
                    "wrist_rgb": images.get("wrist"),
                    "proprio": state,
                }
            else:
                raise AdapterError(
                    "SO101 target adapter expected `sensors` block from raw_obs; "
                    f"got keys: {list(raw_obs.keys())}"
                )

        # The server emits {<camera>_rgb: uint8 HWC, proprio: float32 [6]}.
        front = sensors_in.get("front_rgb")
        wrist = sensors_in.get("wrist_rgb")
        proprio = sensors_in.get("proprio")

        if front is None or wrist is None:
            raise AdapterError("SO101 target adapter requires both front_rgb and wrist_rgb sensors")
        if proprio is None:
            raise AdapterError("SO101 target adapter requires proprio sensor")

        front_arr = np.asarray(front, dtype=np.uint8)
        wrist_arr = np.asarray(wrist, dtype=np.uint8)
        proprio_arr = np.asarray(proprio, dtype=np.float32).reshape(-1)
        if proprio_arr.shape[0] != 6:
            raise AdapterError(f"SO101 proprio must have shape [6], got {proprio_arr.shape}")

        sensors: dict[str, Any] = {
            "front_rgb": {
                "kind": "image",
                "observation_key": "front_rgb",
                "data": np.ascontiguousarray(front_arr),
                "dtype": "uint8",
                "layout": "HWC",
            },
            "wrist_rgb": {
                "kind": "image",
                "observation_key": "wrist_rgb",
                "data": np.ascontiguousarray(wrist_arr),
                "dtype": "uint8",
                "layout": "HWC",
            },
            "proprio": {
                "kind": "vector",
                "observation_key": "proprio",
                "data": proprio_arr,
                "dtype": "float32",
            },
        }

        return {
            "observation_id": raw_obs.get(
                "observation_id",
                f"so101_obs_{target_info.get('step_index', 0)}",
            ),
            "sensors": sensors,
            "target_info": target_info,
            "so101_isaac": {
                "task_description": raw_obs.get("task_description"),
                "step_index": int(raw_obs.get("step_index", 0)),
            },
        }

    def to_executable_action_chunk(
        self,
        action_chunk: dict[str, Any],
        target_info: dict[str, Any],
    ) -> dict[str, Any]:
        actions = np.asarray(action_chunk.get("actions"), dtype=np.float32)
        if actions.ndim != 2:
            raise AdapterError(f"SO101 actions must have shape [T,6], got {actions.shape}")
        if actions.shape[1] != 6:
            raise AdapterError(f"SO101 action shape mismatch: expected [T,6], got {actions.shape}")

        max_chunk_size = target_info.get("max_chunk_size")
        if max_chunk_size is not None and actions.shape[0] > int(max_chunk_size):
            raise AdapterError(
                f"SO101 action chunk too large: {actions.shape[0]} > {int(max_chunk_size)}"
            )
        if not np.isfinite(actions).all():
            raise AdapterError("SO101 actions contain NaN or Inf")

        contract = dict(action_chunk.get("action_contract", {}))
        contract.setdefault("id", target_info.get("action_contract_id", "so101_joint_pos_v1"))
        contract.setdefault("shape", [actions.shape[0], actions.shape[1]])
        contract.setdefault("representation", "joint_position_targets")
        contract.setdefault("frame", "joint")

        return {
            "chunk_id": action_chunk.get("chunk_id", "so101_chunk"),
            "source_observation_id": action_chunk.get("source_observation_id"),
            "source_policy_seq": action_chunk.get("source_policy_seq"),
            "action_contract": contract,
            "provenance": action_chunk.get("provenance", {}),
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
        }
