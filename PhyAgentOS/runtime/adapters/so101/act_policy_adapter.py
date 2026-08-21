"""ACT policy adapter for IPU-deployed ACT models.

Bridges the canonical PhyAgentOS runtime observation <-> the leisaac-sgs IPU
ACT server input/output format.

Server-side expectations (lerobot_ipu/act.py + server.py):
  Input obs keys:
    observation.state        : float32 [6]
    observation.images.front: uint8 HWC [480, 640, 3]
    observation.images.wrist: uint8 HWC [480, 640, 3]
    task                     : str  (unused by ACT but protocol-required)

  Output action:
    float32 [T, 6]   (joint position targets in LeRobot motor-degree scale)

The IPU ACT model was trained with state/action in LeRobot "motor degree"
scale (each joint mapped to its motor range, e.g. [-100, 100]). Isaac Sim
joints are in radians. This adapter performs the rad <-> motor-degree
conversion in both directions so the rest of the pipeline can stay in
Isaac rad space.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from PhyAgentOS.runtime.adapters.base import BasePolicyAdapter
from PhyAgentOS.runtime.watchdog.errors import AdapterError


# --------------------------------------------------------------------------
# SO-101 joint order + Isaac Sim USD limits vs LeRobot motor limits.
# The IPU ACT checkpoint was trained on LeRobot motor-degree state/action,
# so we must convert Isaac radians <-> motor degrees at the adapter boundary.
# Mirrors leisaac.utils.robot_utils.convert_leisaac_action_to_lerobot.
# --------------------------------------------------------------------------

_SO101_JOINT_ORDER = (
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper",
)
_SO101_USD_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 90.0),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-160.0, 160.0),
    "gripper": (-10.0, 100.0),
}
_SO101_MOTOR_LIMITS = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}


def _isaac_rad_to_lerobot_motor(isaac_rad: np.ndarray) -> np.ndarray:
    """Isaac Sim joint pos (rad) -> LeRobot motor degree.

    Mirrors leisaac.utils.robot_utils.convert_leisaac_action_to_lerobot.
    The IPU board's act.img was trained on SO101 motor-degree state
    (STATE_MEAN/ACTION_MEAN in act.py are in [-100, 100]-scale), so we
    must convert Isaac rad -> motor degree before sending.
    """
    arr = np.asarray(isaac_rad, dtype=np.float64)
    deg = arr / np.pi * 180.0
    out = np.zeros_like(deg)
    for i, name in enumerate(_SO101_JOINT_ORDER):
        j_lo, j_hi = _SO101_USD_JOINT_LIMITS_DEG[name]
        m_lo, m_hi = _SO101_MOTOR_LIMITS[name]
        out[..., i] = (deg[..., i] - j_lo) / (j_hi - j_lo) * (m_hi - m_lo) + m_lo
    return out.astype(np.float32)


def _lerobot_motor_to_isaac_rad(motor: np.ndarray) -> np.ndarray:
    """LeRobot motor degree -> Isaac Sim joint pos (rad).

    Mirrors leisaac.utils.robot_utils.convert_lerobot_action_to_leisaac.
    Board returns motor-degree actions; Isaac env expects rad.
    """
    arr = np.asarray(motor, dtype=np.float64)
    out_rad = np.zeros_like(arr)
    for i, name in enumerate(_SO101_JOINT_ORDER):
        j_lo, j_hi = _SO101_USD_JOINT_LIMITS_DEG[name]
        m_lo, m_hi = _SO101_MOTOR_LIMITS[name]
        deg = (arr[..., i] - m_lo) / (m_hi - m_lo) * (j_hi - j_lo) + j_lo
        out_rad[..., i] = deg / 180.0 * np.pi
    return out_rad.astype(np.float32)


class ACTPolicyAdapter(BasePolicyAdapter):
    """PhyAgentOS -> leisaac-sgs IPU ACT data translation."""

    def input_observation_contract(self) -> dict[str, Any]:
        return {
            "sensors": {
                "front_rgb": {"kind": "image", "dtype": "uint8", "layout": "HWC"},
                "wrist_rgb": {"kind": "image", "dtype": "uint8", "layout": "HWC"},
                "proprio": {"kind": "vector", "dtype": "float32", "shape": [6]},
            }
        }

    def output_action_contract(self) -> dict[str, Any]:
        return {"actions": {"dtype": "float32", "shape": ["T", 6]}}

    def to_policy_input(
        self,
        runtime_observation: dict[str, Any],
        session_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            sensors = runtime_observation["sensors"]
            front = sensors["front_rgb"]
            wrist = sensors["wrist_rgb"]
            proprio = sensors["proprio"]
        except KeyError as exc:
            raise AdapterError(f"ACT policy adapter missing sensor key: {exc.args[0]}") from exc

        front_data = self._extract_image(front, "front_rgb")
        wrist_data = self._extract_image(wrist, "wrist_rgb")
        proprio_data = self._extract_state(proprio)
        # Isaac rad -> LeRobot motor degree (model trained on motor-degree state).
        proprio_data = _isaac_rad_to_lerobot_motor(proprio_data).astype(np.float32)

        # task_description: propagate from runtime_observation or session_ctx.
        task_desc = (
            runtime_observation.get("task_description")
            or (runtime_observation.get("so101_isaac") or {}).get("task_description")
            or session_ctx.get("task_description")
            or ""
        )

        return {
            "observation.state": proprio_data,
            "observation.images.front": front_data,
            "observation.images.wrist": wrist_data,
            "task_description": str(task_desc),
        }

    def from_policy_output(
        self,
        policy_output: dict[str, Any],
        session_ctx: dict[str, Any],
    ) -> dict[str, Any]:
        if "actions" not in policy_output:
            raise AdapterError("ACT policy payload missing `actions`")
        actions = np.asarray(policy_output["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]
        if actions.ndim != 2:
            raise AdapterError(
                f"ACT `actions` must have shape [A] or [T,A], got {actions.shape}"
            )
        if actions.shape[1] != 6:
            raise AdapterError(
                f"ACT action shape mismatch: expected [T,6], got {actions.shape}"
            )

        # LeRobot motor degree -> Isaac rad (model outputs motor-degree actions).
        actions = _lerobot_motor_to_isaac_rad(actions).astype(np.float32)

        # Optional chunk truncation: clamp to session max_chunk_size.
        max_chunk = session_ctx.get("max_chunk_size")
        if isinstance(max_chunk, int) and actions.shape[0] > int(max_chunk):
            actions = actions[: int(max_chunk)]

        policy_meta = dict(policy_output.get("policy_meta", {}))
        action_contract = session_ctx.get("policy_action_contract", {}) or {}

        return {
            "chunk_id": f"act_chunk_{policy_meta.get('policy_latency_ms', 0):.0f}",
            "source_observation_id": session_ctx.get("source_observation_id"),
            "source_policy_seq": policy_meta.get("policy_seq", 0),
            "action_contract": {
                "id": action_contract.get("action_space_id", "so101_joint_pos_v1"),
                "representation": action_contract.get("representation", "joint_position_targets"),
                "frame": action_contract.get("frame", "joint"),
                "control_mode": action_contract.get("control_mode", "joint_position"),
                "dtype": "float32",
                "shape": [int(actions.shape[0]), 6],
                "normalized": False,  # motor-deg -> rad applied in this adapter
                "components": action_contract.get(
                    "components",
                    [
                        {"name": "shoulder_pan", "unit": "rad"},
                        {"name": "shoulder_lift", "unit": "rad"},
                        {"name": "elbow_flex", "unit": "rad"},
                        {"name": "wrist_flex", "unit": "rad"},
                        {"name": "wrist_roll", "unit": "rad"},
                        {"name": "gripper", "unit": "normalized"},
                    ],
                ),
            },
            "timing": {
                "policy_hz": action_contract.get("chunk", {}).get("policy_hz", 30),
                "horizon_steps": int(actions.shape[0]),
            },
            "actions": np.ascontiguousarray(actions, dtype=np.float32),
            "policy_meta": policy_meta,
        }

    # -- helpers --

    @staticmethod
    def _extract_image(sensor: Any, name: str) -> np.ndarray:
        """Accept both {"data": ...} sensor dict and raw ndarray."""
        if isinstance(sensor, dict):
            data = sensor.get("data")
        else:
            data = sensor
        if data is None:
            raise AdapterError(f"ACT policy adapter `{name}` has empty data")
        arr = np.asarray(data)
        if arr.ndim != 3:
            raise AdapterError(f"ACT `{name}` image must be rank-3, got {arr.shape}")
        # CHW -> HWC (server / IPU preprocessor expects HWC).
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.ascontiguousarray(arr)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        # Convert float [0,1] -> uint8 if needed. We pass uint8 to the IPU
        # client; the IPU server's own ACTPreprocessor handles ImageNet norm.
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 0.0
            if max_val <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    @staticmethod
    def _extract_state(sensor: Any) -> np.ndarray:
        if isinstance(sensor, dict):
            data = sensor.get("data")
        else:
            data = sensor
        if data is None:
            raise AdapterError("ACT policy adapter `proprio` has empty data")
        arr = np.asarray(data, dtype=np.float32).reshape(-1)
        if arr.shape[0] != 6:
            raise AdapterError(
                f"ACT proprio must have shape [6], got {arr.shape}"
            )
        return arr
