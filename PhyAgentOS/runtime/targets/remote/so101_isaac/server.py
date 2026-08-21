"""Standalone TargetWS server backed by a REAL leisaac SO-101 Isaac Sim env.

Runs in the `leisaac` conda env (py3.11 + Isaac Sim 5.1.0 + Isaac Lab 2.3.0).
Deliberately self-contained: does NOT import the PhyAgentOS package (keeps
the heavy deps isolated). Speaks the TargetWS msgpack-over-websocket wire
format, so the runtime side (`SO101IsaacRemoteTargetProxy` +
`SO101TargetAdapter`, in the `paos` env) talks to it transparently.

Launch (leisaac env):

  conda run -n leisaac python \\
    PhyAgentOS/runtime/targets/remote/so101_isaac/server.py \\
    --host 0.0.0.0 --port 9005 \\
    --task LeIsaac-SO101-PickOrange-v0 \\
    --sim-warmup-steps 30 --max-steps 280 \\
    --record-dir /tmp/so101_episodes --record-fps 30

Optional env vars:
  EXP_LEISAAC_ASSET_ROOT : override `assets/` directory (default: <repo>/assets)
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import traceback
from typing import Any, Dict, List, Tuple

import numpy as np

try:
    import cv2
except Exception:  # noqa: BLE001
    cv2 = None

import msgpack

# --- wire codec (string-key envelope, byte-compatible with PhyAgentOS) ---

RPC_VERSION = "phyagentos.runtime_rpc.v2"

# Hold pose during sim warmup (matches policy_inference.py:_sim_warmup).
SO101_WARMUP_ACTION = None  # filled lazily after env reset


def _pack_array(obj: Any) -> Any:
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError("Unsupported dtype: %s" % obj.dtype)
    if isinstance(obj, np.ndarray):
        return {
            "__ndarray__": True,
            "data": obj.tobytes(),
            "dtype": obj.dtype.str,
            "shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {"__npgeneric__": True, "data": obj.item(), "dtype": obj.dtype.str}
    return obj


def _unpack_array(obj: Dict[Any, Any]) -> Any:
    if not isinstance(obj, dict):
        return obj
    if obj.get("__ndarray__") or obj.get(b"__ndarray__"):
        data = obj.get("data", obj.get(b"data"))
        if isinstance(data, str):
            data = data.encode("latin1")
        dtype = obj.get("dtype", obj.get(b"dtype"))
        shape = obj.get("shape", obj.get(b"shape"))
        return np.ndarray(buffer=data, dtype=np.dtype(dtype), shape=shape)
    if obj.get("__npgeneric__") or obj.get(b"__npgeneric__"):
        data = obj.get("data", obj.get(b"data"))
        dtype = obj.get("dtype", obj.get(b"dtype"))
        return np.dtype(dtype).type(data)
    return obj


def packb(payload: Any) -> bytes:
    return msgpack.packb(payload, use_bin_type=True, default=_pack_array)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_unpack_array)


def make_response(request: Dict[str, Any], response_type: str, payload: Dict[str, Any]) -> bytes:
    return packb(
        {
            "version": RPC_VERSION,
            "type": response_type,
            "session_id": request.get("session_id"),
            "target_id": request.get("target_id"),
            "skillruntime_id": request.get("skillruntime_id"),
            "episode_id": request.get("episode_id"),
            "seq": int(request.get("seq", 0)),
            "timestamp_ns": time.time_ns(),
            "trace_id": request.get("trace_id"),
            "payload": payload,
        }
    )


class TargetProtocolError(Exception):
    pass


# --- leisaac SO-101 runtime ---------------------------------------------------

SO101_ISAAC_DEFAULT_CONFIG: Dict[str, Any] = {
    "task": "LeIsaac-SO101-PickOrange-v0",
    "device": "cuda:0",
    "seed": None,
    "episode_length_s": 24.0,  # ~ 24s * 30Hz = 720 max steps (capped by max_steps)
    "sim_warmup_steps": 30,
    "max_steps": 280,
    "max_chunk_size": 100,
    "language_instruction": "pick the orange and place it on the plate",
    "record_dir": None,
    "record_fps": 30,
    "eval_rounds": 0,  # <=0 disables time_out termination
    "asset_root": None,
}

# SO-101 joint order, MUST match lerobot_ipu/act.py:37-46
SO101_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
SO101_ACTION_DIM = 6


class SO101IsaacRuntime:
    """Wraps a leisaac / Isaac Lab ManagerBasedRLEnv for one session == one episode."""

    def __init__(self, config: Dict[str, Any] | None = None):
        self.config = dict(SO101_ISAAC_DEFAULT_CONFIG)
        self.config.update(config or {})
        self.session_id: str | None = None
        self.env = None
        self.task_type: str | None = None
        self.sim_cameras: List[str] = []
        self.language: str = str(self.config.get("language_instruction", ""))
        self.step_idx = 0
        self.success = False
        self.done = False
        self._total_reward = 0.0
        self._episode_chunks: List[Dict[str, Any]] = []
        self._last_obs: Dict[str, Any] | None = None
        self._last_raw_obs: Dict[str, Any] | None = None
        self._last_status: Dict[str, Any] = {
            "accepted": True,
            "safety_status": "idle",
            "executed_steps": 0,
        }
        self._frames: List[np.ndarray] = []
        self._episode_idx = 0
        self._last_video_path: str | None = None
        self._sim_app = None  # SimulationApp handle (must outlive env)

    # -- lifecycle --

    def _ensure_env(self) -> None:
        """Lazy env creation. Imported here so the leisaac env stays isolated."""
        if self.env is not None:
            return

        # Import leisaac FIRST so that gym entry points get registered.
        # This mirrors policy_inference.py:175 (`import leisaac`).
        import leisaac  # noqa: F401
        import gymnasium as gym
        import torch  # noqa: F401  (env.step needs torch tensors)
        from isaaclab.envs import ManagerBasedRLEnv
        from isaaclab.sensors import Camera
        from leisaac.envs.mdp.observations import get_task_type

        task_name = str(self.config["task"])
        device = str(self.config.get("device", "cuda:0"))

        # parse_env_cfg lives in leisaac.utils.io or isaaclab_tasks depending on version;
        # try both.
        try:
            from leisaac.utils.io import parse_env_cfg  # type: ignore
        except Exception:
            from isaaclab_tasks.utils.parse_cfg import parse_env_cfg  # type: ignore

        env_cfg = parse_env_cfg(task_name, device=device, num_envs=1)
        self.task_type = get_task_type(task_name)
        env_cfg.use_teleop_device(self.task_type)
        env_cfg.seed = int(self.config["seed"]) if self.config.get("seed") else int(time.time())
        env_cfg.episode_length_s = float(self.config.get("episode_length_s", 24.0))

        # Disable time_out if eval_rounds <= 0 (matches policy_inference.py).
        if int(self.config.get("eval_rounds", 0)) <= 0:
            if hasattr(env_cfg, "terminations") and hasattr(env_cfg.terminations, "time_out"):
                env_cfg.terminations.time_out = None
        env_cfg.recorders = None

        env: ManagerBasedRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped
        self.env = env

        # Resolve camera sensor keys (matches policy_inference.py:332-334).
        self.sim_cameras = [
            key
            for key, sensor in env.scene.sensors.items()
            if isinstance(sensor, Camera)
        ]
        if not self.sim_cameras:
            # Some tasks may use different sensor names; fall back to front/wrist.
            self.sim_cameras = ["front", "wrist"]

        # Keep a reference to the underlying SimulationApp so it doesn't get GC'd.
        try:
            self._sim_app = env.sim  # type: ignore[attr-defined]
        except Exception:
            self._sim_app = None

        print(
            "[so101_isaac] env ready task=%s cameras=%s task_type=%s"
            % (task_name, self.sim_cameras, self.task_type),
            flush=True,
        )

    def describe(self) -> Dict[str, Any]:
        self._ensure_env()
        sensor_schema: Dict[str, Any] = {}
        for key in self.sim_cameras:
            try:
                h, w = self.env.scene.sensors[key].image_shape  # (H, W)
            except Exception:
                h, w = 480, 640
            sensor_schema[key + "_rgb"] = {
                "kind": "image",
                "dtype": "uint8",
                "layout": "HWC",
                "shape": [int(h), int(w), 3],
            }
        sensor_schema["proprio"] = {
            "kind": "vector",
            "dtype": "float32",
            "shape": [SO101_ACTION_DIM],
        }
        return {
            "runtime": "SO101IsaacRemoteTargetRuntime",
            "task": self.config["task"],
            "task_description": self.language,
            "cameras": list(self.sim_cameras),
            "observation_schema": sensor_schema,
            "action_contract": {
                "id": "so101_joint_pos_v1",
                "shape": ["T", SO101_ACTION_DIM],
                "dtype": "float32",
                "normalized": False,
                "frame": "joint",
                "max_chunk_size": int(self.config["max_chunk_size"]),
            },
        }

    def configure_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        # Accept overrides via metadata.so101_isaac or top-level so101_isaac.
        meta = ctx.get("metadata", {}) or {}
        overrides = meta.get("so101_isaac") or ctx.get("so101_isaac") or {}
        self.config.update(overrides)
        return {
            "configured": True,
            "session_id": self.session_id,
            "so101_isaac": self._metadata(),
        }

    def start_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self.session_id = ctx.get("session_id", self.session_id)
        return {"started": True, "session_id": self.session_id, "so101_isaac": self._metadata()}

    def reset(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._ensure_env()
        self.session_id = ctx.get("session_id", self.session_id)
        meta = ctx.get("metadata", {}) or {}
        overrides = meta.get("so101_isaac") or ctx.get("so101_isaac") or {}
        self.config.update(overrides)

        obs_dict, _info = self.env.reset()
        obs_dict = self._sim_warmup(obs_dict)

        self.step_idx = 0
        self.success = False
        self.done = False
        self._total_reward = 0.0
        self._episode_chunks = []
        self._frames = []
        self._last_video_path = None
        self._record_frame(obs_dict)
        self._last_raw_obs = obs_dict
        self._last_obs = self._format_obs(obs_dict)
        self._last_status = {
            "accepted": True,
            "safety_status": "ok",
            "executed_steps": 0,
            "target_step_index": 0,
            "success": False,
            "done": False,
            "reward": 0.0,
            "obs": self._last_obs,
        }
        return self._last_obs

    def _sim_warmup(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Hold reset pose for N steps to let Isaac Sim settle (DomeLight, render graph)."""
        n = int(self.config.get("sim_warmup_steps", 30))
        if n <= 0:
            return obs_dict
        import torch

        hold = obs_dict["policy"]["joint_pos"].clone()
        for _ in range(n):
            obs_dict, _, _, _, _ = self.env.step(hold)
        return obs_dict

    def observe(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self._last_obs is None:
            raise TargetProtocolError("SO101 observe before reset")
        return self._last_obs

    def action_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        import torch

        actions_np = np.asarray(chunk.get("actions"), dtype=np.float32)
        if actions_np.ndim != 2 or actions_np.shape[1] != SO101_ACTION_DIM:
            raise TargetProtocolError(
                "SO101 expected actions [T,%d], got %s"
                % (SO101_ACTION_DIM, getattr(actions_np, "shape", None))
            )
        if actions_np.shape[0] > int(self.config["max_chunk_size"]):
            raise TargetProtocolError(
                "SO101 action chunk too large: %d > %d"
                % (actions_np.shape[0], int(self.config["max_chunk_size"]))
            )
        if not np.isfinite(actions_np).all():
            raise TargetProtocolError("SO101 actions contain NaN or Inf")

        device = str(self.config.get("device", "cuda:0"))
        max_steps = int(self.config.get("max_steps", 280))
        chunk_reward = 0.0
        first_step = self.step_idx + 1
        obs_dict = self._last_raw_obs

        for i in range(actions_np.shape[0]):
            action = torch.from_numpy(actions_np[i:i + 1]).to(device)  # [1, 6]
            obs_dict, reward, terminated, truncated, _info = self.env.step(action)
            self.step_idx += 1
            chunk_reward += float(reward)
            self._record_frame(obs_dict)

            # Termination policy matches policy_inference.py: success dominates timeout.
            if bool(terminated[0]):
                self.success = True
                self.done = True
                break
            if self.step_idx >= max_steps:
                self.done = True
                break

        self._total_reward += chunk_reward
        self._last_raw_obs = obs_dict
        self._last_obs = self._format_obs(obs_dict)
        if self.done:
            self._last_video_path = self._write_video()

        executed = max(0, self.step_idx - first_step + 1)
        self._episode_chunks.append(
            {
                "chunk_id": chunk.get("chunk_id", "so101_chunk"),
                "first_step": first_step,
                "executed_steps": executed,
                "requested_steps": int(actions_np.shape[0]),
                "reward": chunk_reward,
                "success": bool(self.success),
                "done": bool(self.done),
                "action_shape": [int(actions_np.shape[0]), int(actions_np.shape[1])],
            }
        )
        self._last_status = {
            "chunk_id": chunk.get("chunk_id", "so101_chunk"),
            "accepted": True,
            "buffered_steps": 0,
            "executed_steps": self.step_idx,
            "target_step_index": self.step_idx,
            "need_replan": not self.success,
            "safety_status": "ok",
            "success": bool(self.success),
            "done": bool(self.done),
            "reward": chunk_reward,
            "obs": self._last_obs,
            "so101_isaac": self._metadata(),
            "episode_summary": self._episode_summary(),
        }
        return dict(self._last_status)

    def execution_status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def cancel(self, reason: str) -> Dict[str, Any]:
        self._last_status = dict(self._last_status)
        self._last_status.update({"cancelled": True, "cancel_reason": reason})
        return {"cancelled": True, "reason": reason}

    def close(self) -> Dict[str, Any]:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass
        self.env = None
        return {"closed": True}

    # -- helpers --

    def _format_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Turn Isaac Lab obs_dict['policy'] into the PhyAgentOS sensor layout.

        Output layout (matches SO101TargetAdapter contract):
          sensors:
            front_rgb: uint8 [H, W, 3] HWC
            wrist_rgb: uint8 [H, W, 3] HWC
            proprio:   float32 [6]
          task_description: str
        """
        policy = obs_dict["policy"] if "policy" in obs_dict else obs_dict
        sensors: Dict[str, Any] = {}

        # Images: Isaac Sim Camera returns CHW float in [0,1] (or uint8). Convert to HWC uint8.
        for camera_key in self.sim_cameras:
            if camera_key not in policy:
                continue
            img = self._to_hwc_uint8(policy[camera_key])
            sensors[camera_key + "_rgb"] = img

        # Ensure canonical keys front_rgb / wrist_rgb always present (adapter expects them).
        if "front_rgb" not in sensors and self.sim_cameras:
            sensors["front_rgb"] = sensors[self.sim_cameras[0] + "_rgb"]
        if "wrist_rgb" not in sensors and len(self.sim_cameras) >= 2:
            sensors["wrist_rgb"] = sensors[self.sim_cameras[1] + "_rgb"]

        # Proprio: SO-101 joint_pos, shape [1, 6] -> squeeze to [6].
        joint_pos = policy.get("joint_pos")
        if joint_pos is None:
            raise TargetProtocolError("SO101 obs missing policy.joint_pos")
        proprio = np.asarray(joint_pos.detach().cpu() if hasattr(joint_pos, "detach") else joint_pos, dtype=np.float32)
        proprio = np.atleast_1d(proprio).reshape(-1)[:SO101_ACTION_DIM]
        sensors["proprio"] = proprio

        return {
            "observation_id": "so101_obs_%d" % self.step_idx,
            "sensors": sensors,
            "task_description": self.language,
            "step_index": int(self.step_idx),
            "timestamp_ns": time.time_ns(),
        }

    @staticmethod
    def _to_hwc_uint8(img: Any) -> np.ndarray:
        """Replicates lerobot_ipu/act.py:image_to_hwc_uint8 logic on a single image."""
        arr = img
        if hasattr(arr, "detach"):
            arr = arr.detach().cpu().numpy()
        arr = np.asarray(arr)
        if arr.ndim != 3:
            raise TargetProtocolError("SO101 image must be rank-3, got shape %s" % (arr.shape,))
        # CHW -> HWC if first axis looks like channels.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.ascontiguousarray(arr)
        if arr.shape[-1] == 4:  # RGBA -> RGB
            arr = arr[:, :, :3]
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 0.0
            if max_val <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)
        return arr

    def _record_frame(self, obs_dict: Dict[str, Any]) -> None:
        if not self.config.get("record_dir"):
            return
        try:
            policy = obs_dict["policy"] if "policy" in obs_dict else obs_dict
            cam_keys = self.sim_cameras or ["front", "wrist"]
            frames = [self._to_hwc_uint8(policy[k]) for k in cam_keys if k in policy]
            if not frames:
                return
            self._frames.append(np.ascontiguousarray(np.hstack(frames)))
        except Exception:
            pass  # recording is best-effort

    def _write_video(self) -> str | None:
        record_dir = self.config.get("record_dir")
        if not record_dir or not self._frames:
            return None
        if cv2 is None:
            print("[record] cv2 unavailable; skipping video", flush=True)
            return None
        os.makedirs(str(record_dir), exist_ok=True)
        self._episode_idx += 1
        tag = "success" if self.success else "fail"
        name = "so101_ep_%02d_%s.mp4" % (self._episode_idx, tag)
        path = os.path.join(str(record_dir), name)
        h, w = self._frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, float(self.config.get("record_fps", 30)), (w, h))
        for frame in self._frames:
            writer.write(frame[:, :, ::-1])  # RGB -> BGR
        writer.release()
        print("[record] wrote %d frames -> %s (%s)" % (len(self._frames), path, tag), flush=True)
        return path

    def _metadata(self) -> Dict[str, Any]:
        return {
            "task": self.config["task"],
            "step_index": self.step_idx,
            "task_description": self.language,
            "cameras": list(self.sim_cameras),
        }

    def _episode_summary(self) -> Dict[str, Any]:
        summary = {
            "task": self.config["task"],
            "task_description": self.language,
            "target_step_index": self.step_idx,
            "success": bool(self.success),
            "done": bool(self.done),
            "total_reward": float(self._total_reward),
            "num_action_chunks": len(self._episode_chunks),
            "chunks": list(self._episode_chunks),
            "final_observation_id": self._last_obs.get("observation_id") if self._last_obs else None,
        }
        if self._last_video_path:
            summary["video_path"] = self._last_video_path
        return summary


# --- dispatch + ws server -----------------------------------------------------


def _dispatch(runtime: SO101IsaacRuntime, request: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    rtype = request.get("type")
    payload = request.get("payload") or {}
    if rtype == "target.describe":
        return rtype, runtime.describe()
    if rtype == "target.configure_session":
        return rtype, runtime.configure_session(payload)
    if rtype == "target.start_session":
        return rtype, runtime.start_session(payload)
    if rtype == "target.reset":
        return rtype, runtime.reset(payload)
    if rtype == "target.observe":
        return "target.observation", runtime.observe(payload)
    if rtype == "target.action_chunk":
        return rtype, runtime.action_chunk(payload)
    if rtype == "target.execution_status":
        return rtype, runtime.execution_status()
    if rtype == "target.cancel":
        return rtype, runtime.cancel(str(payload.get("reason", "cancelled")))
    if rtype == "target.close":
        return rtype, runtime.close()
    raise TargetProtocolError("unsupported target RPC type: %s" % rtype)


async def serve(runtime: SO101IsaacRuntime, host: str, port: int) -> None:
    import websockets

    async def handler(ws):
        async for message in ws:
            if isinstance(message, str):
                await ws.send(
                    packb(
                        {
                            "version": RPC_VERSION,
                            "type": "runtime.error",
                            "seq": 0,
                            "timestamp_ns": time.time_ns(),
                            "payload": {
                                "error_code": "BAD_PAYLOAD",
                                "message": "expected binary msgpack",
                            },
                        }
                    )
                )
                continue
            try:
                request = unpackb(message)
            except Exception as exc:
                await ws.send(
                    packb(
                        {
                            "version": RPC_VERSION,
                            "type": "runtime.error",
                            "seq": 0,
                            "timestamp_ns": time.time_ns(),
                            "payload": {
                                "error_code": "BAD_MSGPACK",
                                "message": "failed to decode msgpack: %s" % exc,
                            },
                        }
                    )
                )
                continue
            try:
                rtype, payload = _dispatch(runtime, request)
                await ws.send(make_response(request, rtype, payload))
            except Exception as exc:  # noqa: BLE001
                await ws.send(
                    make_response(
                        request,
                        "runtime.error",
                        {
                            "error_code": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        },
                    )
                )

    async with websockets.serve(handler, host, port, max_size=None, compression=None):
        print("SO-101 Isaac target server listening on ws://%s:%d" % (host, port), flush=True)
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9005)
    parser.add_argument("--task", default="LeIsaac-SO101-PickOrange-v0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--episode-length-s", type=float, default=24.0)
    parser.add_argument("--sim-warmup-steps", type=int, default=30)
    parser.add_argument("--max-steps", type=int, default=280)
    parser.add_argument("--max-chunk-size", type=int, default=100)
    parser.add_argument("--language-instruction", default="pick the orange and place it on the plate")
    parser.add_argument("--eval-rounds", type=int, default=0)
    parser.add_argument("--record-dir", default=None)
    parser.add_argument("--record-fps", type=int, default=30)
    args = parser.parse_args()

    runtime = SO101IsaacRuntime(
        {
            "task": args.task,
            "device": args.device,
            "seed": args.seed,
            "episode_length_s": args.episode_length_s,
            "sim_warmup_steps": args.sim_warmup_steps,
            "max_steps": args.max_steps,
            "max_chunk_size": args.max_chunk_size,
            "language_instruction": args.language_instruction,
            "eval_rounds": args.eval_rounds,
            "record_dir": args.record_dir,
            "record_fps": args.record_fps,
        }
    )
    try:
        asyncio.run(serve(runtime, args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        try:
            runtime.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
