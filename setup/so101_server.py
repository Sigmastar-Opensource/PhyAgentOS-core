"""SO101 Isaac Sim TargetWS server.

Runs on the 4090 GPU server. Wraps an Isaac Sim + LeIsaac environment
(registered as LeIsaac-SO101-PickOrange-v0) as a TargetWS endpoint that
PhyAgentOS's runtime can drive via targetws://<host>:<port>.

This file deliberately does NOT import PhyAgentOS (pydantic/py3.11+ would
conflict with Isaac Sim's python 3.11 + torch environment). It only depends
on: numpy, msgpack, websockets, torch, isaacsim, isaaclab, leisaac.

Launch (Isaac env, from leisaac-sgs repo root):

  conda run -n leisaac python so101_server.py \
    --task LeIsaac-SO101-PickOrange-v0 --port 9002 --gui \
    --step-hz 30 --max-steps 3600

Defaults are tuned for the SO101 task: 30 Hz control, 3600-step (120 s) cap,
32-action chunks.
"""

from __future__ import annotations

import argparse
import logging
import queue
import threading
import time
import traceback
from typing import Any, Dict

import msgpack
import numpy as np

RPC_VERSION = "phyagentos.runtime_rpc.v2"
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# msgpack numpy codec (kept local ¡ª no PhyAgentOS imports)
# --------------------------------------------------------------------------

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
    if obj.get("__ndarray__"):
        return np.ndarray(
            buffer=obj["data"],
            dtype=np.dtype(obj["dtype"]),
            shape=obj["shape"],
        )
    if obj.get("__npgeneric__"):
        return np.dtype(obj["dtype"]).type(obj["data"])
    return obj


def packb(payload: Any) -> bytes:
    return msgpack.packb(payload, use_bin_type=True, default=_pack_array)


def unpackb(data: bytes) -> Any:
    return msgpack.unpackb(data, raw=False, object_hook=_unpack_array)


def make_response(request: Dict[str, Any], response_type: str, payload: Dict[str, Any]) -> bytes:
    return packb({
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
    })


class TargetProtocolError(Exception):
    pass


# --------------------------------------------------------------------------
# SO101 Isaac Sim runtime
# --------------------------------------------------------------------------

SO101_DEFAULT_CONFIG = {
    "task": "LeIsaac-SO101-PickOrange-v0",
    "robot_id": "so101",
    # External contract (matches VLA model output + LeRobot dataset):
    "action_dim": 6,
    "state_dim": 6,
    # Internal Isaac Sim action space (probed at startup):
    # LeIsaac SO101 envs typically expect 8 (6 joints + 2 gripper finger actions).
    # Set to None to auto-probe from env.action_manager.total_action_dim.
    "isaac_action_dim": None,
    # How to fill dims [6:8] when model only outputs 6:
    #   "gripper_mirror" -> action[6]=action[7]=action[5] (gripper cmd -> both fingers)
    #   "zero"           -> action[6]=action[7]=0.0
    "extra_action_fill": "gripper_mirror",
    "max_chunk_size": 32,
    "max_steps": 3600,
    "step_hz": 30,
    "image_size": 480,
    "image_key": "front",
    "wrist_image_key": "wrist",
    "headless": True,
    "device": "cuda",
}


class So101IsaacSimRuntime:
    """Isaac Sim target runtime wrapping a LeIsaac SO101 environment."""

    def __init__(self, env: Any, config: Dict[str, Any] | None = None):
        self.env = env
        self.config = dict(SO101_DEFAULT_CONFIG)
        self.config.update(config or {})
        self.session_id = None
        self.step_idx = 0
        self.success = False
        self.done = False
        self._last_obs: Dict[str, Any] | None = None
        self._last_status: Dict[str, Any] = {
            "accepted": True,
            "safety_status": "idle",
            "executed_steps": 0,
        }
        # Probe the real Isaac Sim action space so we can do model_dim -> env_dim
        # padding inside _to_action_tensor. This is critical: LeIsaac SO101 envs
        # often expect 8 (6 joints + 2 gripper finger actions) while the VLA
        # model outputs 6 (matching the LeRobot dataset).
        self._probe_isaac_action_space()

    def _probe_isaac_action_space(self) -> None:
        """Inspect env.action_manager / action_space and cache the real dim.

        Prints a detailed breakdown so you can verify the layout from logs.
        """
        isaac_dim = self.config.get("isaac_action_dim")
        if isaac_dim is None:
            try:
                am = getattr(getattr(self.env, "unwrapped", self.env), "action_manager", None)
                if am is None:
                    am = getattr(self.env, "action_manager", None)
                if am is not None:
                    isaac_dim = int(getattr(am, "total_action_dim", 0)) or None
            except Exception:
                isaac_dim = None
        if isaac_dim is None:
            try:
                sp = getattr(self.env, "action_space", None)
                if sp is not None and hasattr(sp, "shape"):
                    isaac_dim = int(sp.shape[-1])
            except Exception:
                pass
        if isaac_dim is None:
            # Fallback: assume env matches the external contract.
            isaac_dim = int(self.config["action_dim"])

        self.config["isaac_action_dim"] = int(isaac_dim)
        model_dim = int(self.config["action_dim"])

        if isaac_dim < model_dim:
            print(
                f"[so101_server] WARNING: env dim ({isaac_dim}) < model dim ({model_dim}); "
                f"extra model dims will be truncated.",
                flush=True,
            )
        elif isaac_dim > model_dim:
            extra = isaac_dim - model_dim

    # ------------------------------------------------------------------
    # TargetWS RPC handlers
    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        return {
            "runtime": "So101IsaacSimRuntime",
            "robot_id": self.config.get("robot_id", "so101"),
            "observation_schema": {
                "images": {
                    "front": "uint8 HWC",
                    "wrist": "uint8 HWC",
                },
                "state": {
                    "dtype": "float32",
                    "shape": [self.config["state_dim"]],
                    "keys": [
                        "shoulder_pan.pos",
                        "shoulder_lift.pos",
                        "elbow_flex.pos",
                        "wrist_flex.pos",
                        "wrist_roll.pos",
                        "gripper.pos",
                    ],
                },
            },
            "action_contract": {
                "id": "so101_isaacsim_joint_v1",
                "shape": ["T", int(self.config["action_dim"])],
                "dtype": "float32",
                "representation": "joint_position",
                "max_chunk_size": int(self.config["max_chunk_size"]),
            },
            "agent_tools": [],
        }

    def configure_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._merge_ctx(ctx)
        return {"configured": True, "session_id": self.session_id}

    def start_session(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._merge_ctx(ctx)
        return {"started": True, "session_id": self.session_id}

    def reset(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        self._merge_ctx(ctx)
        try:
            obs_dict, _ = self.env.reset()
        except Exception as exc:
            raise TargetProtocolError(f"env.reset() failed: {exc}") from exc
        self.step_idx = 0
        self.success = False
        self.done = False
	    # === start sim_warmup ===
        warmup_steps = int(self.config.get("sim_warmup_steps", 120))
        if warmup_steps > 0 and obs_dict.get("policy") is not None:
            hold_np = np.zeros(int(self.config["action_dim"]), dtype=np.float32)
            for _ in range(warmup_steps):
                action_tensor = self._to_action_tensor(hold_np)
                obs_dict, _, _, _, _ = self.env.step(action_tensor)
                self.step_idx += 1
	    # === end sim_warmup ===
        self._last_obs = self._format_obs(obs_dict)
        self._last_status = self._base_status(reward=0.0)
        return self._last_obs

    def observe(self, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if self._last_obs is None:
            raise TargetProtocolError("observe() called before reset()")
        return self._last_obs

    def action_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        actions = np.asarray(chunk.get("actions"), dtype=np.float32)
        action_dim = int(self.config["action_dim"])
        if actions.ndim != 2 or actions.shape[1] != action_dim:
            raise TargetProtocolError(
                f"SO101 expected actions [T,{action_dim}], got {actions.shape}"
            )
        max_chunk = int(self.config["max_chunk_size"])
        if actions.shape[0] > max_chunk:
            raise TargetProtocolError(
                f"SO101 action chunk too large: {actions.shape[0]} > {max_chunk}"
            )

        step_hz = float(self.config.get("step_hz", 30))
        step_dt = 1.0 / step_hz if step_hz > 0 else 1.0 / 30.0
        chunk_reward = 0.0

        for row in actions:
            try:
                action_tensor = self._to_action_tensor(row)
                obs_dict, reward, terminated, truncated, info = self.env.step(
                    action_tensor
                )
            except Exception as exc:
                raise TargetProtocolError(
                    f"env.step() failed at step {self.step_idx}: {exc}"
                ) from exc

            self.step_idx += 1
            chunk_reward += float(reward)
            self._last_obs = self._format_obs(obs_dict)

            info_dict = dict(info or {})
            if info_dict.get("success"):
                self.success = True
            if terminated or truncated:
                self.done = True
            if self.step_idx >= int(self.config.get("max_steps", 3600)):
                self.done = True
            if self.done:
                break
            time.sleep(step_dt)  # control rate

        self._last_status = self._base_status(
            reward=chunk_reward, chunk_id=chunk.get("chunk_id")
        )
        return dict(self._last_status)

    def execution_status(self) -> Dict[str, Any]:
        return dict(self._last_status)

    def initial_render(self, warmup_steps: int = 200) -> None:
        """Run a one-shot warmup at startup so the GUI fully renders.

        Stepping the env with a zero action for `warmup_steps` times lets the
        viewport, assets, and physics settle before the server starts accepting
        RPC requests. This is called ONCE from serve_blocking() right after the
        env is built. After this returns, the main loop just waits for RPC and
        never injects any extra step (so it does NOT interfere with real
        action_chunk rollouts — the root cause of the previous "robot jitters"
        bug was that idle steps were being run BETWEEN action chunks).

        This mirrors what reset() already does via its sim_warmup_steps block,
        but here we do it before any session is started, purely for display.
        """
        if warmup_steps <= 0:
            return
        hold_np = np.zeros(int(self.config["action_dim"]), dtype=np.float32)
        try:
            action_tensor = self._to_action_tensor(hold_np)
            for i in range(warmup_steps):
                obs_dict, _r, _t, _tr, _info = self.env.step(action_tensor)
                self._last_obs = self._format_obs(obs_dict)
        except Exception as exc:
            logger.debug("initial_render() skipped: %s", exc)

    def cancel(self, reason: str) -> Dict[str, Any]:
        self._last_status = dict(self._last_status)
        self._last_status.update({"cancelled": True, "cancel_reason": reason})
        return {"cancelled": True, "reason": reason}

    def close(self) -> Dict[str, Any]:
        try:
            self.env.close()
        except Exception:
            pass
        return {"closed": True}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _merge_ctx(self, ctx: Dict[str, Any]) -> None:
        self.session_id = ctx.get("session_id", self.session_id)
        metadata = dict(ctx.get("metadata") or {})
        so101_cfg = dict(metadata.get("so101") or {})
        self.config.update(so101_cfg)

    def _to_action_tensor(self, action: np.ndarray) -> Any:
        """Convert a model-dim action row to Isaac Sim's expected action format.

        The VLA model outputs `action_dim` dims (6 for SO101: 5 arm + 1 gripper).
        Isaac Sim's action_manager may expect more dims (often 8 = 6 joints +
        2 gripper finger positions). This method pads/truncates to match the
        probed `isaac_action_dim`.

        Padding policy (config `extra_action_fill`):
          - "gripper_mirror": extra dims = action[action_dim-1] (gripper cmd)
          - "zero":           extra dims = 0.0
        """
        import torch

        model_dim = int(self.config["action_dim"])
        isaac_dim = int(self.config.get("isaac_action_dim") or model_dim)
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape[0] < model_dim:
            padded = np.zeros((model_dim,), dtype=np.float32)
            padded[: action.shape[0]] = action
            action = padded
        elif action.shape[0] > model_dim:
            action = action[:model_dim]

        if isaac_dim > model_dim:
            fill_value = 0.0
            if self.config.get("extra_action_fill", "gripper_mirror") == "gripper_mirror":
                # Last model dim is the gripper command; mirror it to extra dims.
                fill_value = float(action[model_dim - 1])
            extra = np.full((isaac_dim - model_dim,), fill_value, dtype=np.float32)
            action = np.concatenate([action, extra])
        elif isaac_dim < model_dim:
            action = action[:isaac_dim]

        return torch.tensor(
            action.reshape(1, -1),
            dtype=torch.float32,
            device=self.config.get("device", "cpu"),
        )

    def _format_obs(self, obs_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Isaac obs dict -> TargetWS observation format.

        LeIsaac SO101 environments return obs["policy"] with keys:
          - joint_pos:     torch.Tensor [1, 6] float32 (radians)
          - joint_pos_target: torch.Tensor [1, 6] float32
          - front:         torch.Tensor [1, H, W, 3] uint8
          - wrist:         torch.Tensor [1, H, W, 3] uint8
        Note: NO "observation.state" / "observation.images.front" keys exist
        in this env. Old code fell back to zeros ¡ª that was the bug.
        """
        import torch

        def _to_numpy(t):
            if isinstance(t, torch.Tensor):
                return t.detach().cpu().numpy()
            return t

        def _strip_batch(arr):
            arr = np.asarray(arr)
            if arr.ndim >= 1 and arr.shape[0] == 1:
                arr = arr[0]
            return arr

        def _first_non_none(*candidates):
            """Return the first candidate that is not None.

            Uses explicit `is None` checks (never `or`) so multi-element
            torch tensors don't trigger 'Boolean value of Tensor is ambiguous'.
            """
            for c in candidates:
                if c is not None:
                    return c
            return None

        policy_obs = obs_dict.get("policy", obs_dict)
        if not isinstance(policy_obs, dict):
            policy_obs = {}
        image_key = str(self.config.get("image_key", "front"))
        wrist_key = str(self.config.get("wrist_image_key", "wrist"))

        # ---- State ----
        state = _first_non_none(
            policy_obs.get("joint_pos"),
            policy_obs.get("observation.state"),
            policy_obs.get("state"),
        )
        if state is None:
            state = np.zeros((self.config["state_dim"],), dtype=np.float32)
        state = _strip_batch(_to_numpy(state))
        state = np.asarray(state, dtype=np.float32).reshape(-1)
        if state.shape[0] < self.config["state_dim"]:
            padded = np.zeros((self.config["state_dim"],), dtype=np.float32)
            padded[: state.shape[0]] = state
            state = padded
        else:
            state = state[: self.config["state_dim"]]

        # ---- Images ----
        target_h = int(self.config.get("image_size", 480))
        target_w = int(self.config.get("image_size", 480))

        images = {}

        # Front camera
        front = _first_non_none(
            policy_obs.get(image_key),
            policy_obs.get("observation.images.front"),
            (policy_obs.get("images") or {}).get(image_key) if isinstance(policy_obs.get("images"), dict) else None,
        )
        if front is None:
            front = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        front = _strip_batch(_to_numpy(front))
        front = np.asarray(front, dtype=np.uint8)
        if front.ndim == 3 and front.shape[:2] != (target_h, target_w):
            front = self._resize_image(front, target_h, target_w)
        images[image_key] = np.ascontiguousarray(front, dtype=np.uint8)

        # Wrist camera
        wrist = _first_non_none(
            policy_obs.get(wrist_key),
            policy_obs.get("observation.images.wrist"),
            (policy_obs.get("images") or {}).get(wrist_key) if isinstance(policy_obs.get("images"), dict) else None,
        )
        if wrist is None:
            wrist = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        wrist = _strip_batch(_to_numpy(wrist))
        wrist = np.asarray(wrist, dtype=np.uint8)
        if wrist.ndim == 3 and wrist.shape[:2] != (target_h, target_w):
            wrist = self._resize_image(wrist, target_h, target_w)
        images[wrist_key] = np.ascontiguousarray(wrist, dtype=np.uint8)

        return {
            "observation_id": f"so101_obs_{self.step_idx}",
            "images": images,
            "state": state,
            "step_index": self.step_idx,
            "runtime": "so101_isaacsim",
        }

    @staticmethod
    def _resize_image(img: "np.ndarray", target_h: int, target_w: int) -> "np.ndarray":
        """Resize an HxWx3 image to (target_h, target_w). Uses PIL if available."""
        try:
            from PIL import Image as _PILImage
            pil = _PILImage.fromarray(img)
            pil = pil.resize((target_w, target_h), _PILImage.BILINEAR)
            return np.asarray(pil, dtype=np.uint8)
        except Exception:
            # Fallback: center-crop to square then naive stride resize.
            h, w = img.shape[:2]
            side = min(h, w)
            y0 = (h - side) // 2
            x0 = (w - side) // 2
            img = img[y0:y0 + side, x0:x0 + side]
            ys = np.linspace(0, side - 1, target_h).astype(int)
            xs = np.linspace(0, side - 1, target_w).astype(int)
            return img[np.ix_(ys, xs)]


    def _base_status(self, *, reward: float, chunk_id: str | None = None) -> Dict[str, Any]:
        status = {
            "accepted": True,
            "buffered_steps": 0,
            "executed_steps": self.step_idx,
            "target_step_index": self.step_idx,
            "need_replan": not self.success,
            "safety_status": "ok",
            "success": bool(self.success),
            "done": bool(self.done),
            "reward": reward,
            "obs": self._last_obs,
            "so101": {"robot_id": self.config.get("robot_id", "")},
        }
        if chunk_id:
            status["chunk_id"] = chunk_id
        return status


# --------------------------------------------------------------------------
# RPC dispatch
# --------------------------------------------------------------------------

def _dispatch(runtime: So101IsaacSimRuntime, request: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
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
    raise TargetProtocolError(f"unsupported target RPC type: {rtype}")


# --------------------------------------------------------------------------
# TargetWS server plumbing
# --------------------------------------------------------------------------

class _PendingRpc:
    __slots__ = ("request", "response_queue")

    def __init__(self, request: Dict[str, Any], response_queue: queue.Queue):
        self.request = request
        self.response_queue = response_queue


class So101TargetWsServer:
    """Sync TargetWS server; Isaac Sim runs on the main thread."""

    def __init__(self, runtime: So101IsaacSimRuntime):
        self.runtime = runtime
        self._pending: queue.Queue[_PendingRpc] = queue.Queue()
        self._shutdown = threading.Event()

    def handle(self, websocket: Any) -> None:
        peer = getattr(websocket, "remote_address", None)
        logger.info("SO101 TargetWS client connected: %s", peer)
        try:
            for raw in websocket:
                if isinstance(raw, str):
                    err = make_response(
                        {"seq": 0},
                        "runtime.error",
                        {"error_code": "BAD_PAYLOAD", "message": "expected binary msgpack"},
                    )
                    websocket.send(err)
                    continue
                request = unpackb(raw)
                resp_q: queue.Queue = queue.Queue(maxsize=1)
                self._pending.put(_PendingRpc(request, resp_q))
                try:
                    response = resp_q.get(timeout=3600.0)
                except queue.Empty:
                    response = make_response(
                        request,
                        "runtime.error",
                        {"error_code": "TIMEOUT", "message": "sim request timed out"},
                    )
                websocket.send(response)
        except Exception as exc:
            if type(exc).__name__ != "ConnectionClosed":
                logger.info("SO101 TargetWS client disconnected: %s (%s)", peer, exc)

    def run_main_loop(self) -> None:
        while not self._shutdown.is_set():
            try:
                pending = self._pending.get(timeout=0.25)
            except queue.Empty:
                continue
            request = pending.request
            try:
                rtype, payload = _dispatch(self.runtime, request)
                pending.response_queue.put(make_response(request, rtype, payload))
            except Exception as exc:
                pending.response_queue.put(
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


def _run_targetws_server(server: So101TargetWsServer, host: str, port: int) -> None:
    from websockets.sync.server import serve as sync_serve

    with sync_serve(server.handle, host, port, max_size=None) as ws_server:
        ws_server.serve_forever()


# --------------------------------------------------------------------------
# Isaac Sim environment bootstrap
# --------------------------------------------------------------------------

def _build_env(task: str, headless: bool, device: str, enable_cameras: bool = True):
    """Create the LeIsaac SO101 environment.

    This runs inside the Isaac Sim python interpreter, so imports of
    isaacsim / isaaclab / gymnasium must happen here (not at module top).
    """
    from isaaclab.app import AppLauncher
    AppLauncher(headless=headless, enable_cameras=enable_cameras)
    import gymnasium as gym
    import leisaac  # noqa: F401  ¡ª registers LeIsaac-* envs
    from isaaclab_tasks.utils import parse_env_cfg
    from leisaac.utils.env_utils import get_task_type

    env_cfg = parse_env_cfg(task, device=device, num_envs=1)
    task_type = get_task_type(task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = int(time.time())
    env_cfg.episode_length_s = 60.0
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.recorders = None
    env = gym.make(task, cfg=env_cfg).unwrapped
    return env


def serve_blocking(host: str, port: int, config: dict[str, Any]) -> None:
    env = _build_env(
        task=config["task"],
        headless=bool(config.get("headless", True)),
        device=str(config.get("device", "cuda")),
    )
    runtime = So101IsaacSimRuntime(env, config)

    # One-shot startup warmup: keep stepping the sim with a zero action so the
    # GUI fully renders before we start serving RPCs. This is the ONLY place
    # we render outside of real RPC handlers — the main loop below never injects
    # extra steps, so action_chunk rollouts are not polluted by hold actions.
    warmup = int(config.get("initial_render_steps", 200))
    if warmup > 0:
        print(f"[so101_server] initial render warmup ({warmup} steps)...", flush=True)
        runtime.initial_render(warmup)
        print("[so101_server] initial render done", flush=True)

    server = So101TargetWsServer(runtime)
    ws_thread = threading.Thread(
        target=_run_targetws_server,
        args=(server, host, port),
        name="so101-targetws",
        daemon=True,
    )
    ws_thread.start()
    print(
        f"SO101 Isaac Sim TargetWS server listening on targetws://{host}:{port}",
        flush=True,
    )
    try:
        server.run_main_loop()
    except KeyboardInterrupt:
        server._shutdown.set()
        print("\n[so101_server] stopped", flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 Isaac Sim TargetWS server")
    parser.add_argument(
        "--task",
        default="LeIsaac-SO101-PickOrange-v0",
        help="LeIsaac task name (must be registered via the leisaac package)",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9002)
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run Isaac Sim headless (no GUI window)")
    parser.add_argument("--gui", action="store_true", default=False,
                        help="Explicitly enable GUI (overrides --headless)")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--step-hz", type=int, default=30,
                        help="Control rate for action_chunk inner loop (default 30)")
    parser.add_argument("--max-steps", type=int, default=3600,
                        help="Max simulation steps before forcing done=True (default 3600 = 120s@30Hz)")
    parser.add_argument("--max-chunk-size", type=int, default=32,
                        help="Max action chunk length accepted (default 32)")
    parser.add_argument("--initial-render-steps", type=int, default=200,
                        help="One-shot startup warmup steps to fully render the "
                             "GUI before serving RPC (default 200, set 0 to disable)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = dict(SO101_DEFAULT_CONFIG)
    config.update({
        "task": args.task,
        "headless": args.headless and not args.gui,
        "device": args.device,
        "step_hz": args.step_hz,
        "max_steps": args.max_steps,
        "max_chunk_size": args.max_chunk_size,
        "initial_render_steps": args.initial_render_steps,
    })

    serve_blocking(args.host, args.port, config)


if __name__ == "__main__":
    main()
