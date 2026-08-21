"""gRPC client for the leisaac IPU ACT policy server (lerobot_ipu/server.py).

Wire protocol: gRPC with pickled Python dataclasses (LeRobot async_inference).
Endpoint scheme: grpc://host:port

This client is **self-contained**: it does NOT import the `lerobot` package.
Instead it reconstructs the 4 message types (`Empty`, `Observation`, `Actions`,
`PolicySetup`) at runtime from an embedded FileDescriptorProto. This means the
PhyAgentOS host only needs `grpcio` + `protobuf` installed -- the lerobot stubs
(`lerobot.transport.services_pb2`) are NOT required.

For the pickled dataclasses (`TimedData`, `TimedAction`, `TimedObservation`,
`RemotePolicyConfig`, `FPSTracker`) we replicate the server-side pickle shim
mechanism: classes are renamed to the fully-qualified path
`lerobot.async_inference.helpers.<ClassName>` and a synthetic module is
registered under that path in sys.modules. This lets the IPU server
`pickle.loads()` our payloads without needing the PhyAgentOS package
to be importable on its host.
"""

from __future__ import annotations

import socket
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

import numpy as np

from PhyAgentOS.runtime.policy.base_client import BasePolicyClient
from PhyAgentOS.runtime.watchdog.errors import (
    PolicyConnectionError,
    PolicyProtocolError,
    PolicyTimeoutError,
)


# --- pickle-shim module path ------------------------------------------------
# Mirrors leisaac-sgs/lerobot_ipu/act.py:HELPERS_MODULE_PATH.
# The IPU server rewrites __module__ on its dataclasses to this path at
# startup (install_async_helpers_shim) so that pickle round-trips without
# requiring the lerobot package on every node. We do the same here so the
# server's pickle.loads() can reconstruct objects we send.

HELPERS_MODULE_PATH = "lerobot.async_inference.helpers"


# --- minimal mirror of lerobot_ipu/act.py dataclasses -----------------------
# Field names + order MUST match server side exactly (pickle reconstructs by
# position). See leisaac-sgs/lerobot_ipu/act.py lines 79-139.

@dataclass
class TimedData:
    timestamp: float
    timestep: int

    def get_timestamp(self) -> float:
        return self.timestamp

    def get_timestep(self) -> int:
        return self.timestep


@dataclass
class TimedAction(TimedData):
    action: Any

    def get_action(self) -> Any:
        return self.action


@dataclass
class TimedObservation(TimedData):
    observation: Any
    must_go: bool = False

    def get_observation(self) -> Any:
        return self.observation


@dataclass
class RemotePolicyConfig:
    policy_type: str
    lerobot_features: dict
    actions_per_chunk: int
    pretrained_name_or_path: str | None = None
    policy_checkpoint_path: str | None = None
    device: str = "cpu"
    rename_map: dict = field(default_factory=dict)


@dataclass
class FPSTracker:
    """Mirror of lerobot_ipu/act.py:FPSTracker (not used, but shimmed)."""
    target_fps: float
    first_timestamp: float | None = None
    total_obs_count: int = 0


def _install_helpers_shim() -> None:
    """Register our dataclasses under HELPERS_MODULE_PATH in sys.modules.

    We rewrite __module__ on each class to "lerobot.async_inference.helpers"
    and install a synthetic module (plus its parent packages) in sys.modules.
    Pickle uses sys.modules for lookup, so the fake module must be present
    there -- otherwise pickle.dumps() raises "import of module ... failed".

    Idempotent: safe to call multiple times.
    """
    pkg_root = "lerobot"
    pkg_async = "lerobot.async_inference"
    helpers_path = HELPERS_MODULE_PATH

    # Register parent packages (lerobot, lerobot.async_inference) as fake
    # namespace modules if they don't exist yet.
    for name in (pkg_root, pkg_async):
        if name not in sys.modules:
            m = types.ModuleType(name)
            m.__path__ = []  # mark as package
            sys.modules[name] = m

    # Install (or refresh) the helpers module.
    helpers_module = types.ModuleType(helpers_path)
    helpers_module.__package__ = pkg_async
    for cls in (TimedData, TimedAction, TimedObservation, RemotePolicyConfig, FPSTracker):
        cls.__module__ = helpers_path
        setattr(helpers_module, cls.__name__, cls)
    sys.modules[helpers_path] = helpers_module

    # Make parent packages expose the helpers attribute too (some picklers
    # walk the package tree via getattr rather than sys.modules).
    setattr(sys.modules[pkg_async], "helpers", helpers_module)


# Install at import time so any subsequent pickle.dumps() on our dataclasses
# emits the helpers-module qualname.
_install_helpers_shim()


# --- embedded protobuf schema ------------------------------------------------
# Serialized FileDescriptorProto for transport/services.proto. Mirrors the
# schema used by lerobot 0.4.2 (see leisaac-sgs/modify.patch). We embed the
# descriptor so this client does not depend on the lerobot package.

from google.protobuf import descriptor_pb2 as _descriptor_pb2
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message_factory as _message_factory


def _build_file_descriptor_proto() -> _descriptor_pb2.FileDescriptorProto:
    fd = _descriptor_pb2.FileDescriptorProto()
    fd.name = "services.proto"
    fd.package = "transport"
    fd.syntax = "proto3"

    # Enum: TransferState
    e = fd.enum_type.add()
    e.name = "TransferState"
    for nm, val in (
        ("TRANSFER_UNKNOWN", 0),
        ("TRANSFER_BEGIN", 1),
        ("TRANSFER_MIDDLE", 2),
        ("TRANSFER_END", 3),
    ):
        v = e.value.add()
        v.name = nm
        v.number = val

    # Message: Empty
    fd.message_type.add().name = "Empty"

    # Message: Observation { transfer_state=TransferState; bytes data=2; }
    m = fd.message_type.add()
    m.name = "Observation"
    f1 = m.field.add()
    f1.name = "transfer_state"
    f1.number = 1
    f1.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    f1.type = _descriptor_pb2.FieldDescriptorProto.TYPE_ENUM
    f1.type_name = ".transport.TransferState"
    f2 = m.field.add()
    f2.name = "data"
    f2.number = 2
    f2.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    f2.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    # Message: Actions { bytes data=1; }
    m = fd.message_type.add()
    m.name = "Actions"
    f = m.field.add()
    f.name = "data"
    f.number = 1
    f.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    f.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    # Message: PolicySetup { bytes data=1; }
    m = fd.message_type.add()
    m.name = "PolicySetup"
    f = m.field.add()
    f.name = "data"
    f.number = 1
    f.label = _descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    f.type = _descriptor_pb2.FieldDescriptorProto.TYPE_BYTES

    # Service: AsyncInference
    svc = fd.service.add()
    svc.name = "AsyncInference"

    for method_name, in_type, out_type, client_stream in (
        ("Ready", "Empty", "Empty", False),
        ("SendPolicyInstructions", "PolicySetup", "Empty", False),
        ("SendObservations", "Observation", "Empty", True),
        ("GetActions", "Empty", "Actions", False),
    ):
        mth = svc.method.add()
        mth.name = method_name
        mth.input_type = f".transport.{in_type}"
        mth.output_type = f".transport.{out_type}"
        mth.client_streaming = client_stream
        mth.server_streaming = False

    return fd


_POOL = _descriptor_pool.DescriptorPool()
_FD_PROTO = _build_file_descriptor_proto()
_POOL.Add(_FD_PROTO)
_FILE_DESC = _POOL.FindFileByName("services.proto")

Empty = _message_factory.GetMessageClass(_FILE_DESC.message_types_by_name["Empty"])
Observation = _message_factory.GetMessageClass(_FILE_DESC.message_types_by_name["Observation"])
Actions = _message_factory.GetMessageClass(_FILE_DESC.message_types_by_name["Actions"])
PolicySetup = _message_factory.GetMessageClass(_FILE_DESC.message_types_by_name["PolicySetup"])

# TransferState enum values (mirror of services_pb2)
TRANSFER_UNKNOWN = 0
TRANSFER_BEGIN = 1
TRANSFER_MIDDLE = 2
TRANSFER_END = 3


# --- chunked byte transfer (matches receive_bytes_in_chunks) -----------------

_CHUNK_BYTES = 4 * 1024 * 1024  # 4 MiB per Observation chunk; mirrors upstream


def _build_observation_chunks(pickle_bytes: bytes) -> list:
    """Split pickled payload into chunks of Observation messages.

    First chunk gets TRANSFER_BEGIN, last gets TRANSFER_END, middle ones
    TRANSFER_MIDDLE. Single-chunk payload gets TRANSFER_END only.
    Mirrors lerobot.transport.utils.receive_bytes_in_chunks semantics.
    """
    chunks = []
    total = len(pickle_bytes)
    if total == 0:
        chunks.append(Observation(transfer_state=TRANSFER_END, data=b""))
        return chunks
    offset = 0
    while offset < total:
        end = min(offset + _CHUNK_BYTES, total)
        is_first = offset == 0
        is_last = end == total
        if is_first and is_last:
            state = TRANSFER_END
        elif is_first:
            state = TRANSFER_BEGIN
        elif is_last:
            state = TRANSFER_END
        else:
            state = TRANSFER_MIDDLE
        chunks.append(Observation(transfer_state=state, data=pickle_bytes[offset:end]))
        offset = end
    return chunks


# --- client class ------------------------------------------------------------

# Fully qualified RPC method names (as expected by gRPC wire protocol)
_RPC_READY = "/transport.AsyncInference/Ready"
_RPC_SEND_POLICY = "/transport.AsyncInference/SendPolicyInstructions"
_RPC_SEND_OBS = "/transport.AsyncInference/SendObservations"
_RPC_GET_ACTIONS = "/transport.AsyncInference/GetActions"


class IPUACTPolicyClient(BasePolicyClient):
    """gRPC client for the IPU ACT policy server (lerobot_ipu/server.py).

    Lifecycle:
      1. __init__: open gRPC channel, call Ready() to reset server queues.
      2. infer(): on first call, send RemotePolicyConfig via
         SendPolicyInstructions (lazy setup using fields from the adapter /
         session_ctx).
      3. Each infer(): pickle TimedObservation, stream chunks via
         SendObservations, then GetActions -> unpickle list[TimedAction].

    Args:
      endpoint: grpc://host:port
      policy_type: typically "act"
      policy_checkpoint_path: path to .img model file on the IPU server host
      actions_per_chunk: default 16, clamped to <=100 by server (chunk_size)
      camera_keys: which keys in observation to send as lerobot image features
      proprio_key: which key holds the joint state vector
      timeout_s: per-RPC timeout (default 15s, matches policy_timeout_ms=15000)
    """

    def __init__(
        self,
        endpoint: str,
        *,
        policy_type: str = "act",
        policy_checkpoint_path: str | None = None,
        actions_per_chunk: int = 16,
        camera_keys: tuple[str, ...] = ("observation.images.front", "observation.images.wrist"),
        proprio_key: str = "observation.state",
        lerobot_features: dict[str, dict] | None = None,
        timeout_s: float = 15.0,
        device: str = "cpu",
        rename_map: dict[str, str] | None = None,
    ):
        parsed = urlparse(endpoint)
        if parsed.scheme != "grpc":
            raise PolicyConnectionError(
                f"IPUACTPolicyClient expects grpc:// endpoint, got {endpoint}"
            )
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 8080
        self.timeout_s = float(timeout_s)

        self.policy_type = policy_type
        self.policy_checkpoint_path = policy_checkpoint_path
        self.actions_per_chunk = int(actions_per_chunk)
        self.camera_keys = tuple(camera_keys)
        self.proprio_key = proprio_key
        self.rename_map = dict(rename_map or {})
        self.device = device

        # Default lerobot_features mirrors lerobot_ipu/act.py + SO-101 joints.
        self.lerobot_features = lerobot_features or {
            "observation.images.front": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.images.wrist": {"dtype": "image", "shape": [480, 640, 3]},
            "observation.state": {"shape": [6], "dtype": "float64"},
        }

        self._channel = None
        self._timestep = 0
        self._policy_loaded = False
        self._connect_and_ready()

    # -- connection --

    def _connect_and_ready(self) -> None:
        import grpc

        try:
            self._channel = grpc.insecure_channel(
                f"{self.host}:{self.port}",
                options=[
                    ("grpc.max_send_message_length", 64 * 1024 * 1024),
                    ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ],
            )
            # Ready() handshake resets server queues (matches server.py:Ready).
            ready_rpc = self._channel.unary_unary(
                _RPC_READY,
                request_serializer=Empty.SerializeToString,
                response_deserializer=Empty.FromString,
            )
            ready_rpc(Empty(), timeout=self.timeout_s)
        except (socket.timeout, TimeoutError) as exc:
            raise PolicyTimeoutError(
                f"timed out connecting to IPU ACT gRPC {self.host}:{self.port}"
            ) from exc
        except Exception as exc:
            raise PolicyConnectionError(
                f"failed to connect / Ready() IPU ACT gRPC {self.host}:{self.port}: {exc}"
            ) from exc

    def close(self) -> None:
        self._policy_loaded = False
        if self._channel is not None:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None

    # -- setup (one-shot, lazy on first infer) --

    def _ensure_policy_loaded(self) -> None:
        if self._policy_loaded:
            return
        import pickle

        if not self.policy_checkpoint_path:
            raise PolicyProtocolError(
                "IPUACTPolicyClient requires policy_checkpoint_path "
                "(path to .img on the IPU server host)"
            )

        cfg = RemotePolicyConfig(
            policy_type=self.policy_type,
            lerobot_features=self.lerobot_features,
            actions_per_chunk=self.actions_per_chunk,
            pretrained_name_or_path=self.policy_checkpoint_path,
            policy_checkpoint_path=self.policy_checkpoint_path,
            device=self.device,
            rename_map=self.rename_map,
        )
        setup_msg = PolicySetup(data=pickle.dumps(cfg))
        rpc = self._channel.unary_unary(
            _RPC_SEND_POLICY,
            request_serializer=PolicySetup.SerializeToString,
            response_deserializer=Empty.FromString,
        )
        rpc(setup_msg, timeout=self.timeout_s)
        self._policy_loaded = True

    # -- inference --

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        """Send observation, receive action chunk.

        Expected input format (after ACTPolicyAdapter.to_policy_input):
          {
            "observation.state":         np.float32 [6],
            "observation.images.front":  np.uint8 [H, W, 3]  HWC,
            "observation.images.wrist":  np.uint8 [H, W, 3]  HWC,
            "task_description":          str,
          }

        Output:
          {"actions": np.float32 [T, 6], "policy_meta": {"policy_latency_ms": float}}
        """
        import pickle

        self._ensure_policy_loaded()
        started = time.perf_counter()

        # Build a flat obs dict in leisaac convention:
        # joint_pos keys are flattened to "{joint_name}.pos".
        flat_obs: dict[str, Any] = {}
        for cam_key in self.camera_keys:
            if cam_key in observation:
                flat_obs[cam_key] = self._ensure_uint8_hwc(observation[cam_key], cam_key)
        state = None
        if self.proprio_key in observation:
            state = np.asarray(observation[self.proprio_key], dtype=np.float32).reshape(-1)
            flat_obs[self.proprio_key] = state
        # Also expose per-joint "{name}.pos" keys, which is what
        # lerobot_ipu/act.py:extract_so101_state expects.
        if state is not None and state.shape[0] == 6:
            for jname, jval in zip(
                ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"),
                state,
            ):
                flat_obs[f"{jname}.pos"] = float(jval)
        flat_obs["task"] = str(observation.get("task_description", ""))

        timed = TimedObservation(
            timestamp=time.time(),
            timestep=self._timestep,
            observation=flat_obs,
            must_go=True,  # bypass server-side dedup on critical first call
        )
        self._timestep += 1

        payload_bytes = pickle.dumps(timed)
        chunks = _build_observation_chunks(payload_bytes)

        try:
            # SendObservations is stream-unary.
            send_obs_rpc = self._channel.stream_unary(
                _RPC_SEND_OBS,
                request_serializer=Observation.SerializeToString,
                response_deserializer=Empty.FromString,
            )
            send_obs_rpc(iter(chunks), timeout=self.timeout_s)
            get_actions_rpc = self._channel.unary_unary(
                _RPC_GET_ACTIONS,
                request_serializer=Empty.SerializeToString,
                response_deserializer=Actions.FromString,
            )
            actions_msg = get_actions_rpc(Empty(), timeout=self.timeout_s)
        except (socket.timeout, TimeoutError) as exc:
            raise PolicyTimeoutError(
                f"IPU ACT gRPC inference timed out after {self.timeout_s:.3f}s"
            ) from exc
        except Exception as exc:
            raise PolicyConnectionError(f"IPU ACT gRPC inference failed: {exc}") from exc

        try:
            timed_actions = pickle.loads(actions_msg.data)
        except Exception as exc:
            raise PolicyProtocolError(f"failed to unpickle IPU ACT actions: {exc}") from exc

        if not isinstance(timed_actions, list) or not timed_actions:
            raise PolicyProtocolError("IPU ACT returned empty action list")

        # timed_actions: list[TimedAction] where action is torch.Tensor [1, action_dim].
        actions_list = []
        for ta in timed_actions:
            act = ta.get_action() if hasattr(ta, "get_action") else ta.action
            if hasattr(act, "detach"):
                act = act.detach().cpu().numpy()
            act = np.asarray(act, dtype=np.float32)
            if act.ndim == 2:
                act = act[0]  # squeeze batch dim
            actions_list.append(act)
        actions = np.stack(actions_list, axis=0).astype(np.float32, copy=False)

        latency_ms = (time.perf_counter() - started) * 1000.0
        return {
            "actions": actions,  # [T, 6]
            "policy_meta": {
                "policy_latency_ms": latency_ms,
                "server_timestep_end": int(timed_actions[-1].timestep),
                "policy_type": self.policy_type,
            },
        }

    @staticmethod
    def _ensure_uint8_hwc(img: Any, name: str) -> np.ndarray:
        arr = np.asarray(img)
        if arr.ndim != 3:
            raise PolicyProtocolError(
                f"IPU ACT `{name}` image must be rank-3, got {arr.shape}"
            )
        # CHW -> HWC if leading axis looks like channels.
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        arr = np.ascontiguousarray(arr)
        if arr.shape[-1] == 4:
            arr = arr[:, :, :3]
        if np.issubdtype(arr.dtype, np.floating):
            max_val = float(arr.max()) if arr.size else 0.0
            if max_val <= 1.0:
                arr = np.clip(arr * 255.0, 0, 255)
            arr = arr.astype(np.uint8)
        else:
            arr = arr.astype(np.uint8, copy=False)
        return arr
