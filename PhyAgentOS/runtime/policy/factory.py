"""Policy client factory and endpoint parsing."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from PhyAgentOS.runtime.policy.base_client import BasePolicyClient
from PhyAgentOS.runtime.policy.b1k_websocket_client import Behavior1kWebsocketPolicyClient
from PhyAgentOS.runtime.policy.dummy_client import DummyPolicyClient
from PhyAgentOS.runtime.policy.openpi.client import OpenPIClientPolicyWrapper
from PhyAgentOS.runtime.watchdog.errors import PolicyConnectionError


def parse_policy_endpoint(endpoint: str) -> tuple[str, str, int]:
    parsed = urlparse(endpoint)
    if parsed.scheme == "dummy" and parsed.netloc == "local":
        return ("dummy", "local", 0)
    if parsed.scheme == "openpi" and parsed.hostname and parsed.port:
        return ("openpi", parsed.hostname, int(parsed.port))
    if parsed.scheme in ("b1k-ws", "b1kws") and parsed.hostname and parsed.port:
        return ("b1k-ws", parsed.hostname, int(parsed.port))
    if parsed.scheme == "policyws" and parsed.hostname and parsed.port:
        return ("policyws", parsed.hostname, int(parsed.port))
    if parsed.scheme == "grpc" and parsed.hostname and parsed.port:
        return ("grpc", parsed.hostname, int(parsed.port))
    raise PolicyConnectionError(f"unsupported policy endpoint: {endpoint}")


def build_policy_client(
    endpoint: str,
    *,
    timeout_s: float = 5.0,
    action_dim: int = 7,
    chunk_size: int = 4,
    skillruntime_config: dict[str, Any] | None = None,
    session_ctx: dict[str, Any] | None = None,
) -> BasePolicyClient:
    kind, host, port = parse_policy_endpoint(endpoint)
    if kind == "dummy":
        return DummyPolicyClient(action_dim=action_dim, chunk_size=chunk_size)
    if kind in {"openpi", "policyws"}:
        return OpenPIClientPolicyWrapper(host=host, port=port, timeout_s=timeout_s)
    if kind == "b1k-ws":
        return Behavior1kWebsocketPolicyClient(host=host, port=port, timeout_s=timeout_s)
    if kind == "grpc":
        return _build_ipu_act_client(
            endpoint=endpoint,
            host=host,
            port=port,
            timeout_s=timeout_s,
            skillruntime_config=skillruntime_config,
            session_ctx=session_ctx,
        )
    raise PolicyConnectionError(f"unsupported policy endpoint kind: {kind}")


def _build_ipu_act_client(
    *,
    endpoint: str,
    host: str,
    port: int,
    timeout_s: float,
    skillruntime_config: dict[str, Any] | None,
    session_ctx: dict[str, Any] | None,
) -> BasePolicyClient:
    from PhyAgentOS.runtime.policy.ipu_act.client import IPUACTPolicyClient

    cfg = dict(skillruntime_config or {})

    if "ipu_act" in cfg and isinstance(cfg["ipu_act"], dict):
        cfg = {**cfg, **cfg["ipu_act"]}

    if session_ctx:
        for key in (
            "policy_checkpoint_path",
            "pretrained_name_or_path",
            "actions_per_chunk",
            "policy_type",
            "camera_keys",
            "proprio_key",
            "lerobot_features",
            "device",
            "rename_map",
        ):
            if key not in cfg and key in session_ctx:
                cfg[key] = session_ctx[key]

    checkpoint = cfg.get("policy_checkpoint_path") or cfg.get("pretrained_name_or_path")
    if not checkpoint:
        raise PolicyConnectionError(
            "grpc:// policy endpoint requires `policy_checkpoint_path` "
            "(set it in SKILLRUNTIME.md under config.ipu_act, or pass via session_ctx)"
        )

    camera_keys = cfg.get("camera_keys") or (
        "observation.images.front",
        "observation.images.wrist",
    )
    if isinstance(camera_keys, list):
        camera_keys = tuple(camera_keys)

    return IPUACTPolicyClient(
        endpoint=f"grpc://{host}:{port}",
        policy_type=str(cfg.get("policy_type", "act")),
        policy_checkpoint_path=str(checkpoint),
        actions_per_chunk=int(cfg.get("actions_per_chunk", 32)),
        camera_keys=tuple(camera_keys),
        proprio_key=str(cfg.get("proprio_key", "observation.state")),
        lerobot_features=cfg.get("lerobot_features"),
        timeout_s=float(cfg.get("timeout_s", timeout_s)),
        device=str(cfg.get("device", "cpu")),
        rename_map=cfg.get("rename_map") or {},
    )
