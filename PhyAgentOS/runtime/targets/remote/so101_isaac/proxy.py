"""SO-101 Isaac Sim remote target proxy."""

from __future__ import annotations

from typing import Any

import numpy as np

from PhyAgentOS.runtime.communication.target_ws_client import TargetWSClient
from PhyAgentOS.runtime.targets.remote.proxy import RemoteTargetProxy
from PhyAgentOS.runtime.watchdog.errors import TargetProtocolError


SO101_ISAAC_DEFAULT_CONFIG: dict[str, Any] = {
    "task": "LeIsaac-SO101-PickOrange-v0",
    "action_dim": 6,
    "max_chunk_size": 100,
    "max_steps": 280,
    "sim_warmup_steps": 30,
}


class SO101IsaacRemoteTargetProxy(RemoteTargetProxy):
    """Remote target proxy with SO-101 Isaac-specific defaults and action validation."""

    def __init__(self, client: TargetWSClient, *, config: dict[str, Any] | None = None):
        merged = {**SO101_ISAAC_DEFAULT_CONFIG, **dict(config or {})}
        super().__init__(client, config=merged)

    def configure_session(self, session_ctx: dict[str, Any]) -> dict[str, Any]:
        merged_ctx = self._with_so101_config(session_ctx)
        return super().configure_session(merged_ctx)

    def reset(self, session_ctx: dict[str, Any]) -> dict[str, Any]:
        merged_ctx = self._with_so101_config(session_ctx)
        return super().reset(merged_ctx)

    def action_chunk(self, executable_action_chunk: dict[str, Any]) -> dict[str, Any]:
        actions = np.asarray(executable_action_chunk.get("actions"), dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] != 6:
            raise TargetProtocolError(
                f"SO101 Isaac actions must have shape [T,6], got {actions.shape}"
            )
        if not np.isfinite(actions).all():
            raise TargetProtocolError("SO101 Isaac actions contain NaN or Inf")
        max_chunk_size = int(self.config.get("max_chunk_size", SO101_ISAAC_DEFAULT_CONFIG["max_chunk_size"]))
        if actions.shape[0] > max_chunk_size:
            raise TargetProtocolError(
                f"SO101 Isaac action chunk too large: {actions.shape[0]} > {max_chunk_size}"
            )
        updated = dict(executable_action_chunk)
        updated["actions"] = np.ascontiguousarray(actions, dtype=np.float32)
        return super().action_chunk(updated)

    def _with_so101_config(self, session_ctx: dict[str, Any]) -> dict[str, Any]:
        metadata = dict(session_ctx.get("metadata", {}) or {})
        so101_config = {
            **SO101_ISAAC_DEFAULT_CONFIG,
            **self.config,
            **metadata.get("so101_isaac", {}),
        }
        metadata["so101_isaac"] = so101_config
        return {
            **session_ctx,
            "metadata": metadata,
            "so101_isaac": so101_config,
        }
