"""Skill runtime for the leisaac-sgs IPU ACT policy backend.

Drives a single rollout session through the canonical PhyAgentOS policy loop:
  observe -> adapter.to_policy_input -> IPUACTPolicyClient.infer
          -> adapter.from_policy_output -> target_handle.action_chunk
until success / done / max_steps.

Differs from OpenPISkillRuntime only in policy client construction (gRPC
vs WebSocket) and in injecting IPU-specific defaults into session_ctx.
"""

from __future__ import annotations

from statistics import mean
from typing import Any
from urllib.parse import urlparse

from PhyAgentOS.runtime.adapters.factory import build_policy_adapter
from PhyAgentOS.runtime.policy.base_client import BasePolicyClient
from PhyAgentOS.runtime.schemas import AdapterPlan
from PhyAgentOS.runtime.sessions.models import (
    EnvironmentRequest,
    SkillContext,
    SkillRuntimeResult,
)
from PhyAgentOS.runtime.skillruntime.policy.base import PolicySkillRuntime
from PhyAgentOS.runtime.watchdog.errors import PolicyProtocolError


def build_ipu_act_client(endpoint: str, session_ctx: dict[str, Any]) -> BasePolicyClient:
    """Construct an IPUACTPolicyClient from session_ctx + skillruntime config."""
    from PhyAgentOS.runtime.policy.ipu_act.client import IPUACTPolicyClient

    config = session_ctx.get("ipu_act") or {}
    policy_endpoint = endpoint
    parsed = urlparse(policy_endpoint)
    if parsed.scheme != "grpc":
        raise PolicyProtocolError(
            f"IPU ACT skill runtime expects grpc:// endpoint, got {policy_endpoint}"
        )

    checkpoint = (
        config.get("policy_checkpoint_path")
        or session_ctx.get("policy_checkpoint_path")
        or session_ctx.get("pretrained_name_or_path")
    )
    if not checkpoint:
        raise PolicyProtocolError(
            "IPU ACT requires `policy_checkpoint_path` in skillruntime config "
            "(path to .img model file on the IPU server host)"
        )

    return IPUACTPolicyClient(
        endpoint=policy_endpoint,
        policy_type=str(config.get("policy_type", "act")),
        policy_checkpoint_path=str(checkpoint),
        actions_per_chunk=int(config.get("actions_per_chunk", 16)),
        camera_keys=tuple(config.get("camera_keys", (
            "observation.images.front",
            "observation.images.wrist",
        ))),
        proprio_key=str(config.get("proprio_key", "observation.state")),
        lerobot_features=config.get("lerobot_features"),
        timeout_s=float(config.get("timeout_s", 15.0)),
        device=str(config.get("device", "cpu")),
        rename_map=config.get("rename_map") or {},
    )


class IPUACTSkillRuntime(PolicySkillRuntime):
    """Policy skill runtime that talks to a leisaac IPU ACT gRPC server."""

    def __init__(self):
        self._snapshot: dict[str, Any] = {}

    def start(self, skill_ctx: SkillContext) -> None:
        self._snapshot = {"session_id": skill_ctx.session.session_id, "started": True}

    def cancel(self, skill_ctx: SkillContext, reason: str) -> None:
        self._snapshot = {**self._snapshot, "cancelled": True, "cancel_reason": reason}

    def snapshot(self, skill_ctx: SkillContext) -> dict[str, Any]:
        return dict(self._snapshot)

    def run_policy_loop(
        self,
        skill_ctx: SkillContext,
        target_handle,
        adapter_plan: AdapterPlan,
        policy_client: BasePolicyClient,
    ) -> SkillRuntimeResult:
        if not adapter_plan.policy_adapter:
            raise PolicyProtocolError("IPU ACT runtime requires a policy adapter")
        policy_adapter = build_policy_adapter(adapter_plan.policy_adapter)

        latencies: list[float] = []
        total_reward = 0.0
        session_ctx = skill_ctx.session.model_dump(mode="json")

        action_cfg = skill_ctx.target.config.get("action", {})
        action_dim = int(skill_ctx.target.config.get("action_dim", action_cfg.get("action_dim", 6)))
        session_ctx["action_dim"] = action_dim
        session_ctx["max_chunk_size"] = int(
            skill_ctx.target.config.get("max_chunk_size", action_cfg.get("chunk_size", 100))
        )
        # IPU ACT-specific defaults
        ipu_cfg = (skill_ctx.skillruntime.config or {}).get("ipu_act") or {}
        session_ctx.setdefault("ipu_act", ipu_cfg)
        session_ctx["policy_action_contract"] = self._policy_action_contract(skill_ctx, action_dim)

        for step_idx in range(skill_ctx.session.execution.max_steps):
            observation = target_handle.observe()
            if step_idx == 0 and skill_ctx.skillruntime.requires.environment_outputs:
                target_handle.request_environment_refresh(
                    EnvironmentRequest(
                        requested_outputs=list(skill_ctx.skillruntime.requires.environment_outputs)
                    )
                )
            session_ctx["source_observation_id"] = observation.data.get("observation_id")

            policy_input = policy_adapter.to_policy_input(observation.data, session_ctx)
            action_payload = policy_client.infer(policy_input)
            policy_meta = action_payload.get("policy_meta", {})
            if "policy_latency_ms" in policy_meta:
                latencies.append(float(policy_meta["policy_latency_ms"]))

            policy_action_chunk = policy_adapter.from_policy_output(action_payload, session_ctx)
            if policy_action_chunk.get("actions") is None:
                raise PolicyProtocolError("IPU ACT returned an empty action buffer")

            status = target_handle.action_chunk(policy_action_chunk)
            total_reward += float(status.get("reward", 0.0))
            self._snapshot = {
                "session_id": skill_ctx.session.session_id,
                "step_index": step_idx,
                "status": status,
            }
            if bool(status.get("success", False)) or bool(status.get("done", False)):
                return SkillRuntimeResult(
                    status="succeeded" if bool(status.get("success", False)) else "failed",
                    success=bool(status.get("success", False)),
                    final_status=status,
                    metadata={
                        "done": bool(status.get("done", False)),
                        "mean_policy_latency_ms": mean(latencies) if latencies else None,
                        "return_value": total_reward,
                    },
                )

        return SkillRuntimeResult(
            status="failed",
            success=False,
            final_status=target_handle.execution_status(),
            error_code="MAX_STEPS_EXCEEDED",
            error_message="session reached max_steps without success",
            metadata={
                "mean_policy_latency_ms": mean(latencies) if latencies else None,
                "return_value": total_reward,
            },
        )

    def _policy_action_contract(self, skill_ctx: SkillContext, action_dim: int) -> dict[str, Any]:
        action_contract = skill_ctx.skillruntime.output_contract.get("action", {})
        if action_contract:
            return action_contract
        return {
            "action_space_id": "so101_joint_pos_v1",
            "shape": ["T", action_dim],
            "dtype": "float32",
            "normalized": False,
            "representation": "joint_position_targets",
            "frame": "joint",
            "control_mode": "joint_position",
            "components": [
                {"name": "shoulder_pan", "unit": "rad"},
                {"name": "shoulder_lift", "unit": "rad"},
                {"name": "elbow_flex", "unit": "rad"},
                {"name": "wrist_flex", "unit": "rad"},
                {"name": "wrist_roll", "unit": "rad"},
                {"name": "gripper", "unit": "normalized"},
            ],
            "chunk": {"policy_hz": skill_ctx.target.config.get("control_hz", 30)},
        }
