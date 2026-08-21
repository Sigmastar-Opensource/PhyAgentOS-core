"""Runtime skill registry schemas."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SkillRequirements(BaseModel):
    sensors: list[str] = Field(default_factory=list)
    environment_outputs: list[str] = Field(default_factory=list)
    object_queries_from_task: bool = False
    min_confidence: float | None = None
    require_metric_geometry: bool = False
    strict_environment_contract: bool = True


class SkillPolicySpec(BaseModel):
    policy_client: str
    policy_adapter: str
    supports_chunk: bool = False


class SkillObservationContract(BaseModel):
    observation_type: Literal[
        "empty",
        "structured",
        "visual",
        "proprioceptive",
        "multimodal",
        "environment_only",
    ] = "multimodal"
    empty_observation_allowed: bool = False
    empty_observation_semantics: str | None = None


class TargetToolPolicy(BaseModel):
    expose: list[str] = Field(default_factory=list)
    forbidden: list[str] = Field(default_factory=list)
    require_tool_schema_validation: bool = True
    require_action_validation: bool = True
    require_target_side_validation: bool = True
    require_operator_override_for_real_robot: bool = False
    allow_reset_by_agent: bool = False
    allow_close_by_agent: bool = False


class SkillBenchmarkCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    benchmark_id: str
    execution_mode: Literal["policy_loop", "target_native"]
    target_interface: Literal["rollout_episode_v1", "target_benchmark_job_v1"]
    result_schema: Literal["benchmark_execution_result_v1"] = "benchmark_execution_result_v1"
    reset_owner: Literal["session_runner", "skillruntime"]


class SkillRuntimeSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    runtime: str
    runtime_kind: Literal["policy", "builtin"]
    loop_mode: str
    agent_exposure: Literal["none", "target_tools", "constrained_target_tools"] = "none"
    supported_target_kinds: list[Literal["game", "debug", "simulation", "real_robot"]]
    policy: SkillPolicySpec | None = None
    observation_contract: SkillObservationContract = Field(default_factory=SkillObservationContract)
    target_tool_policy: TargetToolPolicy | None = None
    supports_chunk: bool = False
    default_replan_every: int = 1
    input_contract: dict[str, Any] = Field(default_factory=dict)
    output_contract: dict[str, Any] = Field(default_factory=dict)
    adapter_requirements: dict[str, Any] = Field(default_factory=dict)
    requires: SkillRequirements = Field(default_factory=SkillRequirements)
    config: dict[str, Any] = Field(default_factory=dict)
    benchmark: SkillBenchmarkCapability | None = None


class SkillRuntimeDocument(BaseModel):
    version: Literal["runtime_skill_registry_v1"] = "runtime_skill_registry_v1"
    skillruntimes: list[SkillRuntimeSpec] = Field(default_factory=list)
