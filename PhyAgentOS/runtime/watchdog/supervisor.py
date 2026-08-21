"""Runtime v2 watchdog supervisor."""

from __future__ import annotations

import socket
from pathlib import Path

from pydantic import ValidationError

from PhyAgentOS.runtime.perception import PerceptionRuntime
from PhyAgentOS.runtime.policy.factory import build_policy_client
from PhyAgentOS.runtime.preflight import RuntimeCompatibilityPreflight
from PhyAgentOS.runtime.schemas import SessionsDocument, SkillRuntimeDocument, TargetsDocument
from PhyAgentOS.runtime.schemas.result import SessionResult
from PhyAgentOS.runtime.sessions.session_runner import SessionRunner
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
from PhyAgentOS.runtime.state_io.workspace_paths import RuntimeWorkspacePaths
from PhyAgentOS.runtime.watchdog.errors import SchemaValidationError
from PhyAgentOS.runtime.watchdog.failure import FailureEscalator
from PhyAgentOS.runtime.watchdog.health import HealthMonitor
from PhyAgentOS.runtime.watchdog.registry import SessionRegistry
from PhyAgentOS.runtime.watchdog.result_writer import ResultWriter
from PhyAgentOS.runtime.watchdog.runner_thread import RunnerThreadHandle
from PhyAgentOS.runtime.watchdog.runtime_registry import SkillRuntimeRegistry, TargetRuntimeRegistry
from PhyAgentOS.runtime.watchdog.scheduler import SessionScheduleError, SessionScheduler
from PhyAgentOS.runtime.watchdog.watcher import WorkspaceWatcher


class WatchdogSupervisor:
    """Claim and execute runtime sessions from a workspace."""

    def __init__(
        self,
        workspace: str | Path,
        worker_id: str | None = None,
        environment_workspace: str | Path | None = None,
        verification_enabled: bool = False,
        verification_settings: dict | None = None,
    ):
        self.paths = RuntimeWorkspacePaths.from_path(workspace)
        self.workspace = self.paths.workspace
        self.environment_workspace = (
            Path(environment_workspace).expanduser() if environment_workspace is not None else self.workspace
        )
        self.worker_id = worker_id or f"runtime-watchdog@{socket.gethostname()}"
        self.verification_enabled = verification_enabled
        self.verification_settings = dict(verification_settings or {})
        self.registry = SessionRegistry(self.paths.sessions)
        self.result_writer = ResultWriter(self.workspace)
        self.watcher = WorkspaceWatcher(self.paths)
        self.scheduler = SessionScheduler()
        self.target_registry = TargetRuntimeRegistry()
        self.skill_registry = SkillRuntimeRegistry()
        self.perception_runtime = PerceptionRuntime(self.workspace, self.environment_workspace)
        self.health_monitor = HealthMonitor()
        self.failure_escalator = FailureEscalator()
        self.preflight = RuntimeCompatibilityPreflight(self.workspace, self.skill_registry)

    def run_once(self, *, session_id: str | None = None) -> bool:
        if session_id:
            self.registry.prepare_for_run(session_id)
            print(f"[runtime-watchdog] prepared {session_id} -> pending", flush=True)
        sessions_doc, targets_doc, skillruntimes_doc = self._load_runtime_documents()
        try:
            scheduled = self.scheduler.select_next(
                sessions_doc, targets_doc, skillruntimes_doc, session_id=session_id
            )
        except SessionScheduleError as exc:
            self.failure_escalator.handle(exc.session_id, exc, self.registry)
            return True
        if scheduled is None:
            #print("[runtime-watchdog] no pending session (check SESSIONS.md status=pending)", flush=True)
            return False
        session_id = scheduled.session.session_id
        if not self.registry.try_claim(session_id, self.worker_id):
            print(f"[runtime-watchdog] could not claim {session_id}", flush=True)
            return False
        print(f"[runtime-watchdog] running {session_id} ...", flush=True)

        try:
            session = self.registry.get_session(session_id)
            _, targets_doc, skillruntimes_doc = self._load_runtime_documents()
            scheduled = self.scheduler.resolve_session(session, targets_doc, skillruntimes_doc)
            self.registry.mark_preflight_checking(session_id)
            session = self.registry.get_session(session_id)
            scheduled = self.scheduler.resolve_session(session, targets_doc, skillruntimes_doc)
            health_report = self.health_monitor.preflight(scheduled)
            if not health_report.ok:
                raise SchemaValidationError(health_report.summary())

            perception_plan = self.perception_runtime.resolve_and_check(scheduled)
            preflight_result = self.preflight.check(scheduled, perception_plan)
            if preflight_result.verdict == "rejected":
                result = SessionResult(
                    status="rejected",
                    success=False,
                    error_code="RUNTIME_PREFLIGHT_FAILED",
                    error_message=self._preflight_error_message(preflight_result),
                    metadata={"preflight": preflight_result.model_dump(mode="json", exclude_none=True)},
                )
                self.result_writer.write_lesson(
                    session,
                    scheduled.target_spec.id,
                    scheduled.skillruntime_id,
                    "preflight_checking",
                    result.error_code,
                    result.error_message or "runtime compatibility preflight failed",
                    preflight_result.model_dump(mode="json", exclude_none=True),
                )
                self.registry.mark_rejected(session_id, result)
                return True

            if session.verification_profile != "strict":
                self._wait_for_verification_service(session, scheduled.target_spec)

            self._write_preflight_metadata(session_id, preflight_result)

            self.registry.mark_running(session_id)
            session = self.registry.get_session(session_id)
            scheduled = self.scheduler.resolve_session(session, targets_doc, skillruntimes_doc)
            target_endpoint = session.routing.target_endpoint or scheduled.target_spec.runtime.target_endpoint
            target = self.target_registry.build(scheduled.target_spec, target_endpoint=target_endpoint)
            policy_client = None
            runner = None
            cleanup_in_background = False
            cleanup_error: Exception | None = None
            try:
                if scheduled.skillruntime_spec.runtime_kind == "policy":
                    policy_client = self._build_policy_client(
                        session, scheduled.target_spec, scheduled.skillruntime_spec
                    )
                runtime = self.skill_registry.build(scheduled.skillruntime_spec.runtime)
                runner = SessionRunner(
                    session=session,
                    target_spec=scheduled.target_spec,
                    skillruntime_spec=scheduled.skillruntime_spec,
                    adapter_plan=preflight_result.adapter_plan,
                    target=target,
                    skill_runtime=runtime,
                    policy_client=policy_client,
                    perception_runtime=self.perception_runtime,
                    perception_plan=perception_plan,
                    target_tool_manifest=preflight_result.target_tool_manifest,
                    verification_settings=self.verification_settings,
                )
                thread_handle = RunnerThreadHandle(runner)
                thread_handle.start()
                while not thread_handle.done:
                    thread_handle.snapshot()
                    if thread_handle.elapsed_s() >= float(session.timeouts.execute_timeout_s):
                        result = SessionResult(
                            status="timed_out",
                            success=False,
                            error_code="EXECUTION_TIMEOUT",
                            error_message=(
                                "session exceeded execute_timeout_s="
                                f"{session.timeouts.execute_timeout_s}"
                            ),
                            metadata={
                                "timeout_s": session.timeouts.execute_timeout_s,
                                "runner_snapshot": thread_handle.snapshot(),
                                "cleanup": "best_effort_threaded",
                            },
                        )
                        thread_handle.request_cancel_and_close("execution timeout")
                        cleanup_in_background = True
                        self._close_policy_client(policy_client)
                        self.registry.mark_finalizing(session_id)
                        result = self.result_writer.write_episode(
                            session,
                            scheduled.target_spec,
                            scheduled.skillruntime_id,
                            result,
                        )
                        self.result_writer.write_session_history(session, scheduled.target_spec, result)
                        self.registry.mark_timed_out(session_id, result)
                        return True
                    self._sleep_runner_poll_interval(session)
                if thread_handle.exception is not None:
                    raise thread_handle.exception
                result = thread_handle.result
                if result is None:
                    raise RuntimeError("runner thread finished without a result")
            finally:
                self._close_policy_client(policy_client)
                if cleanup_in_background:
                    pass
                elif runner is not None:
                    cleanup_error = self._close_runner_best_effort(runner)
                else:
                    cleanup_error = self._close_target_best_effort(target)

            if cleanup_error is not None:
                cleanup = result.metadata.get("cleanup")
                if not isinstance(cleanup, dict):
                    cleanup = {}
                cleanup.update(
                    {
                        "close_error_code": type(cleanup_error).__name__,
                        "close_error_message": str(cleanup_error),
                    }
                )
                result.metadata["cleanup"] = cleanup
            self.registry.mark_finalizing(session_id)
            result = self.result_writer.write_episode(
                session,
                scheduled.target_spec,
                scheduled.skillruntime_id,
                result,
            )
            self.result_writer.write_session_history(session, scheduled.target_spec, result)
            verification_profile = session.verification_profile
            session_verification = (
                self.verification_enabled
                and verification_profile in {"audit", "recovery"}
                and (session.benchmark is None or session.benchmark.execution_mode == "policy_loop")
            )
            if session_verification:
                self.result_writer.write_verification_bundle(
                    session,
                    scheduled.target_spec,
                    scheduled.skillruntime_id,
                    result,
                    environment_workspace=self.environment_workspace,
                    initial_observation=runner.initial_observation if runner is not None else None,
                    final_observation=runner.final_observation if runner is not None else None,
                )
                self.registry.mark_awaiting_verification(session_id, result)
            else:
                self.registry.mark_finished(session_id, result)
            return True
        except Exception as exc:
            self._write_execution_failure_lesson(session_id, exc)
            self.failure_escalator.handle(session_id, exc, self.registry)
            return True

    def _load_runtime_documents(self) -> tuple[SessionsDocument, TargetsDocument, SkillRuntimeDocument]:
        try:
            sessions_doc = SessionsDocument.model_validate(read_yaml_block(self.paths.sessions))
            targets_doc = TargetsDocument.model_validate(read_yaml_block(self.paths.targets))
            skillruntimes_doc = SkillRuntimeDocument.model_validate(read_yaml_block(self.paths.skillruntimes))
            return sessions_doc, targets_doc, skillruntimes_doc
        except ValidationError as exc:
            message = str(exc)
            if "benchmark" in message or "verification_profile" in message:
                message = "UNSUPPORTED_BENCHMARK_SCHEMA: " + message
            raise SchemaValidationError(message) from exc

    def _load_registries(self) -> tuple[TargetsDocument, SkillRuntimeDocument]:
        try:
            targets_doc = TargetsDocument.model_validate(read_yaml_block(self.paths.targets))
            skillruntimes_doc = SkillRuntimeDocument.model_validate(read_yaml_block(self.paths.skillruntimes))
            return targets_doc, skillruntimes_doc
        except ValidationError as exc:
            message = str(exc)
            if "benchmark" in message:
                message = "UNSUPPORTED_BENCHMARK_SCHEMA: " + message
            raise SchemaValidationError(message) from exc

    def _build_policy_client(self, session, target_spec, skillruntime_spec=None):
        action_cfg = target_spec.config.get("action", {})
        action_dim = target_spec.config.get("action_dim", action_cfg.get("action_dim", 7))
        chunk_size = target_spec.config.get("chunk_size", action_cfg.get("chunk_size", 4))
        skillruntime_config = None
        if skillruntime_spec is not None:
            skillruntime_config = getattr(skillruntime_spec, "config", None) or {}
        return build_policy_client(
            session.routing.policy_endpoint or "dummy://local",
            timeout_s=session.timeouts.policy_timeout_s,
            action_dim=int(action_dim),
            chunk_size=int(chunk_size),
            skillruntime_config=skillruntime_config,
        )

    def _close_policy_client(self, policy_client) -> None:
        if policy_client is None:
            return
        try:
            policy_client.close()
        except Exception:
            pass

    def _close_runner_best_effort(self, runner: SessionRunner) -> Exception | None:
        try:
            runner.close()
        except Exception as exc:  # noqa: BLE001
            print(
                "[runtime-watchdog] warning: runner cleanup close failed for "
                f"{runner.session.session_id}: {type(exc).__name__}: {exc}",
                flush=True,
            )
            return exc
        return None

    def _close_target_best_effort(self, target) -> Exception | None:
        try:
            target.close()
        except Exception as exc:  # noqa: BLE001
            print(
                "[runtime-watchdog] warning: target cleanup close failed: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            return exc
        return None

    def _sleep_runner_poll_interval(self, session) -> None:
        import time

        timeout_s = float(session.timeouts.execute_timeout_s)
        time.sleep(max(0.01, min(0.05, timeout_s / 10.0)))

    def _wait_for_verification_service(self, session, target_spec) -> None:
        import time
        from urllib import error, request
        from urllib.parse import urlparse

        service_url = str(self.verification_settings.get("local_service_url") or "").rstrip("/")
        if not self.verification_enabled or not service_url:
            raise SchemaValidationError("VERIFICATION_SERVICE_UNAVAILABLE: global verification service is disabled")
        target_endpoint = session.routing.target_endpoint or target_spec.runtime.target_endpoint
        target_host = urlparse(str(target_endpoint or "").replace("targetws://", "ws://", 1)).hostname
        if (
            session.benchmark is not None
            and session.benchmark.execution_mode == "target_native"
            and target_spec.target_class == "remote"
            and target_host not in {None, "localhost", "127.0.0.1", "::1"}
            and not self.verification_settings.get("remote_target_url")
        ):
            raise SchemaValidationError("VERIFICATION_REMOTE_URL_REQUIRED: configure agents.verification.remoteTargetUrl")
        opener = request.build_opener(request.ProxyHandler({}))
        deadline = time.monotonic() + float(session.timeouts.preflight_timeout_s)
        last_error = "not ready"
        while time.monotonic() < deadline:
            try:
                with opener.open(service_url + "/healthz", timeout=0.5) as response:  # noqa: S310
                    if response.status == 200:
                        return
            except (OSError, TimeoutError, error.URLError) as exc:
                last_error = str(exc)
            time.sleep(0.05)
        raise SchemaValidationError("VERIFICATION_SERVICE_UNAVAILABLE: " + last_error)

    def _preflight_error_message(self, preflight_result) -> str:
        if not preflight_result.missing_items:
            return "runtime compatibility preflight failed"
        return "; ".join(
            f"{item.code}: {item.field} expected {item.expected}"
            + (f", found {item.found}" if item.found is not None else "")
            for item in preflight_result.missing_items
        )

    def _write_preflight_metadata(self, session_id: str, preflight_result) -> None:
        document = self.registry.load()
        for session in document.sessions:
            if session.session_id != session_id:
                continue
            metadata = dict(session.result.metadata)
            metadata["preflight"] = preflight_result.model_dump(mode="json", exclude_none=True)
            session.result.metadata = metadata
            self.registry.save(document)
            return

    def _write_execution_failure_lesson(self, session_id: str, exc: Exception) -> None:
        try:
            session = self.registry.get_session(session_id)
            targets_doc, skillruntimes_doc = self._load_registries()
            scheduled = self.scheduler.resolve_session(session, targets_doc, skillruntimes_doc)
            from PhyAgentOS.runtime.watchdog.errors import error_code_for

            error_code = error_code_for(exc)
            summary = f"{type(exc).__name__}: {exc}"
            self.result_writer.write_lesson(
                session,
                scheduled.target_spec.id,
                scheduled.skillruntime_id,
                str(session.status),
                error_code,
                summary,
                {
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "session_status": str(session.status),
                },
            )
        except Exception:
            return
