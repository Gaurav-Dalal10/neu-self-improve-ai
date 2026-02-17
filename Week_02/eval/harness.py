"""Lite evaluation harness for SWE-bench-style patch validation.

This harness evaluates candidate patches by:
1. checking out a repository at base commit,
2. applying the generated patch,
3. running tests with timeouts and log caps,
4. computing resolved=1 only when tests pass.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import git


@dataclass
class EvaluationConfig:
    """Runtime configuration for patch evaluation."""

    timeout_seconds: int = 900
    max_log_bytes: int = 200_000
    default_tests_command: str = "python3 -m pytest -q"


@dataclass
class EvaluationRecord:
    """Normalized evaluation output for one trajectory/instance."""

    instance_id: str
    repo: str
    base_commit: str
    patch_path: str
    tests_command: str
    tests_exit_code: Optional[int]
    resolved: int
    patch_generated: int
    runtime_seconds: float
    error_type: str
    logs_path: str


class LiteHarness:
    """Faithful-lite SWE-bench evaluator.

    Notes:
    - This is not the official SWE-bench Docker harness.
    - It still enforces key semantics: patch must apply and tests must pass.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        self.config = config or EvaluationConfig()

    def infer_tests_command(
        self,
        instance: Dict[str, Any],
        tests_command_override: Optional[str] = None,
    ) -> str:
        """Infer test command from instance metadata or explicit override."""
        if tests_command_override:
            return tests_command_override

        fail_to_pass = instance.get("FAIL_TO_PASS") or []
        if isinstance(fail_to_pass, str):
            try:
                fail_to_pass = json.loads(fail_to_pass)
            except json.JSONDecodeError:
                fail_to_pass = []

        if fail_to_pass:
            quoted_tests = " ".join(shlex.quote(str(test_id)) for test_id in fail_to_pass)
            return f"python3 -m pytest -q {quoted_tests}"

        return self.config.default_tests_command

    def evaluate_instance(
        self,
        instance: Dict[str, Any],
        patch_text: str,
        output_dir: Path,
        tests_command_override: Optional[str] = None,
    ) -> EvaluationRecord:
        """Evaluate one patch candidate for an instance."""
        instance_id = str(instance["instance_id"])
        repo = str(instance["repo"])
        base_commit = str(instance["base_commit"])
        tests_command = self.infer_tests_command(instance, tests_command_override)

        instance_dir = output_dir / instance_id.replace("/", "_")
        instance_dir.mkdir(parents=True, exist_ok=True)

        patch_generated = int(bool((patch_text or "").strip()))
        patch_path = instance_dir / "patch.diff"
        if patch_generated:
            normalized_patch = patch_text if patch_text.endswith("\n") else patch_text + "\n"
            patch_path.write_text(normalized_patch, encoding="utf-8")

        if not patch_generated:
            return self._finalize_record(
                instance=instance,
                patch_path=patch_path if patch_generated else None,
                tests_command=tests_command,
                tests_exit_code=None,
                resolved=0,
                patch_generated=0,
                runtime_seconds=0.0,
                error_type="no_patch",
                logs_dir=instance_dir,
            )

        started = time.monotonic()
        try:
            with tempfile.TemporaryDirectory(prefix="swe_eval_") as work_tmp:
                work_path = Path(work_tmp)
                eval_repo_path = work_path / "repo"

                clone_error = self._clone_and_checkout(repo, base_commit, eval_repo_path)
                if clone_error:
                    runtime_seconds = time.monotonic() - started
                    return self._finalize_record(
                        instance=instance,
                        patch_path=patch_path,
                        tests_command=tests_command,
                        tests_exit_code=None,
                        resolved=0,
                        patch_generated=1,
                        runtime_seconds=runtime_seconds,
                        error_type=clone_error,
                        logs_dir=instance_dir,
                    )

                apply_error = self._apply_patch(eval_repo_path, patch_path)
                if apply_error:
                    runtime_seconds = time.monotonic() - started
                    return self._finalize_record(
                        instance=instance,
                        patch_path=patch_path,
                        tests_command=tests_command,
                        tests_exit_code=None,
                        resolved=0,
                        patch_generated=1,
                        runtime_seconds=runtime_seconds,
                        error_type=apply_error,
                        logs_dir=instance_dir,
                    )

                run_result = self._run_tests(eval_repo_path, tests_command)
                runtime_seconds = time.monotonic() - started

                stdout_path = instance_dir / "stdout.log"
                stderr_path = instance_dir / "stderr.log"
                stdout_path.write_text(run_result["stdout"], encoding="utf-8")
                stderr_path.write_text(run_result["stderr"], encoding="utf-8")

                resolved = int(run_result["exit_code"] == 0)
                return self._finalize_record(
                    instance=instance,
                    patch_path=patch_path,
                    tests_command=tests_command,
                    tests_exit_code=run_result["exit_code"],
                    resolved=resolved,
                    patch_generated=1,
                    runtime_seconds=runtime_seconds,
                    error_type=run_result["error_type"],
                    logs_dir=instance_dir,
                )
        except Exception:
            runtime_seconds = time.monotonic() - started
            return self._finalize_record(
                instance=instance,
                patch_path=patch_path,
                tests_command=tests_command,
                tests_exit_code=None,
                resolved=0,
                patch_generated=1,
                runtime_seconds=runtime_seconds,
                error_type="harness_exception",
                logs_dir=instance_dir,
            )

    def _clone_and_checkout(self, repo: str, base_commit: str, repo_path: Path) -> str:
        """Clone repo and checkout base commit. Returns empty string on success."""
        try:
            repo_url = self._resolve_repo_url(repo)
            repo_obj = git.Repo.clone_from(repo_url, repo_path)
            repo_obj.git.checkout(base_commit, force=True)
            return ""
        except Exception:
            return "clone_or_checkout_failed"

    def _apply_patch(self, repo_path: Path, patch_path: Path) -> str:
        """Apply unified diff patch. Returns empty string on success."""
        try:
            subprocess.run(
                ["git", "apply", "--whitespace=nowarn", str(patch_path)],
                cwd=repo_path,
                check=True,
                capture_output=True,
                text=True,
            )
            return ""
        except subprocess.CalledProcessError:
            return "patch_apply_failed"

    def _run_tests(self, repo_path: Path, command: str) -> Dict[str, Any]:
        """Run tests with timeout and bounded logs."""
        try:
            completed = subprocess.run(
                command,
                cwd=repo_path,
                shell=True,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
            )
            return {
                "exit_code": completed.returncode,
                "stdout": self._cap_text(completed.stdout),
                "stderr": self._cap_text(completed.stderr),
                "error_type": "none",
            }
        except subprocess.TimeoutExpired as exc:
            stdout = self._cap_text((exc.stdout or "") + "\n[Test run timed out]\n")
            stderr = self._cap_text(exc.stderr or "")
            return {
                "exit_code": 124,
                "stdout": stdout,
                "stderr": stderr,
                "error_type": "test_timeout",
            }
        except Exception as exc:
            return {
                "exit_code": 1,
                "stdout": "",
                "stderr": self._cap_text(str(exc)),
                "error_type": "test_runner_failed",
            }

    def _finalize_record(
        self,
        instance: Dict[str, Any],
        patch_path: Optional[Path],
        tests_command: str,
        tests_exit_code: Optional[int],
        resolved: int,
        patch_generated: int,
        runtime_seconds: float,
        error_type: str,
        logs_dir: Path,
    ) -> EvaluationRecord:
        record = EvaluationRecord(
            instance_id=str(instance["instance_id"]),
            repo=str(instance["repo"]),
            base_commit=str(instance["base_commit"]),
            patch_path=str(patch_path) if patch_path else "",
            tests_command=tests_command,
            tests_exit_code=tests_exit_code,
            resolved=resolved,
            patch_generated=patch_generated,
            runtime_seconds=round(runtime_seconds, 3),
            error_type=error_type,
            logs_path=str(logs_dir),
        )

        metadata = {
            **asdict(record),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
        }
        (logs_dir / "evaluation.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        return record

    def _resolve_repo_url(self, repo: str) -> str:
        candidate = Path(repo)
        if candidate.exists():
            return str(candidate.resolve())
        if repo.startswith("http://") or repo.startswith("https://"):
            return repo
        return f"https://github.com/{repo}"

    def _cap_text(self, text: str) -> str:
        raw = text.encode("utf-8", errors="replace")
        if len(raw) <= self.config.max_log_bytes:
            return text

        clipped = raw[: self.config.max_log_bytes].decode("utf-8", errors="ignore")
        return clipped + "\n[output truncated]\n"


def aggregate_records(records: List[EvaluationRecord]) -> Dict[str, Any]:
    """Compute per-run and deduplicated-per-instance metrics."""
    if not records:
        return {
            "total_runs": 0,
            "resolved_runs": 0,
            "resolved_rate_runs": 0.0,
            "patch_generated_runs": 0,
            "patch_generated_rate_runs": 0.0,
            "total_unique_instances": 0,
            "resolved_unique_instances": 0,
            "resolved_rate_unique_instances": 0.0,
            "patch_generated_unique_instances": 0,
            "patch_generated_rate_unique_instances": 0.0,
        }

    total_runs = len(records)
    resolved_runs = sum(r.resolved for r in records)
    patch_generated_runs = sum(r.patch_generated for r in records)

    deduped: Dict[str, EvaluationRecord] = {}
    for record in records:
        # Keep last occurrence for pass@1-style unique instance reporting.
        deduped[record.instance_id] = record

    deduped_records = list(deduped.values())
    total_unique = len(deduped_records)
    resolved_unique = sum(r.resolved for r in deduped_records)
    patch_generated_unique = sum(r.patch_generated for r in deduped_records)

    return {
        "total_runs": total_runs,
        "resolved_runs": resolved_runs,
        "resolved_rate_runs": round(resolved_runs / total_runs, 4),
        "patch_generated_runs": patch_generated_runs,
        "patch_generated_rate_runs": round(patch_generated_runs / total_runs, 4),
        "total_unique_instances": total_unique,
        "resolved_unique_instances": resolved_unique,
        "resolved_rate_unique_instances": round(
            resolved_unique / total_unique if total_unique else 0.0, 4
        ),
        "patch_generated_unique_instances": patch_generated_unique,
        "patch_generated_rate_unique_instances": round(
            patch_generated_unique / total_unique if total_unique else 0.0, 4
        ),
    }
