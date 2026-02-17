from pathlib import Path

import git

from eval.harness import EvaluationConfig, LiteHarness


def _init_repo(repo_dir: Path):
    repo_dir.mkdir(parents=True, exist_ok=True)
    repo = git.Repo.init(repo_dir)
    with repo.config_writer() as cfg:
        cfg.set_value("user", "name", "Test User")
        cfg.set_value("user", "email", "test@example.com")

    target_file = repo_dir / "data.txt"
    target_file.write_text("value=1\n", encoding="utf-8")
    repo.index.add(["data.txt"])
    repo.index.commit("initial commit")
    return repo, target_file


def _create_patch(repo: git.Repo, file_path: Path, new_content: str) -> str:
    file_path.write_text(new_content, encoding="utf-8")
    patch = repo.git.diff()
    repo.git.reset("--hard")
    return patch


def test_diff_but_failing_tests_resolved_zero(tmp_path: Path):
    repo, target_file = _init_repo(tmp_path / "repo_fail")
    base_commit = repo.head.commit.hexsha
    patch = _create_patch(repo, target_file, "value=2\n")

    harness = LiteHarness(EvaluationConfig(timeout_seconds=30, max_log_bytes=10000))
    record = harness.evaluate_instance(
        instance={
            "instance_id": "case_fail",
            "repo": str((tmp_path / "repo_fail").resolve()),
            "base_commit": base_commit,
            "FAIL_TO_PASS": [],
        },
        patch_text=patch,
        output_dir=tmp_path / "out_fail",
        tests_command_override='python3 -c "import sys; sys.exit(1)"',
    )

    assert record.patch_generated == 1
    assert record.tests_exit_code == 1
    assert record.resolved == 0


def test_no_diff_resolved_zero(tmp_path: Path):
    repo, _ = _init_repo(tmp_path / "repo_no_patch")
    base_commit = repo.head.commit.hexsha

    harness = LiteHarness(EvaluationConfig(timeout_seconds=30, max_log_bytes=10000))
    record = harness.evaluate_instance(
        instance={
            "instance_id": "case_no_patch",
            "repo": str((tmp_path / "repo_no_patch").resolve()),
            "base_commit": base_commit,
            "FAIL_TO_PASS": [],
        },
        patch_text="",
        output_dir=tmp_path / "out_no_patch",
        tests_command_override='python3 -c "import sys; sys.exit(0)"',
    )

    assert record.patch_generated == 0
    assert record.tests_exit_code is None
    assert record.resolved == 0


def test_diff_and_passing_tests_resolved_one(tmp_path: Path):
    repo, target_file = _init_repo(tmp_path / "repo_pass")
    base_commit = repo.head.commit.hexsha
    patch = _create_patch(repo, target_file, "value=2\n")

    harness = LiteHarness(EvaluationConfig(timeout_seconds=30, max_log_bytes=10000))
    command = (
        'python3 -c "import pathlib, sys; '
        'txt=pathlib.Path(\'data.txt\').read_text(); '
        'sys.exit(0 if \'value=2\' in txt else 1)"'
    )
    record = harness.evaluate_instance(
        instance={
            "instance_id": "case_pass",
            "repo": str((tmp_path / "repo_pass").resolve()),
            "base_commit": base_commit,
            "FAIL_TO_PASS": [],
        },
        patch_text=patch,
        output_dir=tmp_path / "out_pass",
        tests_command_override=command,
    )

    assert record.patch_generated == 1
    assert record.tests_exit_code == 0
    assert record.resolved == 1
