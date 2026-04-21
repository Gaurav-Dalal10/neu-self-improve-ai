import pytest

from final_project_part2.training_modal.modal_train_flowgrpo_lora import _resolve_mode_cfg
from final_project_part2.training_modal.train_launcher import _require_endpoint_env


def test_dev_timeout_policy_enforced():
    cfg = {
        "timeout_dev_sec": 700,
        "timeout_final_sec": 21600,
        "max_steps_dev": 10,
        "max_steps_final": 100,
    }
    with pytest.raises(ValueError):
        _resolve_mode_cfg(cfg, "dev")


def test_require_endpoint_env(monkeypatch):
    monkeypatch.delenv("OPENAI_COMPAT_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        _require_endpoint_env()
