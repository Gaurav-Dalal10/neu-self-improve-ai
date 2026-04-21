from final_project_part2.wrapper.env_setup import validate_provider_env


def test_provider_env_validation_missing(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    out = validate_provider_env()
    assert out["ok"] is False
    assert "OPENAI_BASE_URL" in out["missing"]
    assert "OPENAI_API_KEY" in out["missing"]
