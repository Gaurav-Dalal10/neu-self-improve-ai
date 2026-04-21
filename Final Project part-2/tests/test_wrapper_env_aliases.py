from final_project_part2.wrapper.env_setup import canonicalize_openai_env


def test_openai_alias_mapping(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "k")
    monkeypatch.setenv("OPENAI_COMPAT_BASE_URL", "https://api.portkey.ai/v1")

    out = canonicalize_openai_env()
    assert out["OPENAI_API_KEY"] == "set"
    assert out["OPENAI_BASE_URL"] == "https://api.portkey.ai/v1"
