import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_dotenv(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    data: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        data[k.strip()] = v.strip().strip('"').strip("'")
    return data


def validate_provider_env(required: Optional[Iterable[str]] = None) -> Dict[str, List[str]]:
    required_keys = list(required or ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
    missing = [k for k in required_keys if not os.getenv(k)]
    return {"missing": missing, "ok": len(missing) == 0}


def canonicalize_openai_env() -> Dict[str, str]:
    """
    Normalize env names so downstream code can rely on OPENAI_*.
    """
    if not os.getenv("OPENAI_API_KEY") and os.getenv("OPENAI_COMPAT_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_COMPAT_API_KEY", "")
    if not os.getenv("OPENAI_BASE_URL") and os.getenv("OPENAI_COMPAT_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = os.getenv("OPENAI_COMPAT_BASE_URL", "")

    return {
        "OPENAI_API_KEY": "set" if os.getenv("OPENAI_API_KEY") else "",
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", ""),
    }


def assert_stdlib_not_shadowed() -> None:
    """
    Fail fast for the common `types.py` shadowing issue.
    """
    import types as std_types  # local import to avoid import side effects at module load

    p = str(getattr(std_types, "__file__", ""))
    if "python" not in p.lower() or "site-packages" in p.lower():
        raise RuntimeError(
            f"Python stdlib types module appears shadowed: {p}. "
            "Unset PYTHONPATH and set it to AGENTFLOW_DIR only."
        )


def resolve_agentflow_dir(explicit: Optional[str] = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    if os.getenv("AGENTFLOW_DIR"):
        candidates.append(Path(os.getenv("AGENTFLOW_DIR", "")))
    candidates.append(Path("AgentFlow"))

    for c in candidates:
        if c and c.exists() and (c / "test").exists():
            return c.resolve()
    raise FileNotFoundError(
        "AgentFlow repo not found. Provide --agentflow_dir or set AGENTFLOW_DIR to local clone path."
    )


def ensure_agentflow_env(agentflow_dir: Path, source_env_file: str = ".env") -> Path:
    af_env_template = agentflow_dir / "agentflow" / ".env.template"
    af_env = agentflow_dir / "agentflow" / ".env"

    if not af_env_template.exists():
        raise FileNotFoundError(f"Missing env template: {af_env_template}")

    if not af_env.exists():
        af_env.write_text(af_env_template.read_text(encoding="utf-8"), encoding="utf-8")

    merged = parse_dotenv(af_env)
    merged.update(parse_dotenv(Path(source_env_file)))

    # Also pull process env for known keys.
    for k in [
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_API_KEY",
        "PORTKEY_API_KEY",
        "TOGETHER_API_KEY",
        "DASHSCOPE_API_KEY",
    ]:
        if os.getenv(k):
            merged[k] = os.getenv(k, "")

    lines = [f"{k}={v}" for k, v in sorted(merged.items())]
    af_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return af_env
