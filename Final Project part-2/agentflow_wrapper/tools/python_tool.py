import subprocess
import tempfile
from typing import Dict


def run_python(code: str, timeout_sec: int = 5) -> Dict[str, object]:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=True) as fp:
        fp.write(code)
        fp.flush()
        try:
            proc = subprocess.run(
                ["python3", fp.name],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                check=False,
            )
            return {
                "status": "ok" if proc.returncode == 0 else "error",
                "returncode": proc.returncode,
                "stdout": proc.stdout[-4000:],
                "stderr": proc.stderr[-4000:],
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": f"Timed out after {timeout_sec}s"}
