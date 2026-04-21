import argparse
import json
import os
from pathlib import Path

REQUIRED_ENV = [
    "OPENAI_COMPAT_BASE_URL",
    "OPENAI_COMPAT_API_KEY",
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--check_endpoint", action="store_true")
    p.add_argument("--output_dir", type=str, default="final_project_part2/training_modal/checkpoints")
    args = p.parse_args()

    report = {
        "endpoint": {"ok": True, "missing": []},
        "output_dir": {"ok": True, "path": args.output_dir},
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.check_endpoint:
        missing = [k for k in REQUIRED_ENV if not os.getenv(k)]
        report["endpoint"]["missing"] = missing
        report["endpoint"]["ok"] = len(missing) == 0

    report_path = out / "preflight_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if args.check_endpoint and not report["endpoint"]["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
