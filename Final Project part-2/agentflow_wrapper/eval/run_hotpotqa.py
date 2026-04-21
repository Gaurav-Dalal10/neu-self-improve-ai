from ._runner_common import build_arg_parser, run_benchmark


if __name__ == "__main__":
    args = build_arg_parser("hotpotqa").parse_args()
    print(run_benchmark("hotpotqa", args))
