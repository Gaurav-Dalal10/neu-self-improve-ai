from ._runner_common import build_arg_parser, run_benchmark


if __name__ == "__main__":
    args = build_arg_parser("2wiki").parse_args()
    print(run_benchmark("2wiki", args))
