import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

from vllm.benchmarks.serve import add_cli_args, main


def load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def build_args(cfg: dict) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    add_cli_args(parser)

    argv: list[str] = []
    for key, value in cfg.items():
        if key == "env":
            continue
        flag = f"--{key}"
        if isinstance(value, bool):
            if value:
                argv.append(flag)
        elif isinstance(value, list):
            for v in value:
                argv += [flag, str(v)]
        else:
            argv += [flag, str(value)]
    return parser.parse_args(argv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to YAML config file")
    cli = parser.parse_args()

    cfg = load_config(cli.config)
    for k, v in (cfg.get("env") or {}).items():
        os.environ.setdefault(k, str(v))

    tokenizer_override = os.environ.get("VLLM_BENCH_TOKENIZER_DIR")
    if tokenizer_override:
        cfg["tokenizer"] = tokenizer_override

    args = build_args(cfg)
    result = main(args)
    print(result)

    results_dir = Path(os.environ.get("VLLM_BENCH_RESULTS_DIR", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / f"{cli.config.stem}.json"
    payload = {
        "args": vars(args),
        "result": result,
    }
    with out_path.open("w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"[bench] wrote results to {out_path}")
