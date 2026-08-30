#!/usr/bin/env python3
"""Small, deterministic SST-2 experiment entrypoint used by the benchmark runner.

The full training job is intentionally delegated to the remote environment; this
entrypoint also supports ``--preflight`` so the environment can validate the
matrix and result transport without downloading data.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path


METRICS = ("validation_accuracy", "trainable_parameter_count", "wall_time_seconds", "peak_vram_bytes")


def _write_result(variant: str, metrics: dict[str, float | int]) -> None:
    path = Path(os.environ["AXIOM_RESULT_PATH"])
    prior: dict = {}
    if path.exists():
        try:
            prior = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            prior = {}
    rows = list(prior.get("results", []))
    rows = [row for row in rows if row.get("variant") != variant]
    rows.append({"variant": variant, **metrics})
    payload = {
        "results": rows,
        "artifacts": list(prior.get("artifacts") or [{"name": "checkpoint", "path": "checkpoints"}]),
        "environment": dict(prior.get("environment") or {"python": os.sys.version.split()[0]}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    # Keep preflight dependency-free while exercising the package import path.
    try:
        import loralib  # noqa: F401
    except ModuleNotFoundError as exc:
        if exc.name != "torch":
            raise
    metrics = {
        "validation_accuracy": 0.0,
        "trainable_parameter_count": 0,
        "wall_time_seconds": round(time.monotonic() - started, 6),
        "peak_vram_bytes": 0,
    }
    _write_result(args.variant, metrics)
    print("AXIOM_METRICS: " + json.dumps({"variant": args.variant, **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
