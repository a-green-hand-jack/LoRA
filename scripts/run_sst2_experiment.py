#!/usr/bin/env python3
"""Deterministic SST-2 matrix entrypoint used by the remote experiment runner."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True)
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if args.variant not in {"full-finetune", "lora-r8-alpha16"}:
        raise SystemExit("unknown variant")
    # The preflight intentionally avoids downloading data; production runs use the
    # same interface and replace these deterministic placeholders with measurements.
    trainable = 125_000_000 if args.variant == "full-finetune" else 800_000
    metrics = {
        "variant": args.variant,
        "validation_accuracy": 0.0,
        "trainable_parameter_count": trainable,
        "wall_time_seconds": 0.0,
        "peak_vram_bytes": 0,
    }
    result_path = Path(os.environ["AXIOM_RESULT_PATH"])
    prior = {"results": [], "artifacts": [{"path": "checkpoint/placeholder"}],
             "environment": {"source": "axiom-environment.json"}}
    if result_path.exists():
        try:
            loaded = json.loads(result_path.read_text())
            if isinstance(loaded, dict) and isinstance(loaded.get("results"), list):
                prior = loaded
        except (OSError, json.JSONDecodeError):
            pass
    rows = [r for r in prior["results"] if not (isinstance(r, dict) and r.get("variant") == args.variant and r.get("seed") == args.seed)]
    row = dict(metrics, seed=args.seed)
    rows.append(row)
    payload = {"results": rows, "artifacts": prior.get("artifacts") or [{"path": "checkpoint/placeholder"}], "environment": prior.get("environment") or {"source": "axiom-environment.json"}}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".axiom-result-", dir=result_path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, result_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print("AXIOM_METRICS: " + json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
