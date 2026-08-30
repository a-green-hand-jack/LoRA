"""Axiom matrix entrypoint for the SST-2 LoRA comparison.

The runner is intentionally dependency-light for preflight: it validates the
matrix arguments and records a result row using the same atomic result format
used by full runs.  Training infrastructure may enrich the row in-place.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

METRICS = ("validation_accuracy", "trainable_parameter_count", "wall_time_seconds", "peak_vram_bytes")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant-id", required=True, choices=("full-finetune", "lora-r8-alpha16"))
    p.add_argument("--seed", required=True, type=int, choices=(13, 21, 42))
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    if not (args.preflight or args.dry_run):
        p.error("one of --preflight or --dry-run is required")

    row = {
        "variant": args.variant_id,
        "seed": args.seed,
        "validation_accuracy": 0.0,
        "trainable_parameter_count": 125_000_000 if args.variant_id == "full-finetune" else 294_912,
        "wall_time_seconds": 0.0,
        "peak_vram_bytes": 0,
    }
    target = Path(os.environ["AXIOM_RESULT_PATH"])
    target.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if target.exists():
        try:
            prior = json.loads(target.read_text())
        except (OSError, json.JSONDecodeError):
            prior = {}
    results = list(prior.get("results", []))
    # Matrix invocations share one result path: retain prior rows and add this
    # cell exactly once, without replacing an earlier valid result.
    key = (row["variant"], row["seed"])
    if not any((r.get("variant"), r.get("seed")) == key for r in results):
        results.append(row)
    artifacts = prior.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        artifacts = [{"path": "checkpoints/" + args.variant_id}]
    environment = prior.get("environment")
    if not isinstance(environment, dict) or not environment:
        environment = {"source": "axiom-environment.json"}
    payload = {"results": results, "artifacts": artifacts, "environment": environment}
    fd, name = tempfile.mkstemp(prefix=".axiom-result-", dir=str(target.parent))
    with os.fdopen(fd, "w") as f:
        json.dump(payload, f, sort_keys=True)
        f.flush(); os.fsync(f.fileno())
    os.replace(name, target)
    print("AXIOM_METRICS: " + json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
