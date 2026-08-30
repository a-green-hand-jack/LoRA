#!/usr/bin/env python3
"""Deterministic CPU preflight/matrix entrypoint for the LoRA SST-2 plan."""
import argparse, json, os, tempfile, time
from pathlib import Path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant-id", required=True, choices=("full-finetune", "lora-r8-alpha16"))
    p.add_argument("--seed", required=True, type=int, choices=(13, 21, 42))
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    if not (a.preflight or a.dry_run):
        p.error("--preflight or --dry-run is required")
    metrics = {
        "validation_accuracy": 0.0,
        "trainable_parameter_count": 125000000 if a.variant_id == "full-finetune" else 800000,
        "wall_time_seconds": 0.0,
        "peak_vram_bytes": 0,
    }
    row = {"variant": {"id": a.variant_id}, "seed": a.seed, **metrics}
    result_path = Path(os.environ["AXIOM_RESULT_PATH"])
    prior = {"results": [], "artifacts": [{"path": "preflight.json", "kind": "checkpoint"}], "environment": {"code_sha": "c4593f060e6a368d7bb5af5273b8e42810cdef90"}}
    if result_path.exists():
        try:
            loaded = json.loads(result_path.read_text())
            if loaded.get("results") and loaded.get("artifacts") and loaded.get("environment"):
                prior = loaded
        except (OSError, ValueError):
            pass
    prior["results"] = [r for r in prior["results"] if not (r.get("variant", {}).get("id") == a.variant_id and r.get("seed") == a.seed)] + [row]
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".axiom-result-", dir=result_path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(prior, f, sort_keys=True)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, result_path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    print("AXIOM_METRICS: " + json.dumps({"variant": row["variant"], **metrics}, sort_keys=True))

if __name__ == "__main__":
    main()
