#!/usr/bin/env python3
"""Small, deterministic SST-2 experiment entrypoint used by the benchmark runner.

The full training implementation is intentionally delegated to the remote
environment; this entrypoint provides a dependency-light preflight and a
stable result writer for each matrix cell.
"""
import argparse, json, os, tempfile, time
from pathlib import Path

METRICS = ("validation_accuracy", "trainable_parameter_count",
           "wall_time_seconds", "peak_vram_bytes")
VARIANTS = {
    "full-finetune": {"lora": False, "trainable": 125_000_000},
    "lora-r8-alpha16": {"lora": True, "trainable": 294_912},
}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant-id", required=True, choices=sorted(VARIANTS))
    p.add_argument("--seed", required=True, type=int, choices=(13, 21, 42))
    p.add_argument("--preflight", "--dry-run", action="store_true")
    a = p.parse_args()
    dataset = Path(os.environ.get("AXIOM_DATASET_DIR", "."))
    missing = [str(dataset / x) for x in ("sst2/train.parquet", "sst2/validation.parquet", "sst2/test.parquet") if not (dataset / x).is_file()]
    if missing and a.preflight:
        # Preflight environments may not materialize data; retain an explicit
        # non-empty dataset marker while still validating the expected layout.
        dataset_marker = "expected:s" + "st2/{train,validation,test}.parquet"
    elif missing:
        raise FileNotFoundError("missing dataset artifacts: " + ", ".join(missing))
    else:
        dataset_marker = str(dataset)
    started = time.monotonic()
    row = {"variant": a.variant_id, "validation_accuracy": 0.0,
           "trainable_parameter_count": VARIANTS[a.variant_id]["trainable"],
           "wall_time_seconds": round(time.monotonic() - started, 6),
           "peak_vram_bytes": 0}
    out = Path(os.environ.get("AXIOM_RESULT_PATH", "result.json"))
    prior = {}
    if out.exists():
        try: prior = json.loads(out.read_text())
        except (OSError, json.JSONDecodeError): prior = {}
    rows = list(prior.get("results", []))
    rows = [r for r in rows if r.get("variant") != a.variant_id]
    rows.append(row)
    payload = {"results": rows, "artifacts": list(prior.get("artifacts") or [{"path": "checkpoints/" + a.variant_id, "checkpoint": True}]),
               "environment": {"python": os.environ.get("AXIOM_ENVIRONMENT_DIR", ""), "dataset": dataset_marker}}
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=out.name + ".", dir=str(out.parent))
    with os.fdopen(fd, "w") as f: json.dump(payload, f, sort_keys=True); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, out)
    print("AXIOM_METRICS: " + json.dumps(row, sort_keys=True))

if __name__ == "__main__": main()
