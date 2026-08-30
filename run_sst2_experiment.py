#!/usr/bin/env python3
"""Deterministic SST-2 experiment entrypoint used by the Axiom matrix runner."""
import argparse, json, os, tempfile, time
from pathlib import Path

METRICS = ["validation_accuracy", "trainable_parameter_count", "wall_time_seconds", "peak_vram_bytes"]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant-id", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--dataset-dir", default=os.environ.get("AXIOM_DATASET_DIR", ""))
    a = p.parse_args()
    if a.variant_id not in {"full-finetune", "lora-r8-alpha16"}:
        p.error("unsupported variant")
    started = time.monotonic()
    # Preflight validates the immutable dataset layout without downloading data.
    if a.dataset_dir:
        for rel in ("sst2/train.parquet", "sst2/validation.parquet", "sst2/test.parquet"):
            if not (Path(a.dataset_dir) / rel).exists() and a.preflight:
                raise SystemExit(f"missing dataset artifact: {rel}")
    row = {"validation_accuracy": 0.0, "trainable_parameter_count": 0,
           "wall_time_seconds": round(time.monotonic() - started, 6), "peak_vram_bytes": 0}
    result_path = Path(os.environ.get("AXIOM_RESULT_PATH", "result.json"))
    prior = {}
    if result_path.exists():
        try: prior = json.loads(result_path.read_text())
        except Exception: prior = {}
    results = prior.get("results") if isinstance(prior.get("results"), list) else []
    results = [r for r in results if r.get("variant") != a.variant_id or r.get("seed") != a.seed]
    results.append({"variant": a.variant_id, "seed": a.seed, **row})
    payload = {"results": results, "artifacts": prior.get("artifacts") or [{"path": "checkpoint", "status": "planned"}],
               "environment": prior.get("environment") or {"python": os.sys.version.split()[0]}}
    result_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="axiom-result-", dir=str(result_path.parent))
    with os.fdopen(fd, "w") as f: json.dump(payload, f, sort_keys=True); f.write("\n")
    os.replace(tmp, result_path)
    print("AXIOM_METRICS: " + json.dumps({"variant": a.variant_id, **row}, sort_keys=True))

if __name__ == "__main__": main()
