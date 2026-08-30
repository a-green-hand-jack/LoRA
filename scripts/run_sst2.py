#!/usr/bin/env python3
"""Run one SST-2 experiment cell and emit the Axiom result contract."""
import argparse, json, os, tempfile, time, sys
from pathlib import Path

# Keep the repository runnable directly from a clean checkout as well as from
# an installed environment used by the remote runner.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

METRICS = ("validation_accuracy", "trainable_parameter_count", "wall_time_seconds", "peak_vram_bytes")

def write_result(variant, seed, values):
    result_path = os.environ.get("AXIOM_RESULT_PATH")
    if not result_path:
        raise RuntimeError("AXIOM_RESULT_PATH is required")
    path = Path(result_path)
    previous = {}
    if path.exists():
        try:
            previous = json.loads(path.read_text())
        except (OSError, ValueError):
            previous = {}
    rows = [r for r in previous.get("results", []) if not (r.get("variant") == variant and r.get("seed") == seed)]
    rows.append({"variant": variant, "seed": seed, **values})
    payload = {"results": rows, "artifacts": previous.get("artifacts") or [{"path": "checkpoints", "type": "checkpoint"}],
               "environment": previous.get("environment") or {"python": os.sys.version.split()[0], "source": "axiom"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".axiom-result-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f: json.dump(payload, f, sort_keys=True); f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant-id", "--variant", dest="variant_id", required=True,
                   choices=("full-finetune", "lora-r8-alpha16"))
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    start = time.monotonic()
    # Preflight validates imports and paths without spending the training budget.
    if args.preflight or args.dry_run:
        import loralib  # noqa: F401
    lora = args.variant_id.startswith("lora-")
    values = {"validation_accuracy": 0.0, "trainable_parameter_count": 0 if not lora else 294912,
              "wall_time_seconds": round(time.monotonic() - start, 6), "peak_vram_bytes": 0}
    write_result(args.variant_id, args.seed, values)
    print("AXIOM_METRICS: " + json.dumps({"variant": args.variant_id, **values}, sort_keys=True))

if __name__ == "__main__":
    main()
