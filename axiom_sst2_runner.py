#!/usr/bin/env python3
"""Run one approved RoBERTa/SST-2 experiment and emit Axiom results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path


VARIANTS = {
    "full-finetune": {"lora": False},
    "lora-r8-alpha16": {"lora": True, "rank": 8, "alpha": 16},
}
DATA_FILES = {
    "train.parquet": "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    "validation.parquet": "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
    "test.parquet": "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1",
}
METRICS = (
    "validation_accuracy",
    "trainable_parameter_count",
    "wall_time_seconds",
    "peak_vram_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", required=True, type=int, choices=(13, 21, 42))
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_dataset(dataset_dir: Path) -> dict[str, Path]:
    paths = {name: dataset_dir / "sst2" / name for name in DATA_FILES}
    for name, path in paths.items():
        if not path.is_file():
            raise RuntimeError(f"missing dataset artifact: {path}")
        actual = sha256(path)
        if actual != DATA_FILES[name]:
            raise RuntimeError(f"checksum mismatch for {path}: {actual}")
    return paths


def train(args: argparse.Namespace, paths: dict[str, Path], output_dir: Path) -> dict[str, int | float]:
    started = time.monotonic()
    import pyarrow.parquet as parquet

    converted = output_dir / "dataset"
    converted.mkdir(parents=True, exist_ok=True)
    json_paths = {}
    for split in ("train", "validation"):
        json_path = converted / f"{split}.json"
        parquet.read_table(paths[f"{split}.parquet"]).to_pandas().to_json(
            json_path, orient="records", lines=True
        )
        json_paths[split] = json_path
    command = [
        sys.executable,
        "examples/NLU/examples/text-classification/run_glue.py",
        "--model_name_or_path", str(Path(os.environ["AXIOM_ENVIRONMENT_DIR"]) / "models" / "roberta-base"),
        "--train_file", str(json_paths["train"]),
        "--validation_file", str(json_paths["validation"]),
        "--do_train", "--do_eval",
        "--max_seq_length", "128",
        "--per_device_train_batch_size", "32",
        "--per_device_eval_batch_size", "64",
        "--learning_rate", "2e-5",
        "--num_train_epochs", "3",
        "--output_dir", str(output_dir),
        "--overwrite_output_dir",
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--load_best_model_at_end",
        "--metric_for_best_model", "accuracy",
        "--seed", str(args.seed),
    ]
    if VARIANTS[args.variant]["lora"]:
        command.extend(["--apply_lora", "--lora_r", "8", "--lora_alpha", "16"])
    subprocess.run(command, check=True)
    values = json.loads((output_dir / "eval_results.json").read_text())
    return {
        "validation_accuracy": values.get("eval_accuracy", values.get("accuracy")),
        "trainable_parameter_count": values["trainable_parameter_count"],
        "wall_time_seconds": time.monotonic() - started,
        "peak_vram_bytes": values["peak_vram_bytes"],
    }


def merge_result(path: Path, row: dict, artifact: dict, environment: dict) -> None:
    payload = {"schema": "result.json-v1", "results": [], "artifacts": [], "environment": environment}
    if path.exists():
        prior = json.loads(path.read_text())
        if prior.get("schema") != "result.json-v1" or not isinstance(prior.get("results"), list):
            raise RuntimeError(f"invalid prior result: {path}")
        payload = prior
    key = (row["variant"], row["seed"])
    payload["results"] = [item for item in payload["results"] if (item.get("variant"), item.get("seed")) != key]
    payload["results"].append(row)
    if artifact not in payload.get("artifacts", []):
        payload.setdefault("artifacts", []).append(artifact)
    payload["environment"] = payload.get("environment") or environment
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(os.environ["AXIOM_DATASET_DIR"])
    result_path = Path(os.environ["AXIOM_RESULT_PATH"])
    paths = validate_dataset(dataset_dir)
    output_dir = result_path.parent / "checkpoints" / f"{args.variant}-seed-{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.preflight or args.dry_run:
        metrics = {name: 0 for name in METRICS}
        artifact = {"type": "preflight", "path": str(output_dir)}
    else:
        metrics = train(args, paths, output_dir)
        artifact = {"type": "checkpoint", "path": str(output_dir)}
    if any(value is None for value in metrics.values()):
        raise RuntimeError(f"missing required metric: {metrics}")
    row = {"variant": args.variant, "seed": args.seed, **metrics}
    environment = json.loads(Path("axiom-environment.json").read_text())
    merge_result(result_path, row, artifact, environment)
    print("AXIOM_METRICS: " + json.dumps({"variant": args.variant, **metrics}, sort_keys=True))


if __name__ == "__main__":
    main()
