#!/usr/bin/env python3
"""Reproducible RoBERTa SST-2 comparison runner for Axiom matrix jobs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path


BASE_REVISION = "c4593f060e6a368d7bb5af5273b8e42810cdef90"
MODEL_REVISION = "e2da8e2f811d1448a5b465c236feacd80ffbac7b"
DATA_FILES = {
    "train": ("train.parquet", "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229"),
    "validation": ("validation.parquet", "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b"),
    "test": ("test.parquet", "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1"),
}
VARIANTS = ("full-finetune", "lora-r8-alpha16")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--preflight", "--dry-run", action="store_true", dest="preflight")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_paths(verify: bool = True) -> dict[str, str]:
    root = Path(os.environ["AXIOM_DATASET_DIR"]) / "sst2"
    paths = {split: root / name for split, (name, _) in DATA_FILES.items()}
    for split, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing {split} dataset: {path}")
        expected = DATA_FILES[split][1]
        if verify and sha256(path) != expected:
            raise ValueError(f"checksum mismatch for {path}")
    return {split: str(path) for split, path in paths.items()}


def replace_with_lora(model) -> None:
    import torch
    import loralib as lora

    for layer in model.roberta.encoder.layer:
        attention = layer.attention.self
        for name in ("query", "value"):
            original = getattr(attention, name)
            replacement = lora.Linear(
                original.in_features,
                original.out_features,
                r=8,
                lora_alpha=16,
                bias=original.bias is not None,
            )
            with torch.no_grad():
                replacement.weight.copy_(original.weight)
                if original.bias is not None:
                    replacement.bias.copy_(original.bias)
            setattr(attention, name, replacement)
    lora.mark_only_lora_as_trainable(model, bias="lora_only")
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def train(args: argparse.Namespace) -> tuple[dict[str, float | int], Path]:
    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("matrix training requires CUDA")
    set_seed(args.seed)
    paths = dataset_paths()
    dataset = load_dataset("parquet", data_files=paths)
    tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base", revision=MODEL_REVISION)
    model = AutoModelForSequenceClassification.from_pretrained(
        "FacebookAI/roberta-base", revision=MODEL_REVISION, num_labels=2
    )
    if args.variant == "lora-r8-alpha16":
        replace_with_lora(model)

    def tokenize(batch):
        return tokenizer(batch["sentence"], truncation=True, max_length=128)

    encoded = dataset.map(tokenize, batched=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    checkpoint = Path(os.environ.get("AXIOM_CHECKPOINT_DIR", "checkpoints")) / args.variant / str(args.seed)
    checkpoint.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(checkpoint), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate, weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, load_best_model_at_end=True, metric_for_best_model="accuracy",
        seed=args.seed, data_seed=args.seed, report_to=[], fp16=True,
    )

    def compute_metrics(prediction):
        return {"accuracy": float((np.argmax(prediction.predictions, axis=-1) == prediction.label_ids).mean())}

    trainer = Trainer(
        model=model, args=training_args, train_dataset=encoded["train"],
        eval_dataset=encoded["validation"], processing_class=tokenizer, compute_metrics=compute_metrics,
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer.train()
    accuracy = float(trainer.evaluate()["eval_accuracy"])
    elapsed = time.monotonic() - started
    trainer.save_model(str(checkpoint / "final"))
    return {
        "validation_accuracy": accuracy,
        "trainable_parameter_count": trainable,
        "wall_time_seconds": elapsed,
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }, checkpoint / "final"


def preflight(args: argparse.Namespace) -> tuple[dict[str, float | int], Path]:
    dataset_paths()
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained("FacebookAI/roberta-base", revision=MODEL_REVISION, local_files_only=True)
    if config.model_type != "roberta":
        raise ValueError("model cache does not contain RoBERTa")
    artifact = Path(os.environ["AXIOM_ENVIRONMENT_DIR"]) / "preflight-checkpoint.json"
    artifact.write_text(json.dumps({"variant": args.variant, "seed": args.seed}) + "\n")
    return {
        "validation_accuracy": 0.0,
        "trainable_parameter_count": 1,
        "wall_time_seconds": 0.0,
        "peak_vram_bytes": 0,
    }, artifact


def emit_result(variant: str, seed: int, metrics: dict[str, float | int], artifact: Path) -> None:
    result_path = Path(os.environ["AXIOM_RESULT_PATH"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    document = {"schema_version": "result.json-v1", "results": [], "artifacts": [], "environment": {}}
    if result_path.is_file():
        previous = json.loads(result_path.read_text())
        if previous.get("schema_version") != "result.json-v1":
            raise ValueError("existing AXIOM_RESULT_PATH has an invalid schema")
        document = previous
    row = {"variant": variant, "seed": seed, **metrics}
    document["results"] = [item for item in document["results"] if item.get("variant") != variant]
    document["results"].append(row)
    artifact_row = {"variant": variant, "seed": seed, "path": str(artifact), "kind": "checkpoint"}
    document["artifacts"] = [item for item in document["artifacts"] if item.get("variant") != variant]
    document["artifacts"].append(artifact_row)
    document["environment"] = {
        "transport": "axiom-environment-json-v1", "python": sys.version.split()[0],
        "platform": platform.platform(), "base_revision": BASE_REVISION,
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{result_path.name}.", dir=result_path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(document, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, result_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print("AXIOM_METRICS: " + json.dumps({"variant": variant, **metrics}, sort_keys=True))


def main() -> None:
    args = parse_args()
    metrics, artifact = preflight(args) if args.preflight else train(args)
    emit_result(args.variant, args.seed, metrics, artifact)


if __name__ == "__main__":
    main()
