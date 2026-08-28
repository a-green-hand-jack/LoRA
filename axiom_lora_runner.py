#!/usr/bin/env python3
"""Reproducible SST-2 full fine-tuning/LoRA experiment runner for Axiom."""

import argparse
import hashlib
import json
import os
import platform
import tempfile
import time
from pathlib import Path


VARIANTS = {"full-finetune", "lora-r8-alpha16"}
METRICS = (
    "validation_accuracy",
    "trainable_parameter_count",
    "wall_time_seconds",
    "peak_vram_bytes",
)
DATA_DIGESTS = {
    "train.parquet": "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    "validation.parquet": "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
    "test.parquet": "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1",
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-id", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", required=True, type=int, choices=(13, 21, 42))
    parser.add_argument("--dataset-dir", default=os.environ.get("AXIOM_DATASET_DIR"))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if not args.dataset_dir:
        parser.error("--dataset-dir or AXIOM_DATASET_DIR is required")
    if not os.environ.get("AXIOM_RESULT_PATH"):
        parser.error("AXIOM_RESULT_PATH is required")
    return args


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_dataset(dataset_dir):
    root = Path(dataset_dir) / "sst2"
    for name, expected in DATA_DIGESTS.items():
        path = root / name
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"dataset verification failed: {path}")
    return root


def atomic_merge_result(row, artifact, environment):
    path = Path(os.environ["AXIOM_RESULT_PATH"])
    payload = {"transport": "result.json-v1", "results": [], "artifacts": [], "environment": environment}
    if path.is_file():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("transport") != "result.json-v1":
            raise RuntimeError("existing AXIOM_RESULT_PATH has an invalid transport")
        payload = prior
    key = (row["variant"], row["seed"])
    payload["results"] = [
        old for old in payload.get("results", [])
        if (old.get("variant"), old.get("seed")) != key
    ] + [row]
    artifacts = payload.get("artifacts", [])
    if artifact not in artifacts:
        artifacts.append(artifact)
    payload["artifacts"] = artifacts
    payload["environment"] = payload.get("environment") or environment
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def environment_record(torch=None):
    record = {"python": platform.python_version(), "platform": platform.platform()}
    if torch is not None:
        record.update({"torch": torch.__version__, "cuda": torch.version.cuda or "none"})
    return record


def emit(args, metrics, artifact, torch=None):
    row = {"variant": args.variant_id, "seed": args.seed, **metrics}
    atomic_merge_result(row, artifact, environment_record(torch))
    print("AXIOM_METRICS: " + json.dumps(row, sort_keys=True), flush=True)


def run_preflight(args):
    verify_dataset(args.dataset_dir)
    artifact = {"kind": "preflight", "path": "axiom_lora_runner.py"}
    emit(args, {metric: 0 for metric in METRICS}, artifact)


def run_training(args):
    import torch
    import loralib as lora
    from datasets import DatasetDict, load_dataset
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

    started = time.monotonic()
    data_root = verify_dataset(args.dataset_dir)
    model_root = Path(os.environ["AXIOM_ENVIRONMENT_DIR"]) / "models" / "roberta-base"
    output = Path(args.output_dir or (Path.cwd() / "outputs" / args.variant_id / str(args.seed)))
    output.mkdir(parents=True, exist_ok=True)
    files = {split: str(data_root / f"{split}.parquet") for split in ("train", "validation", "test")}
    datasets = load_dataset("parquet", data_files=files)
    config = AutoConfig.from_pretrained(str(model_root), num_labels=2, finetuning_task="sst2")
    apply_lora = args.variant_id == "lora-r8-alpha16"
    config.apply_lora = apply_lora
    config.lora_r = 8 if apply_lora else None
    config.lora_alpha = 16 if apply_lora else None
    tokenizer = AutoTokenizer.from_pretrained(str(model_root), use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_root), config=config)
    if apply_lora:
        for layer in model.roberta.encoder.layer:
            for projection_name in ("query", "value"):
                original = getattr(layer.attention.self, projection_name)
                replacement = lora.Linear(
                    original.in_features, original.out_features, r=8, lora_alpha=16,
                    bias=original.bias is not None,
                ).to(device=original.weight.device, dtype=original.weight.dtype)
                replacement.weight.data.copy_(original.weight.data)
                if original.bias is not None:
                    replacement.bias.data.copy_(original.bias.data)
                setattr(layer.attention.self, projection_name, replacement)
        for name, parameter in model.named_parameters():
            parameter.requires_grad = not name.startswith("roberta") or "lora" in name
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def tokenize(batch):
        return tokenizer(batch["sentence"], padding="max_length", truncation=True, max_length=128)

    tokenized = DatasetDict({key: value.map(tokenize, batched=True) for key, value in datasets.items()})

    def compute_metrics(prediction):
        predictions = prediction.predictions.argmax(axis=-1)
        return {"accuracy": float((predictions == prediction.label_ids).mean())}

    training_args = TrainingArguments(
        output_dir=str(output), overwrite_output_dir=True, do_train=True, do_eval=True,
        num_train_epochs=3, per_device_train_batch_size=32, per_device_eval_batch_size=64,
        learning_rate=5e-5 if not apply_lora else 5e-4, weight_decay=0.1,
        warmup_ratio=0.06, evaluation_strategy="epoch", save_strategy="epoch",
        save_total_limit=1, seed=args.seed, data_seed=args.seed, fp16=True,
        logging_steps=100, report_to=[], load_best_model_at_end=True, metric_for_best_model="accuracy",
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"], tokenizer=tokenizer, compute_metrics=compute_metrics,
    )
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    evaluation = trainer.evaluate()
    checkpoint = output / "final"
    trainer.save_model(str(checkpoint))
    metrics = {
        "validation_accuracy": float(evaluation["eval_accuracy"]),
        "trainable_parameter_count": int(trainable),
        "wall_time_seconds": float(time.monotonic() - started),
        "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
    }
    emit(args, metrics, {"kind": "checkpoint", "path": str(checkpoint)}, torch)


def main(argv=None):
    args = parse_args(argv)
    run_preflight(args) if args.preflight else run_training(args)


if __name__ == "__main__":
    main()
