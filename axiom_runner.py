#!/usr/bin/env python3
"""Reproducible SST-2 full fine-tuning/LoRA matrix runner for Axiom."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import sys
import time
from pathlib import Path


VARIANTS = {"full-finetune": False, "lora-r8-alpha16": True}
METRICS = ("validation_accuracy", "trainable_parameter_count", "wall_time_seconds", "peak_vram_bytes")
EXPECTED = {
    "train.parquet": "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    "validation.parquet": "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
    "test.parquet": "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1",
}


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--preflight", "--dry-run", dest="preflight", action="store_true")
    return parser.parse_args()


def dataset_files():
    root = Path(os.environ["AXIOM_DATASET_DIR"]) / "sst2"
    files = {name: root / name for name in EXPECTED}
    for name, path in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != EXPECTED[name]:
            raise RuntimeError(f"checksum mismatch for {name}: {digest}")
    return files


def merge_result(row, checkpoint):
    target = Path(os.environ["AXIOM_RESULT_PATH"])
    prior = {}
    if target.exists():
        prior = json.loads(target.read_text())
        if prior.get("transport") != "result.json-v1":
            raise RuntimeError("existing result has an incompatible transport")
    results = [item for item in prior.get("results", []) if item.get("variant") != row["variant"]]
    results.append(row)
    artifacts = prior.get("artifacts", [])
    artifact = {"variant": row["variant"], "path": str(checkpoint), "kind": "checkpoint"}
    artifacts = [item for item in artifacts if item.get("variant") != row["variant"]] + [artifact]
    environment = prior.get("environment") or {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "torch": importlib.metadata.version("torch") if importlib.util.find_spec("torch") else "unavailable",
        "cuda": os.environ.get("CUDA_VERSION", "none"),
    }
    payload = {"transport": "result.json-v1", "results": results, "artifacts": artifacts, "environment": environment}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
    os.replace(temporary, target)


def train(args, files):
    import numpy as np
    import torch
    from datasets import DatasetDict, load_dataset
    from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, set_seed

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    data = DatasetDict({split: load_dataset("parquet", data_files=str(files[f"{split}.parquet"]), split="train") for split in ("train", "validation")})
    model_ref = os.environ.get("AXIOM_MODEL_DIR", "FacebookAI/roberta-base")
    config = AutoConfig.from_pretrained(model_ref, num_labels=2)
    config.apply_lora = VARIANTS[args.variant]
    config.lora_r = 8
    config.lora_alpha = 16
    model = AutoModelForSequenceClassification.from_pretrained(model_ref, config=config)
    if VARIANTS[args.variant]:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = "lora" in name or "classifier" in name
    tokenizer = AutoTokenizer.from_pretrained(model_ref, use_fast=True)
    encoded = data.map(lambda batch: tokenizer(batch["sentence"], truncation=True, max_length=128), batched=True)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    output = Path(os.environ["AXIOM_RESULT_PATH"]).parent / "checkpoints" / f"{args.variant}-seed-{args.seed}"
    options = TrainingArguments(output_dir=str(output), overwrite_output_dir=True, num_train_epochs=3,
        per_device_train_batch_size=32, per_device_eval_batch_size=64, learning_rate=2e-5,
        weight_decay=0.1, seed=args.seed, save_strategy="no", logging_strategy="steps", logging_steps=100,
        report_to=[], fp16=torch.cuda.is_available())
    trainer = Trainer(model=model, args=options, train_dataset=encoded["train"], eval_dataset=encoded["validation"], tokenizer=tokenizer,
        compute_metrics=lambda prediction: {"accuracy": float((np.argmax(prediction.predictions, axis=1) == prediction.label_ids).mean())})
    started = time.monotonic()
    trainer.train()
    accuracy = float(trainer.evaluate()["eval_accuracy"])
    elapsed = time.monotonic() - started
    output.mkdir(parents=True, exist_ok=True)
    if VARIANTS[args.variant]:
        torch.save({name: value.cpu() for name, value in model.state_dict().items() if "lora" in name or "classifier" in name}, output / "adapter_model.bin")
        checkpoint = output / "adapter_model.bin"
    else:
        trainer.save_model(str(output))
        checkpoint = output / "pytorch_model.bin"
        if not checkpoint.exists():
            checkpoint = output / "model.safetensors"
    peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    return accuracy, trainable, elapsed, peak, checkpoint


def main():
    args = arguments()
    files = dataset_files()
    if args.preflight:
        checkpoint = Path(os.environ["AXIOM_RESULT_PATH"]).parent / "preflight" / f"{args.variant}-seed-{args.seed}.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"variant": args.variant, "seed": args.seed}) + "\n")
        values = (0.0, 1, 0.0, 0)
    else:
        values = train(args, files)
        checkpoint = values[-1]
        values = values[:-1]
    row = {"variant": args.variant, "seed": args.seed, **dict(zip(METRICS, values))}
    merge_result(row, checkpoint)
    print("AXIOM_METRICS: " + json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
