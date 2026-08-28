#!/usr/bin/env python3
"""Reproducible RoBERTa SST-2 full-finetune/LoRA experiment runner."""

import argparse
import hashlib
import json
import os
import platform
import random
import sys
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
DIGESTS = {
    "train.parquet": "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    "validation.parquet": "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
    "test.parquet": "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1",
}


def atomic_merge_result(path, row, artifact, environment):
    """Merge one matrix row and atomically publish result.json-v1."""
    target = Path(path)
    previous = {}
    if target.exists():
        previous = json.loads(target.read_text(encoding="utf-8"))
        if previous.get("schema_version") != "result.json-v1":
            raise ValueError("existing result has an incompatible schema")
    results = list(previous.get("results", []))
    key = (row["variant"], row["seed"])
    results = [item for item in results if (item.get("variant"), item.get("seed")) != key]
    results.append(row)
    artifacts = list(previous.get("artifacts", []))
    if artifact not in artifacts:
        artifacts.append(artifact)
    payload = {
        "schema_version": "result.json-v1",
        "results": results,
        "artifacts": artifacts,
        "environment": previous.get("environment") or environment,
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=target.name + ".", dir=target.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_dataset(root):
    sst2 = Path(root) / "sst2"
    for name, expected in DIGESTS.items():
        path = sst2 / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"checksum mismatch for {path}: {actual}")
    return sst2


def install_lora(model):
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
            replacement.weight.data.copy_(original.weight.data)
            if original.bias is not None:
                replacement.bias.data.copy_(original.bias.data)
            setattr(attention, name, replacement)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for name, parameter in model.named_parameters():
        if "lora_" in name or name.startswith("classifier."):
            parameter.requires_grad = True


def run_training(args, dataset_root):
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model_root = Path(os.environ["AXIOM_ENVIRONMENT_DIR"]) / "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_root, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_root, num_labels=2, local_files_only=True)
    if args.variant == "lora-r8-alpha16":
        install_lora(model)

    class SST2(Dataset):
        def __init__(self, path):
            table = pq.read_table(path, columns=["sentence", "label"])
            self.sentences = table["sentence"].to_pylist()
            self.labels = table["label"].to_pylist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            encoded = tokenizer(self.sentences[index], truncation=True, max_length=128)
            encoded["labels"] = self.labels[index]
            return encoded

    collator = __import__("transformers").DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(SST2(dataset_root / "train.parquet"), batch_size=32, shuffle=True,
                              generator=generator, collate_fn=collator, num_workers=2)
    validation_loader = DataLoader(SST2(dataset_root / "validation.parquet"), batch_size=64,
                                   collate_fn=collator, num_workers=2)
    device = torch.device("cuda")
    model.to(device)
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    learning_rate = 5e-4 if args.variant == "lora-r8-alpha16" else 2e-5
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate,
                                  weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    model.train()
    for _ in range(3):
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in validation_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            predictions = model(**batch).logits.argmax(dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.numel()
    wall_time = time.monotonic() - started
    peak_vram = torch.cuda.max_memory_allocated()
    checkpoint = Path(args.output_dir) / args.variant / str(args.seed) / "checkpoint.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if args.variant == "full-finetune" or "lora_" in name or name.startswith("classifier.")}
    torch.save({"variant": args.variant, "seed": args.seed, "state_dict": state}, checkpoint)
    return correct / total, trainable, wall_time, peak_vram, checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", default="axiom-checkpoints")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result_path = os.environ.get("AXIOM_RESULT_PATH")
    if not result_path:
        parser.error("AXIOM_RESULT_PATH is required")
    dataset_root = verify_dataset(os.environ["AXIOM_DATASET_DIR"])
    if args.preflight:
        values = (0.0, 1, 0.0, 0)
        checkpoint = Path(args.output_dir) / "preflight" / f"{args.variant}-{args.seed}.json"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text(json.dumps({"variant": args.variant, "seed": args.seed}), encoding="utf-8")
    else:
        training_result = run_training(args, dataset_root)
        values = training_result[:4]
        checkpoint = training_result[4]
    row = {"variant": args.variant, "seed": args.seed, **dict(zip(METRICS, values))}
    environment = {"python": sys.version.split()[0], "platform": platform.platform(),
                   "code_revision": "c4593f060e6a368d7bb5af5273b8e42810cdef90"}
    artifact = {"type": "checkpoint", "path": str(checkpoint), "variant": args.variant, "seed": args.seed}
    atomic_merge_result(result_path, row, artifact, environment)
    print("AXIOM_METRICS: " + json.dumps({"variant": args.variant, **{key: row[key] for key in METRICS}}, sort_keys=True))


if __name__ == "__main__":
    main()
