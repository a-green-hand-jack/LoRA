#!/usr/bin/env python3
"""Axiom runner for the matched RoBERTa-base SST-2 comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
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
DATA_FILES = {
    "train.parquet": "66a253e67968acfabcbe49dbe9da964b42ac1c851c40ab760e8c8942efdb3229",
    "validation.parquet": "a1371f3b3a7b0bcefa8388799a9359dc3ce76c349cc0079507a7991364fd2a9b",
    "test.parquet": "e9d23cf0067211d2baf018328b507f5153fb6704d75117295a8bda47c7adccb1",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--preflight", "--dry-run", action="store_true", dest="preflight")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=128)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_data(dataset_dir: Path) -> dict[str, Path]:
    paths = {name: dataset_dir / "sst2" / name for name in DATA_FILES}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"missing required dataset file: {path}")
        actual = sha256(path)
        if actual != DATA_FILES[name]:
            raise ValueError(f"checksum mismatch for {path}: {actual}")
    return paths


def configure_lora(model) -> None:
    import loralib as lora

    for layer in model.roberta.encoder.layer:
        attention = layer.attention.self
        for name in ("query", "value"):
            source = getattr(attention, name)
            target = lora.Linear(
                source.in_features,
                source.out_features,
                r=8,
                lora_alpha=16,
                lora_dropout=0.0,
                bias=source.bias is not None,
            )
            target.weight.data.copy_(source.weight.data)
            if source.bias is not None:
                target.bias.data.copy_(source.bias.data)
            setattr(attention, name, target)
    lora.mark_only_lora_as_trainable(model)
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True


def run_training(args: argparse.Namespace, data_paths: dict[str, Path], environment_dir: Path):
    import pyarrow.parquet as pq
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("a CUDA GPU is required for matrix training")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model_dir = environment_dir / "models" / "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2, local_files_only=True
    )
    if args.variant == "lora-r8-alpha16":
        configure_lora(model)

    class SST2Dataset(Dataset):
        def __init__(self, path: Path):
            table = pq.read_table(path, columns=["sentence", "label"])
            self.sentences = table.column("sentence").to_pylist()
            self.labels = table.column("label").to_pylist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            return self.sentences[index], self.labels[index]

    def collate(rows):
        sentences, labels = zip(*rows)
        batch = tokenizer(
            list(sentences), padding=True, truncation=True,
            max_length=args.max_length, return_tensors="pt"
        )
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        SST2Dataset(data_paths["train.parquet"]), batch_size=args.batch_size,
        shuffle=True, generator=generator, collate_fn=collate, num_workers=2,
    )
    validation_loader = DataLoader(
        SST2Dataset(data_paths["validation.parquet"]), batch_size=args.batch_size * 2,
        shuffle=False, collate_fn=collate, num_workers=2,
    )
    device = torch.device("cuda")
    model.to(device)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate
    )
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.train()
    for _ in range(args.epochs):
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            model(**batch).loss.backward()
            optimizer.step()
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch in validation_loader:
            labels = batch.pop("labels").to(device)
            logits = model(**{key: value.to(device) for key, value in batch.items()}).logits
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += labels.numel()
    elapsed = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated(device))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    checkpoint_dir = Path(os.environ["AXIOM_RESULT_PATH"]).parent / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{args.variant}-seed-{args.seed}.pt"
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if args.variant == "full-finetune" or "lora_" in name or name.startswith("classifier.")}
    torch.save(state, checkpoint)
    return correct / total, trainable, elapsed, peak_vram, checkpoint


def merge_result(row: dict, artifact: dict, environment: dict) -> None:
    result_path = Path(os.environ["AXIOM_RESULT_PATH"])
    result_path.parent.mkdir(parents=True, exist_ok=True)
    prior = None
    if result_path.exists():
        prior = json.loads(result_path.read_text())
        if prior.get("schema") != "result.json-v1":
            raise ValueError("existing AXIOM_RESULT_PATH has the wrong schema")
    results = [] if prior is None else prior["results"]
    results = [item for item in results if item.get("variant") != row["variant"]]
    results.append(row)
    artifacts = [] if prior is None else prior["artifacts"]
    artifacts = [item for item in artifacts if item.get("variant") != row["variant"]]
    artifacts.append(artifact)
    payload = {
        "schema": "result.json-v1",
        "results": results,
        "artifacts": artifacts,
        "environment": environment if prior is None else prior["environment"],
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{result_path.name}.", dir=result_path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(payload, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, result_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    args = parse_args()
    dataset_dir = Path(os.environ["AXIOM_DATASET_DIR"])
    environment_dir = Path(os.environ["AXIOM_ENVIRONMENT_DIR"])
    data_paths = validate_data(dataset_dir)
    variant_key = f"{args.variant}:seed={args.seed}"
    environment = {
        "python": os.sys.version.split()[0],
        "base_revision": "c4593f060e6a368d7bb5af5273b8e42810cdef90",
        "dataset_dir": str(dataset_dir),
        "environment_dir": str(environment_dir),
    }
    if args.preflight:
        metrics = dict(zip(METRICS, (0.0, 1, 0.0, 0)))
        artifact = {"variant": variant_key, "kind": "preflight", "path": str(data_paths["validation.parquet"])}
    else:
        accuracy, trainable, elapsed, peak_vram, checkpoint = run_training(
            args, data_paths, environment_dir
        )
        metrics = dict(zip(METRICS, (accuracy, trainable, elapsed, peak_vram)))
        artifact = {"variant": variant_key, "kind": "checkpoint", "path": str(checkpoint)}
    row = {"variant": variant_key, **metrics}
    merge_result(row, artifact, environment)
    print("AXIOM_METRICS: " + json.dumps(row, sort_keys=True))


if __name__ == "__main__":
    main()
