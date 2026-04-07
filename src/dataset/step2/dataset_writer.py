"""
Dataset Writer: split train/val/test dan simpan ke JSONL + dataset_info.json.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Dict, List

from src.dataset.step2.validator import ValidDataPoint


def _write_jsonl(datapoints: List[ValidDataPoint], filepath: Path) -> None:
    """Tulis list ValidDataPoint ke file JSONL (satu JSON object per baris)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for dp in datapoints:
            record = {
                "input": dp.input,
                "target": dp.target,
                "metadata": dp.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _load_jsonl(filepath: Path) -> List[dict]:
    """Load JSONL file dan kembalikan list of dict."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stratified_split(
    datapoints: List[ValidDataPoint],
    train_ratio: float,
    val_ratio: float,
    stratify_by: str,
    seed: int = 42,
) -> tuple[List[ValidDataPoint], List[ValidDataPoint], List[ValidDataPoint]]:
    """
    Split data dengan stratifikasi berdasarkan field metadata.
    Memastikan setiap split mengandung semua nilai stratify_by yang ada.
    """
    rng = random.Random(seed)

    # Kelompokkan berdasarkan nilai stratify_by
    groups: Dict[str, List[ValidDataPoint]] = defaultdict(list)
    for dp in datapoints:
        key = dp.metadata.get(stratify_by, "unknown")
        groups[key].append(dp)

    train_all, val_all, test_all = [], [], []

    for key, group in groups.items():
        rng.shuffle(group)
        n = len(group)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        # test mendapat sisanya, minimal 1 jika group >= 3
        n_test = max(0, n - n_train - n_val)

        # Jika group terlalu kecil, pastikan minimal ada di train
        if n == 1:
            train_all.extend(group)
        elif n == 2:
            train_all.append(group[0])
            val_all.append(group[1])
        else:
            train_all.extend(group[:n_train])
            val_all.extend(group[n_train:n_train + n_val])
            test_all.extend(group[n_train + n_val:])

    # Shuffle final splits
    rng.shuffle(train_all)
    rng.shuffle(val_all)
    rng.shuffle(test_all)

    return train_all, val_all, test_all


def _compute_distribution(datapoints: List[ValidDataPoint], field: str) -> Dict[str, int]:
    """Hitung distribusi nilai field metadata."""
    dist: Dict[str, int] = defaultdict(int)
    for dp in datapoints:
        key = dp.metadata.get(field, "unknown")
        dist[str(key)] += 1
    return dict(sorted(dist.items()))


def write_dataset(
    datapoints: List[ValidDataPoint],
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_by: str = "difficulty",
    seed: int = 42,
) -> dict:
    """
    Split dan simpan dataset ke JSONL files.

    Returns:
        dict: dataset_info yang disimpan ke dataset_info.json
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train_ratio + val_ratio + test_ratio harus = 1.0"

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train, val, test = _stratified_split(
        datapoints, train_ratio, val_ratio, stratify_by, seed
    )

    _write_jsonl(train, out_path / "train.jsonl")
    _write_jsonl(val, out_path / "validation.jsonl")
    _write_jsonl(test, out_path / "test.jsonl")

    # Hitung statistik
    total = len(datapoints)
    info = {
        "total": total,
        "splits": {
            "train": len(train),
            "validation": len(val),
            "test": len(test),
        },
        "concept_distribution": _compute_distribution(datapoints, "concept"),
        "difficulty_distribution": _compute_distribution(datapoints, "difficulty"),
        "source_distribution": _compute_distribution(datapoints, "source"),
        "question_type_distribution": _compute_distribution(datapoints, "question_type"),
        "generated_at": str(date.today()),
    }

    info_path = out_path / "dataset_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return info


def load_jsonl_split(filepath: str) -> List[dict]:
    """
    Load satu split JSONL dan kembalikan list of dict.
    Berguna untuk verifikasi dan fine-tuning.
    """
    return _load_jsonl(Path(filepath))
