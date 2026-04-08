"""
Merge semua dataset AQG dari output_modul ke dataset-task-specific.
Gabungkan accumulated.jsonl dari semua modul, lalu split dengan stratifikasi
per modul + per difficulty.
"""
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_jsonl(filepath: str) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: list[dict], filepath: str) -> None:
    """Save list of dict to JSONL file."""
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def merge_all_accumulated(output_modul_dir: str) -> list[dict]:
    """Gabungkan semua accumulated.jsonl dari semua modul."""
    all_data = []
    modul_dirs = sorted(Path(output_modul_dir).iterdir())
    
    for modul_dir in modul_dirs:
        if not modul_dir.is_dir():
            continue
        acc_file = modul_dir / "accumulated.jsonl"
        if acc_file.exists():
            data = load_jsonl(str(acc_file))
            print(f"  {modul_dir.name}: {len(data)} entries")
            all_data.extend(data)
    
    return all_data


def stratified_split(
    data: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify_keys: list[str] = ["module_name", "difficulty"],
    seed: int = 42
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Split data dengan stratifikasi berdasarkan kombinasi module_name + difficulty.
    Memastikan setiap kombinasi terwakili di semua split.
    """
    random.seed(seed)
    
    # Group data by stratification key combination
    groups = defaultdict(list)
    for item in data:
        meta = item.get("metadata", {})
        # Extract module_name from source_file
        source_file = meta.get("source_file", "")
        if "materi/" in source_file:
            parts = source_file.split("materi/")[1].split("/")
            module_name = parts[0] if parts else "unknown"
        else:
            module_name = "unknown"
        
        difficulty = meta.get("difficulty", "unknown")
        key = (module_name, difficulty)
        groups[key].append(item)
    
    train, val, test = [], [], []
    
    for key, items in groups.items():
        random.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        
        # Pastikan minimal 1 di setiap split jika memungkinkan
        if n >= 3:
            train.extend(items[:n_train])
            val.extend(items[n_train:n_train + n_val])
            test.extend(items[n_train + n_val:])
        elif n == 2:
            train.append(items[0])
            test.append(items[1])
        else:  # n == 1
            train.append(items[0])
    
    # Shuffle final splits
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)
    
    return train, val, test


def compute_statistics(data: list[dict]) -> dict:
    """Hitung statistik dataset."""
    stats = {
        "total": len(data),
        "difficulty_distribution": defaultdict(int),
        "concept_distribution": defaultdict(int),
        "module_distribution": defaultdict(int),
        "question_type_distribution": defaultdict(int),
        "source_distribution": defaultdict(int),
    }
    
    for item in data:
        meta = item.get("metadata", {})
        stats["difficulty_distribution"][meta.get("difficulty", "unknown")] += 1
        stats["concept_distribution"][meta.get("concept", "unknown")] += 1
        
        # Extract module_name from source_file path
        source_file = meta.get("source_file", "")
        if "materi/" in source_file:
            parts = source_file.split("materi/")[1].split("/")
            module_name = parts[0] if parts else "unknown"
        else:
            module_name = "unknown"
        stats["module_distribution"][module_name] += 1
        
        stats["question_type_distribution"][meta.get("question_type", "unknown")] += 1
        stats["source_distribution"][meta.get("source", "unknown")] += 1
    
    # Convert defaultdict to dict for JSON serialization
    for key in stats:
        if isinstance(stats[key], defaultdict):
            stats[key] = dict(sorted(stats[key].items()))
    
    return stats


def main():
    output_modul_dir = "dataset_aqg/output_modul"
    output_dir = Path("dataset_aqg/dataset-task-spesifc")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MERGE DATASET AQG")
    print("=" * 60)
    
    # 1. Load semua data
    print("\n[1] Loading accumulated.jsonl from all modules...")
    all_data = merge_all_accumulated(output_modul_dir)
    print(f"\n  TOTAL: {len(all_data)} entries")
    
    # 2. Split dengan stratifikasi
    print("\n[2] Splitting with stratification (module + difficulty)...")
    train, val, test = stratified_split(all_data)
    print(f"  Train: {len(train)} ({len(train)/len(all_data)*100:.1f}%)")
    print(f"  Val:   {len(val)} ({len(val)/len(all_data)*100:.1f}%)")
    print(f"  Test:  {len(test)} ({len(test)/len(all_data)*100:.1f}%)")
    
    # 3. Save splits
    print("\n[3] Saving to dataset-task-specific/...")
    save_jsonl(all_data, str(output_dir / "accumulated.jsonl"))
    save_jsonl(train, str(output_dir / "train.jsonl"))
    save_jsonl(val, str(output_dir / "validation.jsonl"))
    save_jsonl(test, str(output_dir / "test.jsonl"))
    print(f"  Saved: accumulated.jsonl, train.jsonl, validation.jsonl, test.jsonl")
    
    # 4. Compute and save statistics
    print("\n[4] Computing statistics...")
    stats = {
        "total": len(all_data),
        "splits": {
            "train": len(train),
            "validation": len(val),
            "test": len(test)
        },
        "split_ratio": "70/15/15",
        "stratified_by": ["module_name", "difficulty"],
        "difficulty_distribution": compute_statistics(all_data)["difficulty_distribution"],
        "module_distribution": compute_statistics(all_data)["module_distribution"],
        "concept_distribution": compute_statistics(all_data)["concept_distribution"],
        "question_type_distribution": compute_statistics(all_data)["question_type_distribution"],
        "source_distribution": compute_statistics(all_data)["source_distribution"],
        "generated_at": datetime.now().strftime("%Y-%m-%d")
    }
    
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: dataset_info.json")
    
    # 5. Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total entries:    {stats['total']}")
    print(f"Train:            {stats['splits']['train']} (70%)")
    print(f"Validation:       {stats['splits']['validation']} (15%)")
    print(f"Test:             {stats['splits']['test']} (15%)")
    print(f"\nDifficulty distribution:")
    for k, v in stats["difficulty_distribution"].items():
        print(f"  {k}: {v}")
    print(f"\nModule distribution:")
    for k, v in stats["module_distribution"].items():
        print(f"  {k}: {v}")
    print("\nDone!")


if __name__ == "__main__":
    main()