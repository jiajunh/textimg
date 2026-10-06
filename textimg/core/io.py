"""File helpers. Importing this module never runs an experiment."""
import json
from pathlib import Path


def load_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def image_files(directory, args):
    # Ignore filesystem metadata and scratch files; preserve legacy os.listdir order.
    import os
    files = [name for name in os.listdir(directory)
             if Path(name).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
             and Path(name).stem.isdigit() and (Path(directory) / name).is_file()]
    return files if args.limit is None else files[:args.limit]


def limit_rows(data, args):
    return data if args.limit is None else data.select(range(min(args.limit, len(data))))
