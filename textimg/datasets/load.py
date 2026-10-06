"""Load either the original Hub datasets or local parquet fixtures."""
from pathlib import Path

TASKS = ("identical", "multilingual", "reasoning", "context_reasoning", "multiple_choice")


def load_data(args, name, config=None, **kwargs):
    from datasets import load_dataset, DatasetDict
    if getattr(args, "local_data", None) and name == args.dataset and config in TASKS:
        base = Path(args.local_data) / config
        if config == "multilingual":
            files = {lang: str(base / lang / "wikipedia_truncate_by_token.parquet")
                     for lang in args.languages}
        elif config == "multiple_choice":
            files = {split: str(base / f"multiple_choice_{split}.parquet")
                     for split in args.mc_difficulty}
        else:
            filenames = {"identical": "wikitext_truncate_by_word.parquet",
                         "reasoning": "math_sample.parquet",
                         "context_reasoning": "context_reasoning_sample.parquet"}
            files = {"train": str(base / filenames[config])}
        result = load_dataset("parquet", data_files=files)
    else:
        result = load_dataset(name, config, **kwargs)
    fingerprints = getattr(args, "dataset_fingerprints", {})
    if hasattr(result, "items"):
        for split, ds in result.items():
            fingerprints[f"{name}/{config}/{split}"] = getattr(ds, "_fingerprint", None)
    else:
        fingerprints[f"{name}/{config}/{kwargs.get('split', '')}"] = getattr(result, "_fingerprint", None)
    args.dataset_fingerprints = fingerprints
    # LLM smoke runs are bounded before any model call. Evaluation must retain the
    # complete ID lookup, so its limit is applied to image files instead.
    if args.command == "llm" and args.limit is not None:
        result = DatasetDict({key: ds.select(range(min(args.limit, len(ds))))
                              for key, ds in result.items()})
    return result
