"""Small reproducibility sidecars; never serialize API keys or model objects."""
from datetime import datetime, timezone
from importlib.metadata import version, PackageNotFoundError
import hashlib
import json
import platform
from pathlib import Path
from uuid import uuid4


def start(args):
    if not args.manifest_dir:
        args.records_path = None
        return None, {}
    serializable = {key: value for key, value in vars(args).items()
                    if value is None or isinstance(value, (str, int, float, bool, list))}
    packages = {}
    for name in ["datasets", "openai", "google-genai", "diffusers", "transformers", "torch", "paddleocr"]:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    base = Path(args.manifest_dir)
    base.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    path = base / f"{run_id}.json"
    args.records_path = str(base / f"{run_id}.generation.jsonl")
    source_hash = hashlib.sha256()
    for p in sorted(Path(__file__).resolve().parents[1].rglob("*.py")):
        source_hash.update(str(p.relative_to(Path(__file__).resolve().parents[1])).encode())
        source_hash.update(p.read_bytes())
    record = {"run_id": run_id, "started_at": datetime.now(timezone.utc).isoformat(),
              "status": "running", "arguments": serializable, "packages": packages,
              "python": platform.python_version(), "code_sha256": source_hash.hexdigest(),
              "judge_reasoning": "provider model default" if args.judge_provider == "gemini"
                                 else "legacy low effort for scoring; default for extraction",
              "generation_records": args.records_path if args.command == "generate" else None}
    if args.command == "generate":
        from textimg.generation.registry import generation_settings
        record["generation_settings"] = generation_settings(args)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return path, record


def finish(path, record, *, status, summary=None, error=None):
    if path is None:
        return
    record.update(status=status, finished_at=datetime.now(timezone.utc).isoformat())
    if summary is not None:
        record["summary"] = summary
    if error is not None:
        record["error_type"] = type(error).__name__
    path.write_text(json.dumps(record, indent=2) + "\n")
