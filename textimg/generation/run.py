"""Generation orchestration; original prompts and provider defaults are retained."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from .tasks import generation_jobs
from .registry import get_generator_fn


def generate_one(gen_fn, args, sample_id, prompt, text, path):
    result = {"id": sample_id, "path": str(path),
              "input_sha256": hashlib.sha256(f"{prompt}\n\n{text}".encode()).hexdigest()}
    if args.resume and path.is_file():
        result["status"] = "existing"
        return result
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        image = gen_fn(args, prompt, text)
        if image is None:
            result["status"] = "no_image"
            return result
        if isinstance(image, bytes):
            path.write_bytes(image)
        elif hasattr(image, "save"):
            image.save(str(path))
        else:
            raise TypeError(f"Unsupported image response: {type(image).__name__}")
        result["status"] = "saved"
    except Exception as e:
        # Failure coverage is retained without printing credentials or provider payloads.
        message = str(e)
        status = "moderation_blocked" if (
            "moderation_blocked" in message or "safety system" in message) else "failed"
        result.update(status=status, error_type=type(e).__name__)
    return result


def run(args):
    from tqdm import tqdm
    generator = get_generator_fn(args)
    jobs = list(generation_jobs(args))
    counts = {}
    with open(args.records_path, "w", encoding="utf-8") as log, ThreadPoolExecutor(
            max_workers=args.num_workers) as executor:
        futures = [executor.submit(generate_one, generator, args, *job) for job in jobs]
        for future in tqdm(as_completed(futures), total=len(futures), desc=args.model):
            result = future.result()
            counts[result["status"]] = counts.get(result["status"], 0) + 1
            log.write(json.dumps(result, ensure_ascii=False) + "\n")
            log.flush()
    print(json.dumps({"attempted": len(jobs), **counts}, indent=2))
    return {"attempted": len(jobs), **counts}
