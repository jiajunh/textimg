"""CLI entry point for text-in-image experiments."""
import argparse
from importlib import import_module
import json
from pathlib import Path

from .data_loading.load import TASKS
from .generation.registry import MODELS, EXACT_TEXT_ONLY, LOCAL_MODELS

BASE = Path(__file__).resolve().parents[1]


def add_common_args(subparser):
    subparser.add_argument("--dataset-type", "--dataset_type", choices=TASKS, default="identical")
    subparser.add_argument("--dataset", default="jiajunh/img_text")
    subparser.add_argument("--local-data", help="Directory containing the original local dataset subfolders")
    subparser.add_argument("--output-dir", "--output_dir", help="Output root; task/provider subfolders are added. For analyze: existing evaluation results root to read")
    subparser.add_argument("--img-dir", "--img_dir", default="", help="Model image directory, already including task/model")
    subparser.add_argument("--model", default="gpt-image-1.5")
    subparser.add_argument("--judge-provider", choices=["openai", "gemini"], default="openai")
    subparser.add_argument("--judge-model", "--judge_model", default=None)
    subparser.add_argument("--ocr-backend", "--ocr_backend", choices=["PaddleOCR", "DeepSeekOCR"], default="PaddleOCR")
    subparser.add_argument("--ocr-mode", choices=["legacy", "ablation"], default="legacy")
    subparser.add_argument("--deduplicate-steps", action="store_true", help="Opt-in exact step deduplication; legacy default is unchanged")
    # identical / multilingual: text lengths.
    subparser.add_argument("--text-length", "--text_length", nargs="+", type=int, default=[64, 128, 256, 512])
    # multilingual: languages.
    subparser.add_argument("--languages", nargs="+", choices=["ar", "en", "fr", "ja", "ko", "zh"], default=["ar", "en", "fr", "ja", "ko", "zh"])
    # reasoning: generation levels; evaluation reads available images.
    subparser.add_argument("--levels", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    # multiple_choice: difficulty splits.
    subparser.add_argument("--mc-difficulty", "--mc_difficulty", nargs="+", choices=["easy", "challenge"], default=["easy", "challenge"])
    # context_reasoning: no task-specific settings.

    limit = subparser.add_mutually_exclusive_group()
    limit.add_argument("--limit", type=int, help="Examples per group; default 1 for API/model workflows")
    limit.add_argument("--test", action="store_true", help="One example per group (same as --limit 1)")
    limit.add_argument("--full-run", action="store_true", help="Process every example")

    subparser.add_argument("--dry-run", action="store_true", help="Print configuration without reading data, loading models or calling APIs")
    subparser.add_argument("--max-judge-calls", type=int, help="Optional total cap on paid extraction/scoring calls")
    subparser.add_argument("--judge-call-delay", type=float, default=1.0, help="Seconds to wait between judge API calls (default: 1; 0 disables)")
    if subparser.prog.split()[-1] in {"evaluate", "analyze"}:
        subparser.add_argument("--resume", action="store_true", help="Append results and skip IDs already saved; analysis preprocessing skips saved scores")
    subparser.add_argument("--manifest-dir", help="Optional directory for run metadata and generation logs; disabled by default")


def add_generate_args(subparser):
    subparser.add_argument("--generation-call-delay", type=float, default=1.0, help="Minimum seconds between generation API calls (default: 1; 0 disables)")
    subparser.add_argument("--num-workers", "--num_workers", type=int, default=1)
    subparser.add_argument("--run-label", help="Output folder label; model sent to provider remains --model")
    subparser.add_argument("--resume", action="store_true", help="Skip existing images (records them separately)")
    subparser.add_argument("--quality", choices=["low", "medium", "high"], default="medium")
    subparser.add_argument("--device", choices=["cuda", "mps", "cpu"], default="cuda")
    subparser.add_argument("--model-path", help="Qwen-Image-2.1 local checkpoint or Hub ID")
    subparser.add_argument("--revision", help="Pin the Qwen-Image-2.1 checkpoint revision")
    subparser.add_argument("--cpu-offload", action="store_true")
    subparser.add_argument("--width", type=int, default=1024)
    subparser.add_argument("--height", type=int, default=1024)
    subparser.add_argument("--inference-steps", type=int, default=40)
    subparser.add_argument("--generation-seed", type=int)


def add_construct_args(subparser):
    subparser.add_argument("--data-size", "--data_size", type=int, default=300)
    subparser.add_argument("--seed", type=int, default=0)
    subparser.add_argument("--tokenizer-name", "--tokenizer_name", default="xlm-roberta-base")
    subparser.add_argument("--length-control-type", "--length_control_type", choices=["sentence", "word", "character", "token"], default="word")


def add_llm_args(subparser):
    subparser.add_argument("--llm", choices=["gpt-5.2", "qwen3-8b"], default="qwen3-8b")


def add_judge_args(subparser):
    subparser.add_argument("--render-metric", choices=["quality", "clear_ratio", "clear_area"], default="quality")


def add_analyze_args(subparser):
    subparser.add_argument("--tasks", nargs="+", choices=["identical", "reasoning", "context_reasoning", "multiple_choice"], default=["identical"])
    subparser.add_argument("--analysis-source", choices=["image", "llm"], default="image")
    subparser.add_argument("--preprocess", action="store_true", help="Run answer judges before aggregation")


def parser():
    p = argparse.ArgumentParser(description="Text-in-image experiments: preserved legacy algorithms, explicit run controls")
    sub = p.add_subparsers(dest="command", required=True)

    commands = ["construct", "generate", "evaluate", "llm", "judge", "analyze"]
    for command in commands:
        subparser = sub.add_parser(command)
        add_common_args(subparser)

        if command == "generate":
            add_generate_args(subparser)
        elif command == "construct":
            add_construct_args(subparser)
        elif command == "llm":
            add_llm_args(subparser)
        elif command == "judge":
            add_judge_args(subparser)
        elif command == "analyze":
            add_analyze_args(subparser)

    return p


def validate_limits(args, parser):
    if not 0 <= args.judge_call_delay < float("inf"):
        parser.error("--judge-call-delay must be finite and nonnegative")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_judge_calls is not None and args.max_judge_calls < 1:
        parser.error("--max-judge-calls must be positive")
    if args.deduplicate_steps and args.command not in {"evaluate", "llm"}:
        parser.error("--deduplicate-steps changes step scoring in evaluate/llm; analysis uses the stored step scores")


def resolve_limit(args):
    if args.command == "construct" or (args.command == "analyze" and not args.preprocess):
        if args.test or args.limit is not None or args.full_run:
            raise ValueError("Construction uses --data-size; aggregation reads all existing results. Limits apply to model workflows and analyze --preprocess")
        return None
    return None if args.full_run else (args.limit or 1)


def resolve_judge_model(args, parser):
    if args.judge_model is not None:
        return args.judge_model
    if args.judge_provider == "gemini":
        parser.error("For Gemini, explicitly choose a text/multimodal --judge-model available to your account")
    return "gpt-4.1" if args.command == "judge" else "gpt-5.2"


def validate_generate(args, parser):
    if not 0 <= args.generation_call_delay < float("inf"):
        parser.error("--generation-call-delay must be finite and nonnegative")
    if args.model not in MODELS:
        parser.error(f"Unknown image model. Choose from: {', '.join(MODELS)}")
    if args.model in EXACT_TEXT_ONLY and args.dataset_type != "identical":
        parser.error("Diffusion baseline adapters are restricted to the identical/exact-text ablation")
    if args.num_workers < 1:
        parser.error("--num-workers must be positive")
    if args.model in LOCAL_MODELS and args.num_workers != 1:
        parser.error("Local pipelines use --num-workers 1 to avoid concurrent use of the same model")
    if args.width < 1 or args.height < 1 or args.inference_steps < 1:
        parser.error("Image dimensions and inference steps must be positive")
    if args.model != "qwen-image-2.1" and (
        args.width != 1024 or args.height != 1024 or args.inference_steps != 40
        or args.model_path or args.revision or args.cpu_offload or args.generation_seed is not None
        or args.device != "cuda"
    ):
        parser.error("Device/checkpoint/resolution/seed flags currently configure Qwen-Image-2.1 only; legacy adapter defaults are preserved")
    if not args.model.startswith("gpt-image") and args.quality != "medium":
        parser.error("--quality configures GPT image generation only")


def validate_dataset_args(args, parser):
    if args.command in {"judge", "evaluate"} and not args.img_dir:
        parser.error("--img-dir is required")
    if args.command == "evaluate" and args.ocr_backend == "DeepSeekOCR":
        if args.dataset_type != "identical":
            parser.error("DeepSeek OCR is supported only for the identical-text ablation")
        args.ocr_mode = "ablation"
    if args.command == "llm" and args.dataset_type == "multilingual":
        parser.error("The legacy LLM experiments have no multilingual implementation")
    if args.command == "construct" and args.data_size < 1:
        parser.error("--data-size must be positive")


def resolve_output_dir(args):
    roots = {
        "generate": "generations",
        "evaluate": "results",
        "llm": "results",
        "construct": "data",
        "analyze": "results",
        "judge": "render_quality_results",
    }

    if args.command == "judge":
        roots["judge"] = {
            "quality": "render_quality_results",
            "clear_ratio": "render_clear_ratio",
            "clear_area": "render_clear_area",
        }[args.render_metric]

    output = Path(args.output_dir or BASE / roots[args.command]).resolve()
    if args.command != "analyze":
        output = output / args.dataset_type
        if args.command == "evaluate":
            output = output / (
                args.ocr_backend if args.dataset_type in {"identical", "multilingual"}
                else args.judge_model
            )
        elif args.command == "llm" and args.dataset_type in {"reasoning", "context_reasoning", "multiple_choice"}:
            output = output / args.judge_model
        elif args.command == "judge":
            output = output / args.judge_model
    return str(output)


def configure(args, parser):
    validate_limits(args, parser)

    try:
        args.limit = resolve_limit(args)
    except ValueError as exc:
        parser.error(str(exc))

    args.judge_model = resolve_judge_model(args, parser)

    if args.command == "generate":
        validate_generate(args, parser)

    validate_dataset_args(args, parser)
    args.output_dir = resolve_output_dir(args)
    return args


def execute(args):
    if args.command == "generate":
        from .generation.run import run
        return run(args)
    if args.command == "construct":
        from .data_loading import build
        if args.dataset_type == "multilingual":
            for lang in args.languages:
                (Path(args.output_dir) / lang).mkdir(parents=True, exist_ok=True)
        return getattr(build, f"construct_{args.dataset_type}_data")(args)
    if args.command == "evaluate":
        if args.dataset_type in {"identical", "multilingual"}:
            ocr = import_module("textimg.evaluation.ocr")
            args.backend = ocr.get_backend(args)
        mode = "ablation" if args.ocr_backend == "DeepSeekOCR" else "legacy"
        workflow = import_module(f"textimg.evaluation.{mode}")
        return getattr(workflow, f"evaluate_{args.dataset_type}")(args)
    if args.command == "llm":
        if args.llm == "qwen3-8b":
            from .providers.llm_qwen3_8b import load_qwen_model, generate_qwen_response
            args.model, args.tokenizer = load_qwen_model(args)
            args.gen_fn = generate_qwen_response
        else:
            from .providers.llm_gpt import generate_gpt_response
            from .providers.judge import required_key
            required_key("OPENAI_API_KEY")
            args.gen_fn = generate_gpt_response
        from .evaluation import llm
        return getattr(llm, f"evaluate_{args.dataset_type}")(args)
    if args.command == "judge":
        from .judging.run import run
        return run(args)
    if args.command == "analyze":
        from .analysis.run import run
        return run(args)


def main(argv=None):
    parser_obj = parser()
    args = configure(parser_obj.parse_args(argv), parser_obj)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return 0

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    from .core.manifest import start, finish

    path, record = start(args)
    try:
        summary = execute(args)
        record["judge_calls"] = getattr(args, "judge_calls", 0)
        record["dataset_fingerprints"] = getattr(args, "dataset_fingerprints", {})
        status = "completed_with_failures" if isinstance(summary, dict) and (
            summary.get("failed", 0) or summary.get("no_image", 0) or summary.get("moderation_blocked", 0)
        ) else "completed"
        finish(path, record, status=status, summary=summary)
    except BaseException as exc:
        record["judge_calls"] = getattr(args, "judge_calls", 0)
        record["dataset_fingerprints"] = getattr(args, "dataset_fingerprints", {})
        finish(path, record, status="failed", error=exc)
        raise

    if path is not None:
        print(f"Run manifest: {path}")
    return 1 if status == "completed_with_failures" else 0
