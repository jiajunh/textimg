"""Dispatch the original, deliberately tolerant analysis algorithms."""
from importlib import import_module


def run(args):
    task_module = {"reasoning": "reasoning", "context_reasoning": "context",
                   "multiple_choice": "multiple_choice", "identical": "identical"}
    for task in args.tasks:
        mod = import_module(f"textimg.analysis.{args.analysis_source}_{task_module[task]}")
        if args.preprocess:
            if task == "reasoning":
                mod.pre_analyse_reasoning(args)
            elif task == "context_reasoning":
                mod.analyse_context_reasoning_preprocessing(args)
        getattr(mod, f"analyse_{task}")(args)
