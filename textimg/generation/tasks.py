"""Dataset grouping and input formatting, shared by image workflows."""
from pathlib import Path
from textimg.datasets.load import load_data
from textimg.core.io import limit_rows
from . import prompts


def groups(args):
    ds = load_data(args, args.dataset, args.dataset_type)
    task = args.dataset_type
    if task == "identical":
        for length in args.text_length:
            yield (str(length), ds["train"].filter(lambda x: x["target_length"] == length),
                   prompts.IDENTICAL_PROMPT)
    elif task == "multilingual":
        for lang in args.languages:
            for length in args.text_length:
                yield (f"{lang}/{length}", ds[lang].filter(lambda x: x["target_length"] == length),
                       prompts.MULTILINGUAL_PROMPT[lang])
    elif task == "reasoning":
        for level in args.levels:
            yield ("", ds["train"].filter(lambda x: x["level"] == f"Level {level}"),
                   prompts.REASONING_PROMPT)
    elif task == "context_reasoning":
        yield "", ds["train"], prompts.CONTEXT_REASONING_PROMPT
    else:
        for split in args.mc_difficulty:
            yield split, ds[split], prompts.MULTIPLE_CHOICE_PROMPT


def input_text(row, task):
    if task in {"identical", "multilingual"}:
        return row["text"]
    if task == "reasoning":
        return row["problem"]
    if task == "context_reasoning":
        return f"Passage:\n{row['passage']}\n\nQuestion:\n{row['question']}"
    choices = "".join(f"{label}: {text}\n" for label, text in
                      zip(row["choices"]["label"], row["choices"]["text"]))
    return f"Question:\n{row['question']}\n\nChoices:\n{choices}\n"


def generation_jobs(args):
    base = Path(args.output_dir) / (args.run_label or args.model)
    for group, data, prompt in groups(args):
        for row in limit_rows(data, args):
            yield row["id"], prompt, input_text(row, args.dataset_type), base / group / f"{row['id']}.png"
