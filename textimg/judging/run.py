"""A single rendering workflow for the three original rendering metrics."""
import json
from pathlib import Path
from textimg.core.io import image_files
from .rendering import gpt_judge_image, gpt_judge_clear_chars, gpt_judge_clear_areas


def run(args):
    from tqdm import tqdm
    fn = {"quality": gpt_judge_image, "clear_ratio": gpt_judge_clear_chars,
          "clear_area": gpt_judge_clear_areas}[args.render_metric]
    task = args.dataset_type
    base = Path(args.img_dir)
    if task == "multilingual":
        outputs = [(f"{args.model}_{lang}.jsonl", [base / lang / str(k) for k in args.text_length])
                   for lang in args.languages]
    elif task == "identical":
        outputs = [(f"{args.model}.jsonl", [base / str(k) for k in args.text_length])]
    elif task == "multiple_choice":
        outputs = [(f"{args.model}_{split}.jsonl", [base / split]) for split in args.mc_difficulty]
    else:
        outputs = [(f"{args.model}.jsonl", [base])]
    for filename, directories in outputs:
        with open(Path(args.output_dir) / filename, "w", encoding="utf-8") as f:
            for directory in directories:
                for name in tqdm(image_files(directory, args), desc=str(directory)):
                    value = fn(args, str(directory / name))
                    # Raw judge text stays in the historical result schema.
                    f.write(json.dumps({"id": int(Path(name).stem), "render_quality": value}) + "\n")
