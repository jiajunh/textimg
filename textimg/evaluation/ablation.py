import os
import json
from tqdm import tqdm
from textimg.data_loading.load import load_data
from textimg.core.io import image_files, completed_ids, open_results
from textimg.core.text import *
from textimg.judging.reasoning import gpt_extract_text, math_process_score, context_reasoning_score, multiple_choice_reasoning_score
from .deepseek_text import extract_deepseek_ocr_result

def evaluate_identical(args):
    ds = load_data(args, args.dataset, 'identical')['train']
    output_path = os.path.join(args.output_dir, f'{args.model}.jsonl')
    print(output_path)
    id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
    done = completed_ids(output_path, args)
    with open_results(output_path, args) as f:
        for k in args.text_length:
            img_dir = os.path.join(args.img_dir, str(k))
            img_files = image_files(img_dir, args)
            for img_file in tqdm(img_files, desc=f'text_length={k}'):
                img_id = int(img_file.split('.')[0])
                if img_id in done:
                    continue
                if img_id not in id_to_idx:
                    continue
                data_row = ds[id_to_idx[img_id]]
                original_text = data_row['text']
                cur_img_path = os.path.join(img_dir, img_file)
                ocr_texts = ''
                ocr_avg_confidence = 0.0
                ocr_result = args.backend.infer_one(cur_img_path)
                if args.ocr_backend == 'PaddleOCR':
                    ocr_texts, ocr_avg_confidence = extract_paddle_ocr_result(ocr_result)
                elif args.ocr_backend == 'DeepSeekOCR':
                    ocr_texts = extract_deepseek_ocr_result(ocr_result)
                    ocr_avg_confidence = None
                metrics = compute_metric(original_text, ocr_texts)
                metrics['ocr_avg_confidence'] = ocr_avg_confidence
                result = {'id': data_row['id'], **metrics, 'original_text': original_text, 'ocr_text': ocr_texts}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
