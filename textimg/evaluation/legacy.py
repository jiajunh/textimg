import os
import json
from tqdm import tqdm
from textimg.data_loading.load import load_data
from textimg.core.io import image_files, completed_ids, open_results
from textimg.core.text import *
from textimg.judging.reasoning import gpt_extract_text, math_process_score, context_reasoning_score, multiple_choice_reasoning_score

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
                data_row = ds[id_to_idx[img_id]]
                original_text = data_row['text']
                cur_img_path = os.path.join(img_dir, img_file)
                ocr_texts = ''
                ocr_avg_confidence = 0.0
                ocr_result = args.backend.infer_one(cur_img_path)
                if args.ocr_backend == 'PaddleOCR':
                    ocr_texts, ocr_avg_confidence = extract_paddle_ocr_result(ocr_result)
                metrics = compute_metric(original_text, ocr_texts)
                metrics['ocr_avg_confidence'] = ocr_avg_confidence
                result = {'id': data_row['id'], **metrics, 'original_text': original_text, 'ocr_text': ocr_texts}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_multilingual(args):
    ds = load_data(args, args.dataset, 'multilingual')
    for lang in args.languages:
        data = ds[lang]
        id_to_idx = {id_: i for i, id_ in enumerate(data['id'])}
        output_path = os.path.join(args.output_dir, f'{args.model}_{lang}.jsonl')
        print(output_path)
        done = completed_ids(output_path, args)
        with open_results(output_path, args) as f:
            for k in args.text_length:
                img_dir = os.path.join(args.img_dir, lang, str(k))
                img_files = image_files(img_dir, args)
                for img_file in tqdm(img_files, desc=f'language={lang}, text_length={k}'):
                    img_id = int(img_file.split('.')[0])
                    if img_id in done:
                        continue
                    data_row = data[id_to_idx[img_id]]
                    original_text = data_row['text']
                    cur_img_path = os.path.join(img_dir, img_file)
                    ocr_texts = ''
                    ocr_avg_confidence = 0.0
                    ocr_result = args.backend.infer_one(cur_img_path)
                    if args.ocr_backend == 'PaddleOCR':
                        ocr_texts, ocr_avg_confidence = extract_paddle_ocr_result(ocr_result)
                    metrics = compute_metric(original_text, ocr_texts)
                    metrics['ocr_avg_confidence'] = ocr_avg_confidence
                    result = {'id': data_row['id'], **metrics, 'original_text': original_text, 'ocr_text': ocr_texts}
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_reasoning(args):
    ds = load_data(args, args.dataset, 'reasoning')['train']
    id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
    output_path = os.path.join(args.output_dir, f'{args.model}.jsonl')
    print(output_path)
    done = completed_ids(output_path, args)
    with open_results(output_path, args) as f:
        img_dir = args.img_dir
        img_files = image_files(img_dir, args)
        for img_file in tqdm(img_files, desc=f'reasoning'):
            img_id = int(img_file.split('.')[0])
            if img_id in done:
                continue
            data_row = ds[id_to_idx[img_id]]
            problem = data_row['problem']
            solution_text = data_row['solution']
            gt_answer = extract_math_solution(solution_text)
            cur_img_path = os.path.join(img_dir, img_file)
            vlm_text = gpt_extract_text(args, cur_img_path)
            vlm_text = normalize_vlm_text(vlm_text)
            process_scores = math_process_score(args, problem, vlm_text)
            result = {'id': data_row['id'], 'problem': problem, 'vlm_solution': vlm_text, 'gt_answer': gt_answer, 'process_scrores': process_scores}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_context_reasoning(args):
    ds = load_data(args, args.dataset, 'context_reasoning')['train']
    id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
    output_path = os.path.join(args.output_dir, f'{args.model}.jsonl')
    print(output_path)
    done = completed_ids(output_path, args)
    with open_results(output_path, args) as f:
        img_dir = args.img_dir
        img_files = image_files(img_dir, args)
        for img_file in tqdm(img_files, desc=f'context_reasoning'):
            img_id = int(img_file.split('.')[0])
            if img_id in done:
                continue
            data_row = ds[id_to_idx[img_id]]
            passage = data_row['passage']
            question = data_row['question']
            answers_spans = data_row['answers_spans']
            gt_answer = answers_spans['spans']
            cur_img_path = os.path.join(img_dir, img_file)
            vlm_text = gpt_extract_text(args, cur_img_path)
            vlm_text = normalize_vlm_text(vlm_text)
            reasoning_scores = context_reasoning_score(args, passage, question, vlm_text)
            result = {'id': data_row['id'], 'passage': passage, 'question': question, 'vlm_solution': vlm_text, 'gt_answer': gt_answer, 'reasoning_scores': reasoning_scores}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_multiple_choice(args):
    for split in args.mc_difficulty:
        ds = load_data(args, args.dataset, 'multiple_choice')[split]
        id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
        output_path = os.path.join(args.output_dir, f'{args.model}_{split}.jsonl')
        print(output_path)
        done = completed_ids(output_path, args)
        with open_results(output_path, args) as f:
            img_dir = os.path.join(args.img_dir, split)
            img_files = image_files(img_dir, args)
            for img_file in tqdm(img_files, desc=f'multiple_choice'):
                img_id = int(img_file.split('.')[0])
                if img_id in done:
                    continue
                data_row = ds[id_to_idx[img_id]]
                choices = data_row['choices']
                question = data_row['question']
                gt_answer = data_row['answerKey']
                choice_labels = choices['label']
                choice_texts = choices['text']
                c_texts = ''
                for choice_label, choice_text in zip(choice_labels, choice_texts):
                    c_texts = f'{c_texts}{choice_label}: {choice_text}\n'
                text = f'{question}\n\nChoices:\n{c_texts}'
                cur_img_path = os.path.join(img_dir, img_file)
                vlm_text = gpt_extract_text(args, cur_img_path)
                vlm_text = normalize_vlm_text(vlm_text)
                reasoning_scores, reasoning_count = multiple_choice_reasoning_score(args, text, vlm_text)
                result = {'id': data_row['id'], 'question': question, 'vlm_solution': vlm_text, 'gt_answer': gt_answer, 'reasoning_scores': reasoning_scores, 'reasoning_count': reasoning_count}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
