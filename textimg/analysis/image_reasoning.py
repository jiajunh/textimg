import os
import json
from tqdm import tqdm
import numpy as np
from collections import Counter
from textimg.core.answers import answer_from_row, ANSWER_PARSER_VERSION, check_preprocess_resume
from textimg.core.io import load_json, completed_ids, open_results
from textimg.judging.reasoning import math_answer_score, qa_answer_score

def pre_analyse_reasoning(args):
    result_dir = os.path.join(args.output_dir, 'reasoning', args.judge_model, f'{args.model}.jsonl')
    print(result_dir)
    data = load_json(result_dir)
    if args.limit is not None:
        data = data[:args.limit]
    output_path = os.path.join(args.output_dir, 'reasoning', args.judge_model, f'{args.model}_preprocess.jsonl')
    check_preprocess_resume(output_path, args)
    done = completed_ids(output_path, args)
    with open_results(output_path, args) as f:
        for data_row in tqdm(data, desc=f'{args.model}'):
            data_id = data_row['id']
            if data_id in done:
                continue
            level = 5
            if data_id < 800:
                level = 4
            if data_id < 600:
                level = 3
            if data_id < 400:
                level = 2
            if data_id < 200:
                level = 1
            gt_answer = data_row['gt_answer']
            process_scrores = data_row['process_scrores']
            vlm_answer, answer_status = answer_from_row(data_row)
            ans_score = math_answer_score(args, gt_answer, vlm_answer) if answer_status == 'valid' else 0
            result = {'id': data_id, 'level': level, 'ans_score': ans_score, 'answer_status': answer_status, 'answer_parser_version': ANSWER_PARSER_VERSION, 'gt_answer': gt_answer, 'vlm_answer': vlm_answer, 'process_scrores': process_scrores}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def analyse_reasoning(args):
    result_dir = os.path.join(args.output_dir, 'reasoning', args.judge_model, f'{args.model}_preprocess.jsonl')
    print(result_dir)
    data = load_json(result_dir)
    print('Answer validation:', dict(Counter(row.get('answer_status', 'legacy_unvalidated') for row in data)))
    ans_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
    process_scores = {1: [], 2: [], 3: [], 4: [], 5: []}
    all_ans_scores = []
    all_process_scores = []
    for data_row in tqdm(data, desc=f'{args.model}'):
        all_ans_scores.append(data_row['ans_score'])
        all_process_scores.append(np.mean(data_row['process_scrores']))
        ans_scores[data_row['level']].append(data_row['ans_score'])
        process_scores[data_row['level']].append(np.mean(data_row['process_scrores']))
    for lv in range(1, 6):
        print(f'level: {lv}, ans_scores: {np.mean(ans_scores[lv])}, process_scores: {np.mean(process_scores[lv])}')
    print(f'Overall - ans_scores: {np.mean(all_ans_scores)}, process_scores: {np.mean(all_process_scores)}')
