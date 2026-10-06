import os
import json
from tqdm import tqdm
import numpy as np
from textimg.core.io import load_json
from textimg.judging.reasoning import math_answer_score, qa_answer_score

def analyse_context_reasoning_preprocessing(args):
    result_dir = os.path.join(args.output_dir, 'context_reasoning', args.judge_model, f'{args.model}.jsonl')
    print(result_dir)
    data = load_json(result_dir)
    if args.limit is not None:
        data = data[:args.limit]
    print(data[0])
    output_path = os.path.join(args.output_dir, 'context_reasoning', args.judge_model, f'{args.model}_preprocess.jsonl')
    with open(output_path, 'w', encoding='utf-8') as f:
        for data_row in tqdm(data, desc=f'{args.model}'):
            data_id = data_row['id']
            passage = data_row['passage']
            question = data_row['question']
            gt_answer = data_row['gt_answer']
            process_scrores = data_row['reasoning_scores']
            vlm_answer = data_row['vlm_solution']
            vlm_answer_split = vlm_answer.split('Answer:')
            vlm_answer = vlm_answer_split[-1] if len(vlm_answer_split) <= 2 else vlm_answer_split[1]
            ans_score = qa_answer_score(args, gt_answer, vlm_answer, passage, question)
            result = {'id': data_id, 'ans_score': ans_score, 'gt_answer': gt_answer, 'vlm_answer': vlm_answer, 'process_scrores': process_scrores}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def analyse_context_reasoning(args):
    result_dir = os.path.join(args.output_dir, 'context_reasoning', args.judge_model, f'{args.model}_preprocess.jsonl')
    print(result_dir)
    data = load_json(result_dir)
    ans_scores = []
    process_scores = []
    for data_row in tqdm(data, desc=f'{args.model}'):
        ans_scores.append(data_row['ans_score'])
        process_scores.append(np.mean(data_row['process_scrores']))
    print(f'Overall - ans_scores: {np.mean(ans_scores)}, process_scores: {np.mean(process_scores)}')
