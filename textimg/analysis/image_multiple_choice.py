import os
import json
from tqdm import tqdm
import numpy as np
from collections import Counter
from textimg.core.answers import multiple_choice_answer_score
from textimg.core.io import load_json
from textimg.judging.reasoning import math_answer_score, qa_answer_score

def analyse_multiple_choice(args):
    answer_statuses = Counter()
    all_ans_scores = []
    all_process_scores = []
    ans_scores = {'easy': [], 'challenge': []}
    process_scores = {'easy': [], 'challenge': []}
    for split in args.mc_difficulty:
        result_dir = os.path.join(args.output_dir, 'multiple_choice', args.judge_model, f'{args.model}_{split}.jsonl')
        print(result_dir)
        data = load_json(result_dir)
        for data_row in tqdm(data, desc=f'{args.model}'):
            data_id = data_row['id']
            gt_answer = data_row['gt_answer']
            process_scrores = data_row['reasoning_scores']
            score, status = multiple_choice_answer_score(data_row)
            answer_statuses[status] += 1
            all_ans_scores.append(score)
            ans_scores[split].append(score)
            process_scores[split].append(np.mean(process_scrores))
            all_process_scores.append(np.mean(process_scrores))
    print('Answer validation:', dict(answer_statuses))
    for split in args.mc_difficulty:
        print(f'Split: {split}, ans_scores: {np.mean(ans_scores[split])}, process_scores: {np.mean(process_scores[split])}')
    print(f'Overall - ans_scores: {np.mean(all_ans_scores)}, process_scores: {np.mean(all_process_scores)}')
