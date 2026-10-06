import os
import json
from tqdm import tqdm
import numpy as np
from textimg.core.io import load_json
from textimg.judging.reasoning import math_answer_score, qa_answer_score

def analyse_multiple_choice(args):
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
            vlm_answer = data_row['vlm_solution']
            vlm_answer_split = vlm_answer.split('swer:')[-1]
            if gt_answer == '1':
                gt_answer = 'A'
            elif gt_answer == '2':
                gt_answer = 'B'
            elif gt_answer == '3':
                gt_answer = 'C'
            elif gt_answer == '4':
                gt_answer = 'D'
            if vlm_answer_split and gt_answer in vlm_answer_split:
                all_ans_scores.append(1)
                ans_scores[split].append(1)
            else:
                all_ans_scores.append(0)
                ans_scores[split].append(0)
            process_scores[split].append(np.mean(process_scrores))
            all_process_scores.append(np.mean(process_scrores))
    for split in args.mc_difficulty:
        print(f'Split: {split}, ans_scores: {np.mean(ans_scores[split])}, process_scores: {np.mean(process_scores[split])}')
    print(f'Overall - ans_scores: {np.mean(all_ans_scores)}, process_scores: {np.mean(all_process_scores)}')
