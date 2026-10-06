import os
import json
from tqdm import tqdm
import numpy as np
from textimg.core.io import load_json
from textimg.judging.reasoning import math_answer_score, qa_answer_score

def analyse_identical(args):
    result_dir = os.path.join(args.output_dir, 'identical', args.ocr_backend, f'{args.model}.jsonl')
    print(result_dir)
    data = load_json(result_dir)
    print(data[0])
    all_cer_scores = []
    all_wer_scores = []
    all_conf_scores = []
    cer_scores = {64: [], 128: [], 256: [], 512: []}
    wer_scores = {64: [], 128: [], 256: [], 512: []}
    conf_scores = {64: [], 128: [], 256: [], 512: []}
    for data_row in tqdm(data, desc=f'{args.model}'):
        data_id = data_row['id']
        length = 512
        if data_id < 900:
            length = 256
        if data_id < 600:
            length = 128
        if data_id < 300:
            length = 64
        cer = data_row['cer']
        wer = data_row['wer']
        conf = data_row['ocr_avg_confidence']
        cer_scores[length].append(cer)
        wer_scores[length].append(wer)
        conf_scores[length].append(conf)
        all_cer_scores.append(cer)
        all_wer_scores.append(wer)
        all_conf_scores.append(conf)
    for l in [64, 128, 256, 512]:
        print(f'Length: {l} - CER: {np.mean(cer_scores[l])}, WER: {np.mean(wer_scores[l])}, Conf: {np.mean(conf_scores[l])}')
    print(f'Overall - CER: {np.mean(all_cer_scores)}, WER: {np.mean(all_wer_scores)}, Conf: {np.mean(all_conf_scores)}')
