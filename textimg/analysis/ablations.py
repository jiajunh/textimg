import os
from tqdm import tqdm
import numpy as np
from textimg.core.io import load_json

def analyse_dsocr(args):
    models = ['gpt-image-1.5_low', 'gpt-image-1.5', 'flux2-pro', 'gemini-2.5-flash-image', 'qwen-image']
    for model in models:
        result_dir = os.path.join('./../results/', 'identical', 'DeepSeekOCR', f'{model}.jsonl')
        data = load_json(result_dir)
        cer = []
        wer = []
        for data_row in tqdm(data, desc=f'{model}'):
            data_id = data_row['id']
            cer.append(data_row['cer'])
            wer.append(data_row['wer'])
        print(f'{model} - cer: {np.mean(cer)}, wer: {np.mean(wer)}')

def analyse_clear_ratio(args):
    models = ['gpt-image-1.5_low', 'gpt-image-1.5', 'flux2-pro', 'gemini-2.5-flash-image', 'qwen-image']
    cats = ['identical', 'reasoning']
    for model in models:
        for cat in cats:
            result_dir = os.path.join('./../render_clear_ratio/', cat, f'{model}.jsonl')
            data = load_json(result_dir)
            clear_ratio = []
            for data_row in tqdm(data, desc=f'{model}'):
                data_id = data_row['id']
                render_quality = data_row['render_quality'].split(' ')
                clear = int(render_quality[0])
                total = int(render_quality[1])
                clear_ratio.append(1.0 * clear / total)
            print(f'{model}, {cat} - clear_ratio: {np.mean(clear_ratio)}')

def analyse_reasoning(args):
    models = ['gpt-image-1.5_low', 'gpt-image-1.5', 'flux2-pro', 'gemini-2.5-flash-image', 'qwen-image']
    all_render_quality = []
    for model in models:
        result_dir = os.path.join(args.output_dir, 'reasoning', f'{model}.jsonl')
        print(result_dir)
        data = load_json(result_dir)
        render_quality = []
        for data_row in tqdm(data, desc=f'{model}'):
            data_id = data_row['id']
            render_quality.append(int(data_row['render_quality']))
            all_render_quality.append(int(data_row['render_quality']))
        print(f'{model} - render_quality: {np.mean(render_quality)}')
    print(f'Overall - render_quality: {np.mean(all_render_quality)}')

def analyse_identical(args):
    models = ['gpt-image-1.5_low', 'gpt-image-1.5', 'flux2-pro', 'gemini-2.5-flash-image', 'qwen-image']
    all_render_quality = []
    for model in models:
        result_dir = os.path.join(args.output_dir, 'identical', f'{model}.jsonl')
        print(result_dir)
        data = load_json(result_dir)
        render_quality = []
        for data_row in tqdm(data, desc=f'{model}'):
            data_id = data_row['id']
            render_quality.append(int(data_row['render_quality']))
            all_render_quality.append(int(data_row['render_quality']))
        print(f'{model} - render_quality: {np.mean(render_quality)}')
    print(f'Overall - render_quality: {np.mean(all_render_quality)}')
