import os
import json
from tqdm import tqdm
from textimg.data_loading.load import load_data
from textimg.core.text import *
from textimg.judging.reasoning import math_process_score, context_reasoning_score, multiple_choice_reasoning_score

def evaluate_identical(args):
    prompt = 'Generate exactly the same input texts without any modification.\n\nInput text:'
    ds = load_data(args, args.dataset, 'identical')['train']
    output_path = os.path.join(args.output_dir, f'{args.llm}.jsonl')
    print(output_path)
    id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
    with open(output_path, 'w', encoding='utf-8') as f:
        for data_row in tqdm(ds, desc=f'identical'):
            data_id = data_row['id']
            original_text = data_row['text']
            llm_text = args.gen_fn(args, prompt, original_text)
            metrics = compute_metric(original_text, llm_text)
            result = {'id': data_row['id'], **metrics, 'original_text': original_text, 'llm_text': llm_text}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_reasoning(args):
    prompt = (
        'Solve the problem and present a clear, human-readable solution. Include intermediate steps and '
        'justifications.\n'
        'Do not mention hidden chain-of-thought; just show a worked solution.\n'
        '\n'
        '- Each reasoning paragraph starts with "Step:".\n'
        '- The final paragraph is the answer with only one line exactly: "Answer: <final answer>".\n'
        '- <final answer> contains ONLY the final result itself, with no other variables, no equations, no '
        'units, no description, and no explanatory text.\n'
        '\n'
        'Question:'
    )
    ds = load_data(args, args.dataset, 'reasoning')['train']
    id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
    output_path = os.path.join(args.output_dir, f'{args.llm}.jsonl')
    print(output_path)
    with open(output_path, 'w', encoding='utf-8') as f:
        for data_row in tqdm(ds, desc=f'reasoning'):
            data_id = data_row['id']
            problem = data_row['problem']
            solution_text = data_row['solution']
            original_text = problem
            gt_answer = extract_math_solution(solution_text)
            llm_text = args.gen_fn(args, prompt, original_text)
            process_scores = math_process_score(args, problem, llm_text)
            result = {'id': data_row['id'], 'problem': problem, 'vlm_solution': llm_text, 'gt_answer': gt_answer, 'process_scrores': process_scores}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_context_reasoning(args):
    prompt = (
        'You are answering a reading comprehension question using the given passage. \n'
        'Answer the question and present clear intermediate steps and justifications.\n'
        'Do not mention hidden chain-of-thought; just show how to get the answer.\n'
        '\n'
        'Requirements:\n'
        'Provide the reasoning first and then the final answer.\n'
        '- Each reasoning step must be one paragraph starting with "Reasoning:".\n'
        '- The final paragraph is one line, must start with exactly: "Answer: <final_answer>". Output only '
        'the answer text.\n'
        '    '
    )
    ds = load_data(args, args.dataset, 'context_reasoning')['train']
    id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
    output_path = os.path.join(args.output_dir, f'{args.llm}.jsonl')
    print(output_path)
    idx = 0
    with open(output_path, 'w', encoding='utf-8') as f:
        for data_row in tqdm(ds, desc=f'context_reasoning'):
            passage = data_row['passage']
            question = data_row['question']
            answers_spans = data_row['answers_spans']
            gt_answer = answers_spans['spans']
            text = f'Passage:\n{passage}\n\nQuestion:\n{question}'
            print('-' * 60)
            print(text)
            llm_text = args.gen_fn(args, prompt, text)
            print('#' * 60)
            print(llm_text)
            reasoning_scores = context_reasoning_score(args, passage, question, llm_text)
            result = {'id': data_row['id'], 'passage': passage, 'question': question, 'vlm_solution': llm_text, 'gt_answer': gt_answer, 'reasoning_scores': reasoning_scores}
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

def evaluate_multiple_choice(args):
    prompt = (
        'You are answering a multiple choice question. \n'
        'Answer the question and present a clear, human-readable solution for each choice. Include '
        'intermediate steps and justifications.\n'
        'Do not mention hidden chain-of-thought; just show a worked solution.\n'
        '\n'
        'Solution requirements:\n'
        'Output only the reasoning steps and the final answer.\n'
        '- First output the reasoning and then the answer.\n'
        '- Provide exactly one reasoning paragraph for each choice.\n'
        '- Each reasoning step must be one paragraph starting with "Reasoning:".\n'
        '- The final paragraph is one line, must start with exactly: "Answer: <choice>".\n'
        '- The answer must be exactly only one capital letter: A, B, C, or D.\n'
        '- Do NOT include any other text after the answer line.\n'
        '    '
    )
    for split in args.mc_difficulty:
        ds = load_data(args, args.dataset, 'multiple_choice')[split]
        id_to_idx = {id_: i for i, id_ in enumerate(ds['id'])}
        output_path = os.path.join(args.output_dir, f'{args.llm}_{split}.jsonl')
        print(output_path)
        with open(output_path, 'w', encoding='utf-8') as f:
            for data_row in tqdm(ds, desc=f'context_reasoning'):
                choices = data_row['choices']
                question = data_row['question']
                gt_answer = data_row['answerKey']
                choice_labels = choices['label']
                choice_texts = choices['text']
                c_texts = ''
                for choice_label, choice_text in zip(choice_labels, choice_texts):
                    c_texts = f'{c_texts}{choice_label}: {choice_text}\n'
                text = f'Question:\n{question}\n\nChoices:\n{c_texts}'
                llm_text = args.gen_fn(args, prompt, text)
                reasoning_scores, reasoning_count = multiple_choice_reasoning_score(args, text, llm_text)
                result = {'id': data_row['id'], 'question': question, 'vlm_solution': llm_text, 'gt_answer': gt_answer, 'reasoning_scores': reasoning_scores, 'reasoning_count': reasoning_count}
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
