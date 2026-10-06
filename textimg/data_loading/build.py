import os
import re
import random
from collections import defaultdict
from .load import load_data

def is_long_enough(text, k, mode):
    if mode == 'sentence':
        return len(text.split()) >= k
    if mode == 'word':
        return len(text.split()) >= k
    if mode == 'character':
        return len(text) >= k
    if mode == 'token':
        return len(text.split()) >= 2 * k
    raise ValueError(mode)

def truncate_by_sentence(text, target_length):
    sentences = re.split('(?<=[.!?])\\s+', text)
    words = 0
    remaining = []
    for s in sentences:
        w = len(s.split())
        if words + w > target_length:
            break
        remaining.append(s)
        words += w
    return (' '.join(remaining), words)

def truncate_by_words(text, target_length):
    words = text.split()
    truncated = ' '.join(words[:target_length])
    return (truncated, min(len(words), target_length))

def truncate_by_character(text, target_length):
    if len(text) <= target_length:
        return (text, len(text))
    truncated = text[:target_length]
    return (truncated, target_length)

def truncate_by_tokenizer(text, target_length, tokenizer):
    encode = tokenizer(text, add_special_tokens=False, truncation=True, return_attention_mask=False, return_token_type_ids=False, max_length=512)
    input_ids = encode['input_ids']
    if len(input_ids) < target_length:
        return (text, len(input_ids))
    truncated_ids = input_ids[:target_length]
    truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
    return (truncated_text, len(truncated_ids))

def construct_identical_data(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    ds = load_data(args, 'Salesforce/wikitext', 'wikitext-103-v1', split='train')
    all_rows = []
    global_idx = 0
    tokenizer = None
    if args.length_control_type == 'token':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    for k in args.text_length:
        filtered_ds = ds.filter(lambda x: len(x['text'].split()) >= k)
        sampled_ds = filtered_ds.shuffle(seed=args.seed).select(range(args.data_size))
        for i, sample in enumerate(sampled_ds):
            if args.length_control_type == 'sentence':
                truncated_sample, length = truncate_by_sentence(sample['text'], k)
            elif args.length_control_type == 'word':
                truncated_sample, length = truncate_by_words(sample['text'], k)
            elif args.length_control_type == 'character':
                truncated_sample, length = truncate_by_character(sample['text'], k)
            elif args.length_control_type == 'token':
                truncated_sample, length = truncate_by_tokenizer(sample['text'], k, tokenizer)
            all_rows.append({'id': global_idx, 'target_length': k, 'text': truncated_sample, 'length': length})
            global_idx += 1
    output_ds = Dataset.from_list(all_rows)
    out_path = os.path.join(args.output_dir, f'wikitext_truncate_by_{args.length_control_type}.parquet')
    output_ds.to_parquet(out_path)

def construct_multilingual_data(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    tokenizer = None
    if args.length_control_type == 'token':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    for lang in args.languages:
        print(f'construct {lang} data')
        ds = load_data(args, 'wikimedia/wikipedia', f'20231101.{lang}', split='train')
        all_rows = []
        global_idx = 0
        for k in args.text_length:
            filtered_ds = ds.filter(lambda x: is_long_enough(x['text'], k, args.length_control_type))
            sampled_ds = filtered_ds.shuffle(seed=args.seed).select(range(args.data_size))
            avg_length = 0.0
            for i, sample in enumerate(sampled_ds):
                if args.length_control_type == 'sentence':
                    truncated_sample, length = truncate_by_sentence(sample['text'], k)
                elif args.length_control_type == 'word':
                    truncated_sample, length = truncate_by_words(sample['text'], k)
                elif args.length_control_type == 'character':
                    truncated_sample, length = truncate_by_character(sample['text'], k)
                elif args.length_control_type == 'token':
                    truncated_sample, length = truncate_by_tokenizer(sample['text'], k, tokenizer)
                all_rows.append({'id': global_idx, 'target_length': k, 'text': truncated_sample, 'length': length})
                global_idx += 1
                avg_length += length
            print(f'lang: {lang}, target_length: {k}, avg_length: {avg_length / args.data_size}')
        output_ds = Dataset.from_list(all_rows)
        out_path = os.path.join(args.output_dir, lang, f'wikipedia_truncate_by_{args.length_control_type}.parquet')
        output_ds.to_parquet(out_path)

def construct_reasoning_data(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    ds = load_data(args, 'DigitalLearningGmbH/MATH-lighteval', 'default')['train']
    global_idx = 0
    all_rows = []
    for level in args.levels:
        ds_lvl = ds.filter(lambda x: x['level'] == f'Level {level}')
        sampled_ds = ds_lvl.shuffle(seed=args.seed).select(range(args.data_size))
        for row in sampled_ds:
            all_rows.append({'id': global_idx, 'problem': row['problem'], 'solution': row['solution'], 'level': row['level'], 'type': row['type']})
            global_idx += 1
    output_ds = Dataset.from_list(all_rows)
    out_path = os.path.join(args.output_dir, f'math_sample.parquet')
    output_ds.to_parquet(out_path)

def construct_context_reasoning_data(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    ds = load_data(args, 'ucinlp/drop')['train']
    global_idx = 0
    all_rows = []
    rng = random.Random(args.seed)
    groups = defaultdict(list)
    for idx, row in enumerate(ds):
        groups[row['section_id']].append(idx)
    unique_passages = list(groups.keys())
    sample_passages = rng.sample(unique_passages, args.data_size)
    sampled_indices = [rng.choice(groups[c]) for c in sample_passages]
    sampled_ds = ds.select(sampled_indices)
    for row in sampled_ds:
        all_rows.append({'id': global_idx, 'section_id': row['section_id'], 'query_id': row['query_id'], 'passage': row['passage'], 'question': row['question'], 'answers_spans': row['answers_spans']})
        global_idx += 1
    output_ds = Dataset.from_list(all_rows)
    out_path = os.path.join(args.output_dir, f'context_reasoning_sample.parquet')
    output_ds.to_parquet(out_path)

def construct_multiple_choice_data(args):
    from datasets import Dataset
    from transformers import AutoTokenizer
    subsets = ['ARC-Challenge', 'ARC-Easy']
    for subset in subsets:
        ds = load_data(args, 'allenai/ai2_arc', subset)['train']
        sampled_ds = ds.shuffle(seed=args.seed).select(range(args.data_size))
        global_idx = 0
        all_rows = []
        for row in sampled_ds:
            all_rows.append({'id': global_idx, 'question': row['question'], 'choices': row['choices'], 'answerKey': row['answerKey']})
            global_idx += 1
        output_ds = Dataset.from_list(all_rows)
        out_path = os.path.join(args.output_dir, f"multiple_choice_{subset.split('-')[-1].lower()}.parquet")
        output_ds.to_parquet(out_path)
