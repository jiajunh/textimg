import re
import base64
from textimg.providers.judge import judge_client
from textimg.core.text import deduplicate_steps

def norm_header(s_list):
    matching = set()
    for s in s_list:
        s = s.strip()
        s = re.sub('\\s+', ' ', s)
        matching.add(s)
    return list(matching)

def find_fuzzy_spans(text, keyword, threshold=80):
    from rapidfuzz import fuzz
    k = len(keyword)
    spans = []
    for i in range(len(text) - k + 1):
        window = text[i:i + k]
        if fuzz.ratio(window, keyword) >= threshold:
            spans.append(text[i:i + k])
    return spans

def canonicalize_reasoning_headers(text, keyword, threshold=80):
    headers = find_fuzzy_spans(text, keyword, threshold)
    headers = norm_header(headers)
    for header in headers:
        text = re.sub(f'\\b{re.escape(header)}\\b:?', keyword, text, flags=re.IGNORECASE)
    return text

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_extract_text(args, image_path):
    client = judge_client(args)
    prompt = (
        'Extract all visible text from the image.\n'
        'Do not correct spelling, punctuation, or spacing.\n'
        '\n'
        'For mathematical expressions only:\n'
        '- Convert them to LaTeX math notation.\n'
        '- Use inline math ($...$).\n'
        '\n'
        'For non-mathematical text:\n'
        '- Output it exactly as plain text (not LaTeX-formatted).\n'
        '\n'
        'Preserve the original reading order and line breaks.\n'
        '    '
    )
    base64_image = encode_image(image_path)
    response = client.responses.create(model=args.judge_model, input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': f'data:image/png;base64,{base64_image}'}]}])
    return response.output_text

def math_process_score(args, question, solution):
    client = judge_client(args)
    prompt = (
        'You are judging the correctness of one reasoning step, given the problem and all previous steps. '
        'Determine whether the current step is correct.\n'
        '\n'
        'Return only a single digit:\n'
        '  - 1 if the current step is correct\n'
        '  - 0 if the current step is incorrect\n'
        '    '
    )
    m = re.search('(Step.*)(?=Answer:)', solution, flags=re.DOTALL)
    steps = m.group(1).split('Step') if m else None
    if steps:
        steps = [f'Step{s.strip()}' for s in steps if s.strip()]
    previous_steps = ''
    current_step = ''
    if getattr(args, 'deduplicate_steps', False) and steps:
        steps = deduplicate_steps(steps)
    process_scores = []
    if steps is None:
        process_scores.append(0)
        return process_scores
    for step in steps:
        current_step = step
        input_text = f'{prompt} \n Question:\n{question}\nPrevious steps:\n{previous_steps}\nCurrent step:\n{current_step}\n'
        response = client.responses.create(model=args.judge_model, reasoning={'effort': 'low'}, input=input_text)
        if response.output_text.strip() not in ['0', '1']:
            process_scores.append(0)
        else:
            process_scores.append(int(response.output_text.strip()))
        previous_steps += f'\n\n{current_step}'
    return process_scores

def context_reasoning_score(args, passage, question, solution):
    client = judge_client(args)
    prompt = (
        'You are judging the correctness of one reasoning step, given the passage and the reasoning step. '
        'Determine whether the current step is correct given the passage.\n'
        '\n'
        'Return only a single digit:\n'
        '  - 1 if the current step is correct\n'
        '  - 0 if the current step is incorrect\n'
        '    '
    )
    text_with_variant_keyword = canonicalize_reasoning_headers(solution, 'Reasoning:', threshold=80)
    m = re.search('(Reasoning:.*)(?=Answer:)', text_with_variant_keyword, flags=re.DOTALL)
    steps = m.group(1).split('Reasoning:') if m else None
    if steps:
        steps = [{s.strip()} for s in steps if s.strip()]
    if getattr(args, 'deduplicate_steps', False) and steps:
        steps = deduplicate_steps(steps)
    process_scores = []
    if steps is None:
        process_scores.append(0)
        return process_scores
    for step in steps:
        input_text = f'{prompt}\nPassage:\n{passage}\nQuestion:\n{question}\n\nReasoning step:\n{step}'
        response = client.responses.create(model=args.judge_model, reasoning={'effort': 'low'}, input=input_text)
        if response.output_text.strip() not in ['0', '1']:
            process_scores.append(0)
        else:
            process_scores.append(int(response.output_text.strip()))
    return process_scores

def multiple_choice_reasoning_score(args, question, solution):
    client = judge_client(args)
    prompt = (
        'You are judging the correctness of one reasoning step, given the question and the reasoning step. '
        'Determine whether the current step is correct.\n'
        '\n'
        'Return only a single digit:\n'
        '  - 1 if the current step is correct\n'
        '  - 0 if the current step is incorrect\n'
        '    '
    )
    text_with_variant_keyword = canonicalize_reasoning_headers(solution, 'Reasoning:', threshold=80)
    m = re.search('(Reasoning:.*)(?=Answer:)', text_with_variant_keyword, flags=re.DOTALL)
    reasoning_count = 0
    steps = m.group(1).split('Reasoning:') if m else None
    if steps:
        steps = [{s.strip()} for s in steps if s.strip()]
        reasoning_count = len(steps)
    if getattr(args, 'deduplicate_steps', False) and steps:
        steps = deduplicate_steps(steps)
    process_scores = []
    if steps is None:
        process_scores.append(0)
        return (process_scores, reasoning_count)
    for step in steps:
        input_text = f'{prompt}\nQuestion:\n{question}\n\nReasoning step:\n{step}'
        response = client.responses.create(model=args.judge_model, reasoning={'effort': 'low'}, input=input_text)
        if response.output_text.strip() not in ['0', '1']:
            process_scores.append(0)
        else:
            process_scores.append(int(response.output_text.strip()))
    return (process_scores, reasoning_count)

def math_answer_score(args, gt_answer, vlm_answer):
    client = judge_client(args)
    prompt = (
        'You are judging whether the candidate answer is equivalent to the ground truth answer.\n'
        'Consider only the final answer content. Ignore formatting, wording, and explanation.\n'
        '\n'
        'Return only a single digit and do not output anything else:\n'
        '- 1 if the given answer is equivalent to the ground truth\n'
        '- 0 if the given answer is not equivalent\n'
    )
    input_text = f'{prompt}\nGround truth answer:\n{gt_answer}\n\nCandidate answer:\n{vlm_answer}'
    ans_score = 0
    response = client.responses.create(model=args.judge_model, reasoning={'effort': 'low'}, input=input_text)
    if response.output_text.strip() not in ['0', '1']:
        ans_score = 0
    else:
        ans_score = int(response.output_text.strip())
    return ans_score

def qa_answer_score(args, gt_answer, vlm_answer, passage=None, question=None):
    client = judge_client(args)
    prompt = (
        'You are judging whether the candidate answer has the same meaning to the ground truth answer given '
        'the passage and the question.\n'
        '\n'
        'Return only a single digit and do not output anything else:\n'
        '- 1 if the given answer is equivalent to the ground truth\n'
        '- 0 if the given answer is not equivalent\n'
    )
    input_text = f'{prompt}\nPassage:\n{passage}\n\nQuestion:\n{question}\n\nGround truth answer:\n{gt_answer}\n\nCandidate answer:\n{vlm_answer}'
    ans_score = 0
    response = client.responses.create(model=args.judge_model, reasoning={'effort': 'low'}, input=input_text)
    if response.output_text.strip() not in ['0', '1']:
        ans_score = 0
    else:
        ans_score = int(response.output_text.strip())
    return ans_score
