"""Legacy normalization, answer extraction and CER/WER, preserved verbatim."""

def deduplicate_steps(steps):
    """Optional exact deduplication, preserving order and the legacy step type.

    No fuzzy merging: similar-looking steps may make different claims.
    """
    seen = set()
    unique = []
    for step in steps:
        key = tuple(sorted(step)) if isinstance(step, set) else step
        if key not in seen:
            seen.add(key)
            unique.append(step)
    return unique

def remove_boxed(s):
    if '\\boxed ' in s:
        left = '\\boxed '
        assert s[:len(left)] == left
        return s[len(left):]
    left = '\\boxed{'
    assert s[:len(left)] == left
    assert s[-1] == '}'
    return s[len(left):-1]

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if '\\boxed ' in string:
        return '\\boxed ' + string.split('\\boxed ')[-1].split('$')[0]
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    retval = None if right_brace_idx is None else string[idx:right_brace_idx + 1]
    return retval

def extract_math_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))

def normalize_vlm_text(text):
    PLACEHOLDER = '__DOUBLE_NEWLINE__'
    text = text.replace('\n\n', PLACEHOLDER)
    text = text.replace('\n', ' ')
    text = text.replace(PLACEHOLDER, ' \n')
    return text

def extract_paddle_ocr_result(result):
    import numpy as np
    texts = []
    confidence_scores = []
    for res in result:
        for rec_text in res['rec_texts']:
            texts.append(rec_text.strip())
        for rec_score in res['rec_scores']:
            confidence_scores.append(rec_score)
    all_texts = ' '.join(texts)
    avg_confidence_score = np.mean(confidence_scores)
    return (all_texts, avg_confidence_score)

def compute_metric(original_text, ocr_text):
    import Levenshtein
    import editdistance
    metrics = {}
    distance = Levenshtein.distance(original_text, ocr_text)
    cer = distance / len(original_text)
    metrics['distance'] = distance
    metrics['cer'] = cer
    original_words = original_text.strip().lower().split()
    ocr_words = ocr_text.strip().lower().split()
    wer = editdistance.eval(original_words, ocr_words) / len(original_words)
    metrics['wer'] = wer
    return metrics
