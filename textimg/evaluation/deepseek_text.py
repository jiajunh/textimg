import re
import ast
from abc import ABC, abstractmethod
REFDET_RE = re.compile('<\\|ref\\|>(.*?)<\\|/ref\\|>\\s*<\\|det\\|>(\\[\\[.*?\\]\\])<\\|/det\\|>', re.S)

def extract_deepseek_ocr_result(result: str) -> str:
    if result is None:
        return ''
    result = str(result).strip()
    if not result:
        return ''
    items = []
    for text, bbox in REFDET_RE.findall(result):
        try:
            box = ast.literal_eval(bbox)[0]
        except Exception:
            continue
        items.append({'text': text.strip(), 'bbox': box})
    if not items:
        cleaned = re.sub('<\\|/?[a-zA-Z_]+\\|>', '', result)
        cleaned = cleaned.strip()
        return cleaned
    for x in items:
        if len(x['bbox']) < 2:
            return ''
    items.sort(key=lambda x: (x['bbox'][1], x['bbox'][0]))
    lines = []
    current = items[0]['text']
    prev_y = items[0]['bbox'][1]
    LINE_Y_THRESHOLD = 20
    for item in items[1:]:
        y = item['bbox'][1]
        if abs(y - prev_y) < LINE_Y_THRESHOLD:
            current += ' ' + item['text']
        else:
            lines.append(current)
            current = item['text']
        prev_y = y
    lines.append(current)
    final_text = ' '.join(lines)
    return final_text
