import base64
from textimg.providers.judge import judge_client

def encode_image(image_path):
    with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def gpt_judge_image(args, img_path):
    client = judge_client(args)
    img_b64 = encode_image(img_path)
    prompt = (
        'You are judging the text rendering quality from the given image.\n'
        'Are the texts and symbols in the image rendered clearly and is readable by a human?\n'
        'Minor imperfections that do not affect reading are acceptable \n'
        'Return exactly one character: 1 (yes) or 0 (no). Do not output anything else.'
    )
    response = client.responses.create(model=args.judge_model, input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': f'data:image/png;base64,{img_b64}'}]}])
    results = response.output_text
    return results

def gpt_judge_clear_chars(args, img_path):
    client = judge_client(args)
    img_b64 = encode_image(img_path)
    prompt = (
        'You are evaluating text rendering quality in an image.\n'
        'Identify every visible character or symbol in the image, including: letters, numbers, punctuation '
        'and mathematical symbols.\n'
        'Count the total number of visible characters/symbols and how many of them are clearly and '
        'unambiguously readable by a human. Do not count whitespace.\n'
        '- Output exactly two integers separated by a single space.\n'
        '- First number: CLEAR\n'
        '- Second number: TOTAL\n'
        '- No words, no labels, no punctuation, no explanations.\n'
        '- Example valid output: 232 233\n'
    )
    response = client.responses.create(model=args.judge_model, input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': f'data:image/png;base64,{img_b64}'}]}])
    results = response.output_text
    return results

def gpt_judge_clear_areas(args, img_path):
    client = judge_client(args)
    img_b64 = encode_image(img_path)
    prompt = (
        'You are evaluating text rendering quality in an image.\n'
        'Identify every visible character or symbol in the image, including: letters, numbers, punctuation '
        'and mathematical symbols.\n'
        'Output the ratio of area where the characters are clearly and unambiguously readable by a human. Do '
        'not count whitespace.\n'
        '- Output exactly a float with 3 decimals.\n'
        '- No words, no labels, no punctuation, no explanations.\n'
        '- Example valid output: 0.930\n'
    )
    response = client.responses.create(model=args.judge_model, input=[{'role': 'user', 'content': [{'type': 'input_text', 'text': prompt}, {'type': 'input_image', 'image_url': f'data:image/png;base64,{img_b64}'}]}])
    results = response.output_text
    return results
