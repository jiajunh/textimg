import os
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig

def load_qwen_model():
    model_name = 'Qwen/Qwen3-VL-4B-Instruct'
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_name, device_map='auto', quantization_config=bnb_cfg, attn_implementation='sdpa')
    model.eval()
    processor = AutoProcessor.from_pretrained(model_name)
    return (model, processor)

def judge_rendering_batch(model, processor, image_paths):
    prompt = (
        'You are judging the text rendering quality from the given image.\n'
        'Is all the texts in the image rendered clearly and can be unambiguously readable by a human?\n'
        'Return exactly one character: 1 (yes) or 0 (no). Do not output anything else.'
    )
    results = []
    imgs = []
    for p in image_paths:
        with Image.open(p) as im:
            imgs.append(im.convert('RGB'))
    messages = [[{'role': 'user', 'content': [{'type': 'image', 'image': im}, {'type': 'text', 'text': prompt}]}] for im in imgs]
    inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, padding=True, return_tensors='pt', return_dict=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=1, do_sample=False, use_cache=False)
    gen = out[:, inputs['input_ids'].shape[1]:]
    decoded = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
    results = []
    for d in decoded:
        d = d.strip()
        if d.startswith('1'):
            results.append('1')
        elif d.startswith('0'):
            results.append('0')
        else:
            results.append('1' if '1' in d else '0')
    return results
