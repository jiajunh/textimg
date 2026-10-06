"""Legacy adapter; imported only when selected."""

import os
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = 'Qwen/Qwen3-8B'

def load_qwen_model(args):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype='auto', device_map='auto')
    return (model, tokenizer)

def generate_qwen_response(args, prompt, text):
    input_text = f'{prompt}\n{text}'
    messages = [{'role': 'user', 'content': input_text}]
    text = args.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    model_inputs = args.tokenizer([text], return_tensors='pt').to(args.model.device)
    generated_ids = args.model.generate(**model_inputs, max_new_tokens=32768)
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = args.tokenizer.decode(output_ids, skip_special_tokens=True).strip('\n')
    return content
