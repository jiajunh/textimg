"""Legacy adapter; imported only when selected."""

import os
from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch

def load_sd_large_pipe():
    nf4_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.bfloat16)
    model_nf4 = SD3Transformer2DModel.from_pretrained('stabilityai/stable-diffusion-3.5-large', subfolder='transformer', quantization_config=nf4_config, torch_dtype=torch.bfloat16)
    pipeline = StableDiffusion3Pipeline.from_pretrained('stabilityai/stable-diffusion-3.5-large', transformer=model_nf4, torch_dtype=torch.bfloat16)
    pipeline.to('cuda')
    return pipeline

def sd_large_image_generation(args, prompt, text):
    input_prompt = f'{prompt}\n\n{text}'
    print(input_prompt)
    clip_input = f'generate an pure text image, containing only the following text.\n\n{text}'
    image = args.sd_3_5_pipe(prompt=clip_input, prompt_3=input_prompt, num_inference_steps=28, guidance_scale=4.5, max_sequence_length=512).images[0]
    return image
