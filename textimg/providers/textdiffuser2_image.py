"""Legacy adapter; imported only when selected."""

import os
import torch
from diffusers import DiffusionPipeline

def load_textdiffuser2_pipe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = DiffusionPipeline.from_pretrained('JingyeChen22/textdiffuser2-full-ft', torch_dtype=torch.float16).to(device)
    return pipe

def textdiffuser2_image_generation(args, prompt, text):
    input_prompt = f'{prompt}\n\n{text}'
    print(input_prompt)
    image = args.textdiffuser2_pipe(prompt=input_prompt, width=1024, height=1024).images[0]
    return image
