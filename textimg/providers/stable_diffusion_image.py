import os
import torch
from diffusers import StableDiffusionXLPipeline

def load_sdxl_pipe():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pipe = StableDiffusionXLPipeline.from_pretrained('stabilityai/stable-diffusion-xl-base-1.0', torch_dtype=torch.float16).to(device)
    return pipe

def sdxl_image_generation(args, prompt, text):
    input_prompt = f'{prompt}\n\n{text}'
    print(input_prompt)
    image = args.sdxl_pipe(prompt=input_prompt, width=1024, height=1024).images[0]
    return image
