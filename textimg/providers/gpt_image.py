import os
import base64
from openai import OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

def gpt_image_generation(args, prompt, text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    input_prompt = f'{prompt}\n\n{text}'
    result = client.images.generate(
        model=args.model,
        prompt=input_prompt,
        moderation='low',
        quality=args.quality,
        size='1024x1024',
    )
    image_bytes = base64.b64decode(result.data[0].b64_json)
    return image_bytes
