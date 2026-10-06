import json
import os
import requests
from PIL import Image
from io import BytesIO
import dashscope
from dashscope import MultiModalConversation
QWEN_API_KEY = os.environ.get('DASHSCOPE_API_KEY', '')

def qwen_image_image_generation(args, prompt, text):
    dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'
    input_prompt = f'{prompt}\n\n{text}'
    messages = [{'role': 'user', 'content': [{'text': input_prompt}]}]
    response = MultiModalConversation.call(api_key=QWEN_API_KEY, model=args.model, messages=messages, result_format='message', stream=False, watermark=False, size='1328*1328')
    image = None
    if response.status_code == 200:
        image_url = response['output']['choices'][0]['message']['content'][0]['image']
        image = Image.open(BytesIO(requests.get(image_url).content))
    else:
        print(f'HTTP status code: {response.status_code}')
        print(f'Error code: {response.code}')
        print(f'Error message: {response.message}')
    return image
