"""Legacy adapter; imported only when selected."""

import os
import re
import base64
from openai import OpenAI
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')

def generate_gpt_response(args, prompt, text):
    client = OpenAI(api_key=OPENAI_API_KEY)
    input_text = f'{prompt}\n{text}'
    response = client.responses.create(model='gpt-5.2', input=input_text)
    return response.output_text
