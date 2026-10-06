import os
from google import genai
from google.genai import types
from PIL import Image
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')

def gemini_image_generation(args, prompt, text):
    client = genai.Client(api_key=GOOGLE_API_KEY)
    input_prompt = f'{prompt}\n\n{text}'
    response = client.models.generate_content(model=args.model, contents=[input_prompt], config=types.GenerateContentConfig(image_config=types.ImageConfig(aspect_ratio='1:1'), response_modalities=['Image'], safety_settings=[types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE)]))
    if response.parts is None:
        return None
    for part in response.parts:
        if part.inline_data:
            image = part.as_image()
        else:
            image = None
    return image
