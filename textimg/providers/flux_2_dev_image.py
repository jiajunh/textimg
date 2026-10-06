import time
import os
import requests
from io import BytesIO
from PIL import Image
FLUX_API_KEY = os.environ.get('FLUX_API_KEY', '')

def retrieve_result(polling_url, max_retries=100, poll_interval=0.5, timeout=10.0):
    headers = {'accept': 'application/json', 'x-key': FLUX_API_KEY}
    for attempt in range(max_retries):
        try:
            resp = requests.get(polling_url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
        except Exception as e:
            print(f'[Polling error] {e}')
            time.sleep(poll_interval)
            continue
        status = result.get('status')
        if status == 'Ready':
            img_url = result['result']['sample']
            if img_url is None:
                print('[Error] Ready but no sample URL found')
            return img_url
        if status == 'Failed':
            return None
        time.sleep(poll_interval)
    print('[Timeout] Max retries exceeded')
    return None

def flux2_pro_image_generation(args, prompt, text):
    input_prompt = f'{prompt}\n\n{text}'
    response = requests.post('https://api.bfl.ai/v1/flux-2-pro', headers={'accept': 'application/json', 'x-key': FLUX_API_KEY, 'Content-Type': 'application/json'}, json={'prompt': input_prompt, 'width': 1024, 'height': 1024, 'safety_tolerance': 5, 'output_format': 'png'}).json()
    request_id = response['id']
    polling_url = response['polling_url']
    cost = response.get('cost')
    img_url = retrieve_result(polling_url)
    image = None
    if img_url:
        image = Image.open(BytesIO(requests.get(img_url).content))
    return image
