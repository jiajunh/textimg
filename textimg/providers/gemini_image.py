def gemini_image_generation(args, prompt, text):
    from google.genai import types
    from .gemini_client import gemini_client, final_image

    with gemini_client(args) as client:
        response = client.models.generate_content(
            model=args.model, contents=[f"{prompt}\n\n{text}"],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio="1:1"),
                response_modalities=["IMAGE"],
                safety_settings=[types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_LOW_AND_ABOVE)]))
    return final_image(response)
