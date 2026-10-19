def gemini_pro_image_generation(args, prompt, text):
    from google.genai import types
    from .gemini_client import gemini_client, final_image

    with gemini_client(args) as client:
        response = client.models.generate_content(
            model=args.model,
            contents=[f"{prompt}\n\n{text}"],
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
                image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K")))
    return final_image(response)
