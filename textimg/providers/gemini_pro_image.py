def gemini_pro_image_generation(args, prompt, text):
    from google import genai
    from google.genai import types
    from .judge import required_key
    
    client = genai.Client(api_key=required_key("GOOGLE_API_KEY"))
    response = client.models.generate_content(
        model=args.model,
        contents=[f"{prompt}\n\n{text}"],
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
            image_config=types.ImageConfig(aspect_ratio="1:1", image_size="1K")))
    images = [part.as_image() for part in response.parts or []
              if part.inline_data is not None and not getattr(part, "thought", False)]
    if not images:
        raise RuntimeError("Gemini returned no final image.")
    return images[-1]
