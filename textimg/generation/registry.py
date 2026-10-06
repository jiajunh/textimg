"""Image adapters and optional local initializers are declared in one place."""
from importlib import import_module

MODELS = {
    "gpt-image-1.5": ("gpt_image", "gpt_image_generation", "OPENAI_API_KEY"),
    "gpt-image-2": ("gpt_image", "gpt_image_generation", "OPENAI_API_KEY"),
    "gemini-2.5-flash-image": ("gemini_image", "gemini_image_generation", "GOOGLE_API_KEY"),
    "gemini-3-pro-image": ("gemini_pro_image", "gemini_pro_image_generation", "GOOGLE_API_KEY"),
    "gemini-3-pro-image-preview": ("gemini_pro_image", "gemini_pro_image_generation", "GOOGLE_API_KEY"),
    "qwen-image": ("qwen_image_image", "qwen_image_image_generation", "DASHSCOPE_API_KEY"),
    "qwen-image-2.1": ("qwen_image_21", "qwen_image_21_generation", None),
    "flux2-pro": ("flux_2_dev_image", "flux2_pro_image_generation", "FLUX_API_KEY"),
    "stable-diffusion": ("stable_diffusion_image", "sdxl_image_generation", None),
    "sd3.5-large": ("sd_large_image", "sd_large_image_generation", None),
    "textdiffuser-2": ("textdiffuser2_image", "textdiffuser2_image_generation", None),
}
EXACT_TEXT_ONLY = {"stable-diffusion", "sd3.5-large", "textdiffuser-2"}
LOCAL_MODELS = EXACT_TEXT_ONLY | {"qwen-image-2.1"}
INITIALIZERS = {
    "stable-diffusion": ("load_sdxl_pipe", "sdxl_pipe", False),
    "sd3.5-large": ("load_sd_large_pipe", "sd_3_5_pipe", False),
    "textdiffuser-2": ("load_textdiffuser2_pipe", "textdiffuser2_pipe", False),
    "qwen-image-2.1": ("load_qwen_image_21", "qwen_image_21_pipe", True),
}


def generation_settings(args):
    """Record actual adapter settings rather than irrelevant shared CLI defaults."""
    if args.model.startswith("gpt-image"):
        return {"size": "1024x1024", "quality": args.quality, "moderation": "low"}
    if args.model == "qwen-image":
        return {"size": "1328*1328", "watermark": False,
                "endpoint": "https://dashscope-intl.aliyuncs.com/api/v1"}
    if args.model == "flux2-pro":
        return {"width": 1024, "height": 1024, "safety_tolerance": 5, "output_format": "png"}
    if args.model.startswith("gemini-3-pro-image"):
        return {"aspect_ratio": "1:1", "image_size": "1K", "response_modalities": ["TEXT", "IMAGE"]}
    if args.model == "gemini-2.5-flash-image":
        return {"aspect_ratio": "1:1", "image_size": "provider default",
                "response_modalities": ["Image"], "hate_speech_threshold": "BLOCK_LOW_AND_ABOVE"}
    if args.model == "qwen-image-2.1":
        return {"checkpoint": args.model_path or "Qwen/Qwen-Image-2.1", "revision": args.revision,
                "width": args.width, "height": args.height, "num_inference_steps": args.inference_steps,
                "true_cfg_scale": 1.0, "seed": args.generation_seed,
                "device": args.device, "cpu_offload": args.cpu_offload}
    if args.model == "sd3.5-large":
        return {"checkpoint": "stabilityai/stable-diffusion-3.5-large", "device": "cuda",
                "dtype": "bfloat16", "quantization": "nf4", "num_inference_steps": 28,
                "guidance_scale": 4.5, "max_sequence_length": 512, "size": "pipeline default",
                "prompt_routing": "legacy clip_input + full prompt_3"}
    checkpoint = {"stable-diffusion": "stabilityai/stable-diffusion-xl-base-1.0",
                  "textdiffuser-2": "JingyeChen22/textdiffuser2-full-ft"}[args.model]
    return {"checkpoint": checkpoint, "device": "CUDA if available, otherwise CPU",
            "dtype": "float16", "width": 1024, "height": 1024,
            "num_inference_steps": "pipeline default", "guidance_scale": "pipeline default"}


def get_generator_fn(args):
    module, function, key = MODELS[args.model]
    if key:
        from textimg.providers.judge import required_key
        required_key(key)
    adapter = import_module(f"textimg.providers.{module}")
    if args.model in INITIALIZERS:
        loader_name, attribute, takes_args = INITIALIZERS[args.model]
        loader = getattr(adapter, loader_name)
        setattr(args, attribute, loader(args) if takes_args else loader())
    return getattr(adapter, function)
