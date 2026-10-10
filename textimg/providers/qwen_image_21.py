def load_qwen_image_21(args):
    import torch
    from diffusers import QwenImage21Pipeline
    dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
    kwargs = {"dtype": dtype}
    if args.revision:
        kwargs["revision"] = args.revision
    pipe = QwenImage21Pipeline.from_pretrained(args.model_path or "Qwen/Qwen-Image-2.1", **kwargs)
    if args.cpu_offload:
        if args.device != "cuda":
            raise ValueError("CPU offload currently requires --device cuda.")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)
    return pipe


def qwen_image_21_generation(args, prompt, text):
    import torch
    kwargs = {"prompt": f"{prompt}\n\n{text}", "width": args.width,
              "height": args.height, "num_inference_steps": args.inference_steps,
              "num_images_per_prompt": 1,
              "true_cfg_scale": 1.0}
    if args.generation_seed is not None:
        kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.generation_seed)
    try:
        with torch.inference_mode():
            return args.qwen_image_21_pipe(**kwargs).images[0]
    finally:
        if args.device == "cuda" and torch.cuda.is_available():
            # Release unused cached blocks; model weights remain loaded.
            torch.cuda.empty_cache()
