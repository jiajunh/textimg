"""Shared OCR backends with explicit legacy versus ablation configuration."""
from abc import ABC, abstractmethod

class OCRBackend(ABC):

    @abstractmethod
    def infer_one(self, image_path):
        pass

class PaddleOCRBackend(OCRBackend):

    def __init__(self, args):
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            # Avoid the Paddle 3.3.x oneDNN/PIR attribute conversion failure.
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def infer_one(self, image_path):
        result = self.ocr.predict(input=image_path)
        return result

class DeepSeekOCRBackend(OCRBackend):

    def __init__(self, args):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.model_name = 'deepseek-ai/DeepSeek-OCR'
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name,
            _attn_implementation='flash_attention_2',
            trust_remote_code=True,
            use_safetensors=True,
        )
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        self.output_path = args.output_dir
        self.prompt = '<image>\n<|grounding|>OCR this image.'
        self.ablation = args.ocr_mode == 'ablation'

    def infer_one(self, image_path):
        result = self.model.infer(
            self.tokenizer,
            prompt=self.prompt,
            image_file=image_path,
            output_path=self.output_path,
            base_size=1024,
            image_size=1024 if self.ablation else 640,
            crop_mode=not self.ablation,
            save_results=not self.ablation,
            test_compress=False,
            **({'eval_mode': True} if self.ablation else {}),
        )
        return result

def get_backend(args):
    if args.ocr_backend == 'PaddleOCR':
        return PaddleOCRBackend(args)
    elif args.ocr_backend == 'DeepSeekOCR':
        return DeepSeekOCRBackend(args)
    else:
        raise ValueError(f'Unknown backend: {args.ocr_backend}')
