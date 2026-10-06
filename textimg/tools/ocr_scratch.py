import os
import json
import argparse
from abc import ABC, abstractmethod
from tqdm import tqdm

import torch

from paddleocr import PaddleOCR
from transformers import AutoModel, AutoTokenizer



class OCRBackend(ABC):
    @abstractmethod
    def infer_one(self, image_path):
        pass


class PaddleOCRBackend(OCRBackend):
    def __init__(self, args):
        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    def infer_one(self, image_path):
        result = self.ocr.predict(input=image_path)
        return result


class DeepSeekOCRBackend(OCRBackend):
    def __init__(self, args):
        self.model_name = "deepseek-ai/DeepSeek-OCR"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            self.model_name, 
            _attn_implementation='flash_attention_2', 
            trust_remote_code=True, 
            use_safetensors=True)
        self.model = self.model.eval().cuda().to(torch.bfloat16)
        self.output_path = args.output_dir
        self.prompt = "<image>\n<|grounding|>OCR this image."

    def infer_one(self, image_path):
        result = self.model.infer(
            self.tokenizer, 
            prompt=self.prompt, 
            image_file=image_path, 
            output_path=self.output_path,
            base_size=1024, 
            image_size=640, 
            crop_mode=True, 
            save_results=True, 
            test_compress=False
        )
        return result


def get_backend(args):
    if args.ocr_backend == "PaddleOCR":
        return PaddleOCRBackend(args)
    elif args.ocr_backend == "DeepSeekOCR":
        return DeepSeekOCRBackend(args)
    else:
        raise ValueError(f"Unknown backend: {args.ocr_backend}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_type", type=str, default="identical", choices=["identical", "reasoning", "multilingual"])
    parser.add_argument("--img_dir", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./../results/")
    parser.add_argument("--ocr_backend", type=str, default="PaddleOCR", choices=["PaddleOCR", "DeepSeekOCR"])

    # identical and multilingual settings
    parser.add_argument('--text_length', nargs="*", type=int, default=[64, 128, 256, 512])

    # multilingual settings
    parser.add_argument('--languages', nargs="*", type=str, default=["ar", "en", "fr", "ja", "ko", "zh"])

    # reasoning settings
    parser.add_argument('--levels', nargs="*", type=int, default=[1, 2, 3, 4, 5])

    args = parser.parse_args()
    print(args)

    args.output_dir = os.path.join(args.output_dir, args.dataset_type, args.ocr_backend)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


    raw_text = 'Senjō no Valkyria 3 : Unrecorded Chronicles ( Japanese : 戦場のヴァルキュリア3 , lit . Valkyria of the Battlefield 3 ) , commonly referred to as Valkyria Chronicles III outside Japan , is a tactical role @-@ playing video game developed by Sega and Media.Vision for the PlayStation Portable . Released in January 2011 in Japan , it is the third game in the Valkyria series . Employing the same fusion of tactical and real @-@ time gameplay as its predecessors , the story runs parallel to the first game and follows the " Nameless " , a penal military unit serving the nation of Gallia during the Second Europan War who perform secret black operations and are pitted against the Imperial unit " Calamaty Raven'
    ocr_img_path = "./test_img.png"


    # ocr = PaddleOCR(
    #     use_doc_orientation_classify=False,
    #     use_doc_unwarping=False,
    #     use_textline_orientation=False)

    # # Run OCR inference on a sample image 
    # result = ocr.predict(
    #     input=ocr_img_path)

    # # Visualize the results and save the JSON results
    # for res in result:
    #     res.print()
