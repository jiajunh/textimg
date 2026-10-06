"""Provider routing for unchanged extraction and scoring prompts.

Gemini is opt-in and credentials are only read when its adapter is selected.
"""
import base64
import os
from types import SimpleNamespace


def required_key(name):
    value = os.environ.get(name)
    if not value:
        raise ValueError(f"Set {name} before using this provider.")
    return value


class GeminiResponses:
    """Small bridge for the subset of Responses calls used by legacy judges."""
    def __init__(self):
        from google import genai
        self.client = genai.Client(api_key=required_key("GOOGLE_API_KEY"))
        self.responses = self

    def create(self, *, model, input, reasoning=None):
        from google.genai import types
        if isinstance(input, str):
            contents = input
        else:
            contents = []
            for message in input:
                for part in message["content"]:
                    if part["type"] == "input_text":
                        contents.append(part["text"])
                    elif part["type"] == "input_image":
                        header, payload = part["image_url"].split(",", 1)
                        contents.append(types.Part.from_bytes(
                            data=base64.b64decode(payload),
                            mime_type=header.removeprefix("data:").split(";")[0]))
                    else:
                        raise ValueError(f"Unsupported input part: {part['type']}")
        # OpenAI's reasoning effort is not translated into a different model's
        # budget. Gemini uses its model default; this is recorded in the manifest.
        response = self.client.models.generate_content(model=model, contents=contents)
        if response.text is None:
            raise RuntimeError("Gemini judge returned no text; inspect the provider response.")
        return SimpleNamespace(output_text=response.text)


class CountedResponses:
    def __init__(self, args, client):
        self.args = args
        self.client = client
        self.responses = self

    def create(self, **kwargs):
        count = getattr(self.args, "judge_calls", 0)
        cap = getattr(self.args, "max_judge_calls", None)
        if cap is not None and count >= cap:
            raise RuntimeError(f"Reached --max-judge-calls={cap}; run stopped before the next call.")
        self.args.judge_calls = count + 1
        return self.client.responses.create(**kwargs)


def judge_client(args):
    if args.judge_provider == "gemini":
        client = GeminiResponses()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=required_key("OPENAI_API_KEY"))
    return CountedResponses(args, client)
