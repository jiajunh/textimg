import base64
import os
import time
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
        # GPT-4 and GPT-3.5 judges do not support reasoning effort.
        model = kwargs.get("model", "")
        if self.args.judge_provider == "openai" and model.startswith(
            ("gpt-4", "chatgpt-4o", "gpt-3.5")
        ):
            kwargs.pop("reasoning", None)
        count = getattr(self.args, "judge_calls", 0)
        cap = getattr(self.args, "max_judge_calls", None)
        if cap is not None and count >= cap:
            raise RuntimeError(f"Reached --max-judge-calls={cap}; run stopped before the next call.")
        last_finished = getattr(self.args, "_judge_last_finished", None)
        if last_finished is not None:
            remaining = getattr(self.args, "judge_call_delay", 1.0) - (time.monotonic() - last_finished)
            if remaining > 0:
                time.sleep(remaining)
        self.args.judge_calls = count + 1
        try:
            return self.client.responses.create(**kwargs)
        finally:
            self.args._judge_last_finished = time.monotonic()


def judge_client(args):
    if args.judge_provider == "gemini":
        client = GeminiResponses()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=required_key("OPENAI_API_KEY"))
    return CountedResponses(args, client)
