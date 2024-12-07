import tiktoken
from transformers import AutoTokenizer

TIKTOKEN_MODELS = [
    "gpt-4o", "gpt-4o-mini"
]

def init_tiktoken_tokenizer(model: str):
    if model not in TIKTOKEN_MODELS:
        raise ValueError("model should be one of: {}".format(",".join(TIKTOKEN_MODELS)))
    return tiktoken.get_encoding("cl100k_base")

def init_hf_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)

def tiktoken_length(tokenizer, text: str) -> int:
    tokens = tokenizer.encode(text, disallowed_special=())
    return len(tokens)

def hf_length(tokenizer, text: str) -> int:
    return len(tokenizer(text)['input_ids'])