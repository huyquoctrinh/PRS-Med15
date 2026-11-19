from __future__ import annotations

import re


class FakeTokenizer:
    def __init__(self):
        self.vocab = {"<pad>": 0, "</s>": 1}
        self.eos_token_id = 1
        self.pad_token_id = 0
        self.chat_template = "fake"

    def _tokens(self, text):
        return re.findall(r"\{(?:GLOBAL|DETAIL|REGION)\}|\[SEG\]|<[^>]+>|S\d+:E\d+|\w+|[^\s]", text)

    def __call__(self, text, add_special_tokens=False):
        ids = []
        for token in self._tokens(text):
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
            ids.append(self.vocab[token])
        return {"input_ids": ids}

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=True):
        text = " ".join(f"[INST] {x['content']} [/INST]" for x in messages)
        return self(text)["input_ids"] if tokenize else text
