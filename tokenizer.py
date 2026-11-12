"""Small word-level tokenizer (frequency-ranked vocab, 0 = padding, 1 = OOV).

Self-contained so it saves/loads as plain JSON and doesn't depend on the
Keras Tokenizer, which was removed in Keras 3.
"""
import json
from collections import Counter

import numpy as np

PAD, OOV = "<pad>", "<oov>"


class WordTokenizer:
    def __init__(self, max_vocab: int):
        self.max_vocab = max_vocab
        self.word_index: dict[str, int] = {}

    def fit(self, texts):
        counts = Counter(w for t in texts for w in t.split())
        self.word_index = {PAD: 0, OOV: 1}
        for word, _ in counts.most_common(self.max_vocab - 2):
            self.word_index[word] = len(self.word_index)
        return self

    def encode(self, texts, max_len: int) -> np.ndarray:
        """Token ids, post-padded / truncated to max_len."""
        out = np.zeros((len(texts), max_len), dtype="int32")
        for row, text in enumerate(texts):
            ids = [self.word_index.get(w, 1) for w in text.split()][:max_len]
            out[row, : len(ids)] = ids
        return out

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"max_vocab": self.max_vocab, "word_index": self.word_index}, f)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["max_vocab"])
        tok.word_index = data["word_index"]
        return tok
