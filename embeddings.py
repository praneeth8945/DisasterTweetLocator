"""GloVe embedding loading (paper Sec. III-C)."""
import numpy as np


def load_glove(path, dim: int) -> dict:
    """Read a GloVe .txt file into {word: vector}."""
    index = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            if len(parts) != dim + 1:
                continue
            index[parts[0]] = np.asarray(parts[1:], dtype="float32")
    return index


def build_embedding_matrix(word_index: dict, glove: dict, dim: int, max_vocab: int):
    """Matrix whose row i is the GloVe vector for token id i (zeros if unknown).

    Returns (matrix, coverage) where coverage is the share of the vocabulary
    found in GloVe.
    """
    vocab_size = min(max_vocab, len(word_index) + 1)
    matrix = np.zeros((vocab_size, dim), dtype="float32")
    hits = 0
    for word, i in word_index.items():
        if i >= vocab_size:
            continue
        vec = glove.get(word)
        if vec is not None:
            matrix[i] = vec
            hits += 1
    return matrix, hits / max(vocab_size - 1, 1)
