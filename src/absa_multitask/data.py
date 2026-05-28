from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .textproc import Example


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


def build_vocab(seqs: Sequence[Sequence[str]], *, min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for s in seqs:
        for tok in s:
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, c in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        if c < min_freq:
            continue
        if tok in vocab:
            continue
        vocab[tok] = len(vocab)
    return vocab


def encode_seq(seq: Sequence[str], vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(t, vocab[UNK_TOKEN]) for t in seq[:max_len]]
    if len(ids) < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids


def onehot_pos(pos_ids: torch.Tensor, pos_vocab_size: int) -> torch.Tensor:
    # pos_ids: [B, L]
    b, l = pos_ids.shape
    out = torch.zeros((b, l, pos_vocab_size), dtype=torch.float32, device=pos_ids.device)
    out.scatter_(2, pos_ids.unsqueeze(-1), 1.0)
    return out


@dataclass(frozen=True)
class EncodedBatch:
    word_ids: torch.Tensor  # [B, L]
    pos_ids: torch.Tensor  # [B, L]
    y_aspect: torch.Tensor  # [B]
    y_sentiment: torch.Tensor  # [B]


class Absadataset(Dataset):
    def __init__(
        self,
        examples: Sequence[Example],
        *,
        word_vocab: Dict[str, int],
        pos_vocab: Dict[str, int],
        aspect_vocab: Dict[str, int],
        sentiment_vocab: Dict[str, int],
        max_len: int,
    ) -> None:
        self.examples = list(examples)
        self.word_vocab = word_vocab
        self.pos_vocab = pos_vocab
        self.aspect_vocab = aspect_vocab
        self.sentiment_vocab = sentiment_vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int], int, int]:
        ex = self.examples[idx]
        w = encode_seq(ex.tokens, self.word_vocab, self.max_len)
        p = encode_seq(ex.pos, self.pos_vocab, self.max_len)
        ya = self.aspect_vocab[ex.aspect]
        ys = self.sentiment_vocab[ex.sentiment]
        return w, p, ya, ys


def collate_fn(items: Sequence[Tuple[List[int], List[int], int, int]]) -> EncodedBatch:
    w, p, ya, ys = zip(*items, strict=False)
    return EncodedBatch(
        word_ids=torch.tensor(np.array(w), dtype=torch.long),
        pos_ids=torch.tensor(np.array(p), dtype=torch.long),
        y_aspect=torch.tensor(np.array(ya), dtype=torch.long),
        y_sentiment=torch.tensor(np.array(ys), dtype=torch.long),
    )

