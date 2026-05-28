from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import onehot_pos


class CnnBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        y = x.transpose(1, 2)  # [B, C, L]
        y = self.conv(y)
        y = F.relu(y)
        y = y.transpose(1, 2)  # [B, L, out]
        y = self.dropout(y)
        return self.norm(y)


@dataclass(frozen=True)
class ForwardOut:
    logits_aspect: torch.Tensor
    logits_sentiment: torch.Tensor


class MultitaskAbsaModel(nn.Module):
    """
    Flow bám paper:
      word embedding (100d) + POS vector (one-hot) -> concat
      -> 3-layer CNN -> BiLSTM -> average pooling -> dense -> softmax heads.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pos_vocab_size: int,
        aspect_classes: int,
        sentiment_classes: int,
        max_len: int,
        word_dim: int = 100,
        cnn_dim: int = 128,
        lstm_hidden: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.word_dim = word_dim
        self.pos_vocab_size = pos_vocab_size

        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=0)

        in_dim = word_dim + pos_vocab_size
        self.cnn1 = CnnBlock(in_dim, cnn_dim, kernel_size=3, dropout=dropout)
        self.cnn2 = CnnBlock(cnn_dim, cnn_dim, kernel_size=3, dropout=dropout)
        self.cnn3 = CnnBlock(cnn_dim, cnn_dim, kernel_size=3, dropout=dropout)

        self.bilstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden * 2, lstm_hidden * 2)
        self.head_aspect = nn.Linear(lstm_hidden * 2, aspect_classes)
        self.head_sentiment = nn.Linear(lstm_hidden * 2, sentiment_classes)

    def forward(self, word_ids: torch.Tensor, pos_ids: torch.Tensor) -> ForwardOut:
        # word_ids,pos_ids: [B, L]
        w = self.word_emb(word_ids)  # [B, L, word_dim]
        p = onehot_pos(pos_ids, self.pos_vocab_size)  # [B, L, pos_vocab]
        x = torch.cat([w, p], dim=-1)  # [B, L, word_dim+pos]

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        x, _ = self.bilstm(x)  # [B, L, 2H]
        # average pooling over sequence length
        x = x.mean(dim=1)  # [B, 2H]
        x = self.dropout(F.relu(self.fc(x)))
        return ForwardOut(
            logits_aspect=self.head_aspect(x),
            logits_sentiment=self.head_sentiment(x),
        )


class CnnOnlyMultitaskModel(nn.Module):
    """
    Baseline dễ train:
      word embedding (100d) + POS one-hot -> concat
      -> 3-layer CNN -> average pooling -> dense -> 2 softmax heads.

    Khác paper: bỏ BiLSTM (không mô hình hoá phụ thuộc dài hạn).
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        pos_vocab_size: int,
        aspect_classes: int,
        sentiment_classes: int,
        max_len: int,
        word_dim: int = 100,
        cnn_dim: int = 128,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.max_len = max_len
        self.word_dim = word_dim
        self.pos_vocab_size = pos_vocab_size

        self.word_emb = nn.Embedding(vocab_size, word_dim, padding_idx=0)

        in_dim = word_dim + pos_vocab_size
        self.cnn1 = CnnBlock(in_dim, cnn_dim, kernel_size=3, dropout=dropout)
        self.cnn2 = CnnBlock(cnn_dim, cnn_dim, kernel_size=3, dropout=dropout)
        self.cnn3 = CnnBlock(cnn_dim, cnn_dim, kernel_size=3, dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(cnn_dim, cnn_dim)
        self.head_aspect = nn.Linear(cnn_dim, aspect_classes)
        self.head_sentiment = nn.Linear(cnn_dim, sentiment_classes)

    def forward(self, word_ids: torch.Tensor, pos_ids: torch.Tensor) -> ForwardOut:
        w = self.word_emb(word_ids)  # [B, L, word_dim]
        p = onehot_pos(pos_ids, self.pos_vocab_size)  # [B, L, pos_vocab]
        x = torch.cat([w, p], dim=-1)  # [B, L, word_dim+pos]

        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)

        x = x.mean(dim=1)  # [B, cnn_dim]
        x = self.dropout(F.relu(self.fc(x)))
        return ForwardOut(logits_aspect=self.head_aspect(x), logits_sentiment=self.head_sentiment(x))

