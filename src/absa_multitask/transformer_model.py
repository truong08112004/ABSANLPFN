from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

try:
    from transformers import AutoModel
except Exception as e:  # pragma: no cover
    AutoModel = None  # type: ignore[assignment]


@dataclass(frozen=True)
class TransformerOut:
    logits_aspect: torch.Tensor
    logits_sentiment: torch.Tensor


class TransformerMtlModel(nn.Module):
    """
    Encoder Transformer chung (English) + 2 head softmax:
      - aspect classification
      - sentiment classification

    Dùng representation CLS (token đầu) hoặc pooler_output nếu có.
    """

    def __init__(
        self,
        *,
        model_name: str,
        aspect_classes: int,
        sentiment_classes: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if AutoModel is None:
            raise RuntimeError("transformers chưa sẵn sàng. Hãy cài dependency `transformers`.")

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(getattr(self.encoder.config, "hidden_size"))
        self.dropout = nn.Dropout(dropout)
        self.head_aspect = nn.Linear(hidden, aspect_classes)
        self.head_sentiment = nn.Linear(hidden, sentiment_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> TransformerOut:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            rep = out.pooler_output
        else:
            rep = out.last_hidden_state[:, 0, :]
        rep = self.dropout(rep)
        return TransformerOut(
            logits_aspect=self.head_aspect(rep),
            logits_sentiment=self.head_sentiment(rep),
        )

