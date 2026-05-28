from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LABEL_UNK, Absadataset, build_vocab, collate_fn
from .model import CnnOnlyMultitaskModel, MultitaskAbsaModel
from .textproc import build_examples


@dataclass(frozen=True)
class VocabPack:
    word: Dict[str, int]
    pos: Dict[str, int]
    aspect: Dict[str, int]
    sentiment: Dict[str, int]


def build_label_vocab(labels: List[str]) -> Dict[str, int]:
    norm = [str(x).strip() for x in labels]
    uniq = sorted(set(norm))
    # Reserve an UNK label to avoid KeyError on unseen labels in val/test.
    out: Dict[str, int] = {LABEL_UNK: 0}
    for lab in uniq:
        if lab == LABEL_UNK:
            continue
        out.setdefault(lab, len(out))
    return out


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", default="reviewText")
    p.add_argument("--aspect-col", default="aspect")
    p.add_argument("--sentiment-col", default="sentiment")
    p.add_argument("--arch", default="paper", choices=["paper", "cnn_only"])
    p.add_argument("--max-len", type=int, default=62)
    p.add_argument("--min-word-freq", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.aspect_col, args.sentiment_col])

    texts = df[args.text_col].astype(str).tolist()
    aspects = df[args.aspect_col].astype(str).map(lambda x: str(x).strip()).tolist()
    sentiments = df[args.sentiment_col].astype(str).map(lambda x: str(x).strip()).tolist()

    examples = build_examples(texts, aspects, sentiments)
    if len(examples) < 10:
        raise SystemExit(f"Quá ít sample hợp lệ sau preprocessing: {len(examples)}")

    ex_train, ex_val = train_test_split(examples, test_size=0.15, random_state=42, shuffle=True)

    word_vocab = build_vocab([ex.tokens for ex in ex_train], min_freq=args.min_word_freq)
    pos_vocab = build_vocab([ex.pos for ex in ex_train], min_freq=1)
    aspect_vocab = build_label_vocab([ex.aspect for ex in ex_train])
    sentiment_vocab = build_label_vocab([ex.sentiment for ex in ex_train])

    vp = VocabPack(word=word_vocab, pos=pos_vocab, aspect=aspect_vocab, sentiment=sentiment_vocab)

    ds_train = Absadataset(
        ex_train,
        word_vocab=vp.word,
        pos_vocab=vp.pos,
        aspect_vocab=vp.aspect,
        sentiment_vocab=vp.sentiment,
        max_len=args.max_len,
    )
    ds_val = Absadataset(
        ex_val,
        word_vocab=vp.word,
        pos_vocab=vp.pos,
        aspect_vocab=vp.aspect,
        sentiment_vocab=vp.sentiment,
        max_len=args.max_len,
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu")
    )

    if args.arch == "paper":
        model = MultitaskAbsaModel(
            vocab_size=len(vp.word),
            pos_vocab_size=len(vp.pos),
            aspect_classes=len(vp.aspect),
            sentiment_classes=len(vp.sentiment),
            max_len=args.max_len,
            dropout=args.dropout,
        ).to(device)
    else:
        model = CnnOnlyMultitaskModel(
            vocab_size=len(vp.word),
            pos_vocab_size=len(vp.pos),
            aspect_classes=len(vp.aspect),
            sentiment_classes=len(vp.sentiment),
            max_len=args.max_len,
            dropout=args.dropout,
        ).to(device)

    opt = torch.optim.RMSprop(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc_a = 0.0
        tr_acc_s = 0.0
        n_batches = 0
        for batch in tqdm(dl_train, desc=f"train epoch {epoch}"):
            n_batches += 1
            word_ids = batch.word_ids.to(device)
            pos_ids = batch.pos_ids.to(device)
            y_aspect = batch.y_aspect.to(device)
            y_sent = batch.y_sentiment.to(device)

            out = model(word_ids, pos_ids)
            loss_aspect = loss_fn(out.logits_aspect, y_aspect)
            loss_sent = loss_fn(out.logits_sentiment, y_sent)
            loss = loss_aspect + loss_sent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            tr_loss += float(loss.item())
            tr_acc_a += accuracy_from_logits(out.logits_aspect, y_aspect)
            tr_acc_s += accuracy_from_logits(out.logits_sentiment, y_sent)

        tr_loss /= max(1, n_batches)
        tr_acc_a /= max(1, n_batches)
        tr_acc_s /= max(1, n_batches)

        model.eval()
        va_loss = 0.0
        va_acc_a = 0.0
        va_acc_s = 0.0
        va_batches = 0
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"val epoch {epoch}"):
                va_batches += 1
                word_ids = batch.word_ids.to(device)
                pos_ids = batch.pos_ids.to(device)
                y_aspect = batch.y_aspect.to(device)
                y_sent = batch.y_sentiment.to(device)

                out = model(word_ids, pos_ids)
                loss = loss_fn(out.logits_aspect, y_aspect) + loss_fn(out.logits_sentiment, y_sent)
                va_loss += float(loss.item())
                va_acc_a += accuracy_from_logits(out.logits_aspect, y_aspect)
                va_acc_s += accuracy_from_logits(out.logits_sentiment, y_sent)

        va_loss /= max(1, va_batches)
        va_acc_a /= max(1, va_batches)
        va_acc_s /= max(1, va_batches)

        print(
            f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"train_acc_aspect={tr_acc_a:.4f} val_acc_aspect={va_acc_a:.4f} "
            f"train_acc_sent={tr_acc_s:.4f} val_acc_sent={va_acc_s:.4f}"
        )


if __name__ == "__main__":
    main()

