from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LABEL_UNK, Absadataset, build_vocab, collate_fn
from .model import MultitaskAbsaModel
from .textproc import build_examples
from .transformer_model import TransformerMtlModel

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


def build_label_vocab(labels: List[str]) -> Dict[str, int]:
    norm = [str(x).strip() for x in labels]
    uniq = sorted(set(norm))
    out: Dict[str, int] = {LABEL_UNK: 0}
    for lab in uniq:
        if lab == LABEL_UNK:
            continue
        out.setdefault(lab, len(out))
    return out


@dataclass(frozen=True)
class RunResult:
    arch: str
    epochs: int
    val_acc_aspect: float
    val_f1_aspect_macro: float
    val_acc_sent: float
    val_f1_sent_macro: float


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _train_one(
    *,
    arch: str,
    dl_train: DataLoader,
    dl_val: DataLoader,
    vocab_size: int,
    pos_vocab_size: int,
    aspect_classes: int,
    sentiment_classes: int,
    max_len: int,
    dropout: float,
    epochs: int,
    lr: float,
    device: torch.device,
    transformer_name: str,
) -> RunResult:
    if arch == "paper":
        model = MultitaskAbsaModel(
            vocab_size=vocab_size,
            pos_vocab_size=pos_vocab_size,
            aspect_classes=aspect_classes,
            sentiment_classes=sentiment_classes,
            max_len=max_len,
            dropout=dropout,
        ).to(device)
    elif arch == "transformer_mtl":
        model = TransformerMtlModel(
            model_name=transformer_name,
            aspect_classes=aspect_classes,
            sentiment_classes=sentiment_classes,
            dropout=0.1,
        ).to(device)
    else:
        raise ValueError(f"arch không hỗ trợ: {arch}")

    opt = torch.optim.RMSprop(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(dl_train, desc=f"{arch} train epoch {epoch}"):
            if arch == "transformer_mtl":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                y_aspect = batch["y_aspect"].to(device)
                y_sent = batch["y_sentiment"].to(device)
                out = model(input_ids, attention_mask)
            else:
                word_ids = batch.word_ids.to(device)
                pos_ids = batch.pos_ids.to(device)
                y_aspect = batch.y_aspect.to(device)
                y_sent = batch.y_sentiment.to(device)
                out = model(word_ids, pos_ids)
            loss = loss_fn(out.logits_aspect, y_aspect) + loss_fn(out.logits_sentiment, y_sent)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

    # Evaluate
    model.eval()
    y_a_true: List[int] = []
    y_a_pred: List[int] = []
    y_s_true: List[int] = []
    y_s_pred: List[int] = []
    with torch.no_grad():
        for batch in tqdm(dl_val, desc=f"{arch} val"):
            if arch == "transformer_mtl":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids, attention_mask)
                y_a_true.extend(batch["y_aspect"].cpu().numpy().tolist())
                y_s_true.extend(batch["y_sentiment"].cpu().numpy().tolist())
            else:
                word_ids = batch.word_ids.to(device)
                pos_ids = batch.pos_ids.to(device)
                out = model(word_ids, pos_ids)
                y_a_true.extend(batch.y_aspect.cpu().numpy().tolist())
                y_s_true.extend(batch.y_sentiment.cpu().numpy().tolist())
            y_a_pred.extend(out.logits_aspect.argmax(dim=-1).cpu().numpy().tolist())
            y_s_pred.extend(out.logits_sentiment.argmax(dim=-1).cpu().numpy().tolist())

    val_acc_aspect = float(accuracy_score(y_a_true, y_a_pred))
    val_f1_aspect_macro = float(f1_score(y_a_true, y_a_pred, average="macro"))
    val_acc_sent = float(accuracy_score(y_s_true, y_s_pred))
    val_f1_sent_macro = float(f1_score(y_s_true, y_s_pred, average="macro"))
    return RunResult(
        arch=arch,
        epochs=epochs,
        val_acc_aspect=val_acc_aspect,
        val_f1_aspect_macro=val_f1_aspect_macro,
        val_acc_sent=val_acc_sent,
        val_f1_sent_macro=val_f1_sent_macro,
    )


def _arch_title(arch: str) -> str:
    return {
        "paper": "Model A (paper): CNN(3) + BiLSTM + pooling + 2 heads",
        "transformer_mtl": "Model B (Transformer-MTL): RoBERTa/BERT encoder + 2 heads",
    }.get(arch, arch)


def _write_report(path: str, *, csv: str, max_len: int, batch_size: int, lr: float, dropout: float, results: List[RunResult]) -> None:
    lines: List[str] = []
    lines.append("## Report so sánh 2 model (multitask aspect + sentiment)\n")
    lines.append(f"- **Dataset**: `{csv}`")
    lines.append(f"- **max_len**: {max_len}")
    lines.append(f"- **batch_size**: {batch_size}")
    lines.append(f"- **optimizer**: RMSprop, lr={lr}")
    lines.append(f"- **dropout**: {dropout}")
    lines.append(f"- **time**: {datetime.now().isoformat(timespec='seconds')}\n")

    for r in results:
        lines.append(f"### {_arch_title(r.arch)}")
        lines.append(f"- **epochs**: {r.epochs}")
        lines.append(f"- **val aspect**: acc={r.val_acc_aspect:.4f}, macro-F1={r.val_f1_aspect_macro:.4f}")
        lines.append(f"- **val sentiment**: acc={r.val_acc_sent:.4f}, macro-F1={r.val_f1_sent_macro:.4f}\n")

    # Simple winner hints (no tables)
    best_aspect = max(results, key=lambda x: x.val_f1_aspect_macro)
    best_sent = max(results, key=lambda x: x.val_f1_sent_macro)
    lines.append("### Kết luận nhanh")
    lines.append(f"- **Aspect macro-F1 tốt nhất**: `{best_aspect.arch}` ({best_aspect.val_f1_aspect_macro:.4f})")
    lines.append(f"- **Sentiment macro-F1 tốt nhất**: `{best_sent.arch}` ({best_sent.val_f1_sent_macro:.4f})")
    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", default="reviewText")
    p.add_argument("--aspect-col", default="aspect")
    p.add_argument("--sentiment-col", default="sentiment")
    p.add_argument("--max-samples", type=int, default=0, help="Giới hạn số sample (0 = dùng toàn bộ)")
    p.add_argument("--max-len", type=int, default=62)
    p.add_argument("--min-word-freq", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--transformer-name", default="roberta-base")
    p.add_argument("--out-md", default="report-compare.md")
    args = p.parse_args(argv)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.aspect_col, args.sentiment_col])
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)
    texts = df[args.text_col].astype(str).tolist()
    aspects = df[args.aspect_col].astype(str).map(lambda x: str(x).strip()).tolist()
    sentiments = df[args.sentiment_col].astype(str).map(lambda x: str(x).strip()).tolist()

    # Split theo dòng để dùng chung cho cả 2 mô hình.
    idx = np.arange(len(texts))
    idx_train, idx_val = train_test_split(idx, test_size=0.15, random_state=42, shuffle=True)
    texts_train = [texts[i] for i in idx_train]
    texts_val = [texts[i] for i in idx_val]
    aspects_train = [aspects[i] for i in idx_train]
    aspects_val = [aspects[i] for i in idx_val]
    sentiments_train = [sentiments[i] for i in idx_train]
    sentiments_val = [sentiments[i] for i in idx_val]

    # Vocab/labels được fit trên TRAIN để tránh leak.
    aspect_vocab = build_label_vocab(aspects_train)
    sentiment_vocab = build_label_vocab(sentiments_train)

    # ===== Data cho model paper (CNN+BiLSTM) =====
    examples_train = build_examples(texts_train, aspects_train, sentiments_train)
    examples_val = build_examples(texts_val, aspects_val, sentiments_val)
    if len(examples_train) < 10 or len(examples_val) < 2:
        raise SystemExit(
            f"Quá ít sample hợp lệ sau preprocessing. train={len(examples_train)} val={len(examples_val)}"
        )

    word_vocab = build_vocab([ex.tokens for ex in examples_train], min_freq=args.min_word_freq)
    pos_vocab = build_vocab([ex.pos for ex in examples_train], min_freq=1)

    ds_train = Absadataset(
        examples_train,
        word_vocab=word_vocab,
        pos_vocab=pos_vocab,
        aspect_vocab=aspect_vocab,
        sentiment_vocab=sentiment_vocab,
        max_len=args.max_len,
    )
    ds_val = Absadataset(
        examples_val,
        word_vocab=word_vocab,
        pos_vocab=pos_vocab,
        aspect_vocab=aspect_vocab,
        sentiment_vocab=sentiment_vocab,
        max_len=args.max_len,
    )

    dl_train_paper = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val_paper = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = _device(args.device)

    # ===== Data cho Transformer-MTL =====
    if AutoTokenizer is None:
        raise SystemExit("Thiếu dependency `transformers` (AutoTokenizer). Hãy chạy `uv sync` lại.")
    tok = AutoTokenizer.from_pretrained(args.transformer_name)

    def tf_collate(batch_items: List[Tuple[str, int, int]]) -> Dict[str, torch.Tensor]:
        bt = [t for t, _, _ in batch_items]
        ya = torch.tensor([a for _, a, _ in batch_items], dtype=torch.long)
        ys = torch.tensor([s for _, _, s in batch_items], dtype=torch.long)
        enc = tok(bt, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        enc["y_aspect"] = ya
        enc["y_sentiment"] = ys
        return enc

    class TfDs(torch.utils.data.Dataset):  # type: ignore[misc]
        def __init__(self, t: List[str], a: List[str], s: List[str]) -> None:
            self.t = t
            self.a = a
            self.s = s

        def __len__(self) -> int:
            return len(self.t)

        def __getitem__(self, i: int) -> Tuple[str, int, int]:
            return self.t[i], aspect_vocab[str(self.a[i])], sentiment_vocab[str(self.s[i])]

    tf_train = TfDs(texts_train, aspects_train, sentiments_train)
    tf_val = TfDs(texts_val, aspects_val, sentiments_val)
    dl_train_tf = DataLoader(tf_train, batch_size=args.batch_size, shuffle=True, collate_fn=tf_collate)
    dl_val_tf = DataLoader(tf_val, batch_size=args.batch_size, shuffle=False, collate_fn=tf_collate)

    results: List[RunResult] = []
    # Model A: paper flow
    results.append(
        _train_one(
            arch="paper",
            dl_train=dl_train_paper,
            dl_val=dl_val_paper,
            vocab_size=len(word_vocab),
            pos_vocab_size=len(pos_vocab),
            aspect_classes=len(aspect_vocab),
            sentiment_classes=len(sentiment_vocab),
            max_len=args.max_len,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            transformer_name=args.transformer_name,
        )
    )
    # Model B: transformer multitask (English)
    results.append(
        _train_one(
            arch="transformer_mtl",
            dl_train=dl_train_tf,
            dl_val=dl_val_tf,
            vocab_size=len(word_vocab),
            pos_vocab_size=len(pos_vocab),
            aspect_classes=len(aspect_vocab),
            sentiment_classes=len(sentiment_vocab),
            max_len=args.max_len,
            dropout=args.dropout,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            transformer_name=args.transformer_name,
        )
    )

    _write_report(
        args.out_md,
        csv=args.csv,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        results=results,
    )
    print(f"Đã ghi report: {args.out_md}")


if __name__ == "__main__":
    main()

