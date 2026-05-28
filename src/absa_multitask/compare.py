from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, classification_report
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
    best_epoch: int
    total_epochs: int
    val_acc_aspect: float
    val_f1_aspect_macro: float
    val_acc_sent: float
    val_f1_sent_macro: float
    train_history: List[Dict[str, float]]


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _eval_model(model, dl_val, arch, device):
    """Evaluate model on validation set, return (val_loss, acc_aspect, acc_sent, f1_aspect, f1_sent)."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    y_a_true, y_a_pred, y_s_true, y_s_pred = [], [], [], []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in dl_val:
            n_batches += 1
            if arch == "transformer_mtl":
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                out = model(input_ids, attention_mask)
                ya = batch["y_aspect"].to(device)
                ys = batch["y_sentiment"].to(device)
            else:
                word_ids = batch.word_ids.to(device)
                pos_ids = batch.pos_ids.to(device)
                out = model(word_ids, pos_ids)
                ya = batch.y_aspect.to(device)
                ys = batch.y_sentiment.to(device)

            loss = loss_fn(out.logits_aspect, ya) + loss_fn(out.logits_sentiment, ys)
            total_loss += loss.item()

            y_a_true.extend(ya.cpu().numpy().tolist())
            y_s_true.extend(ys.cpu().numpy().tolist())
            y_a_pred.extend(out.logits_aspect.argmax(dim=-1).cpu().numpy().tolist())
            y_s_pred.extend(out.logits_sentiment.argmax(dim=-1).cpu().numpy().tolist())

    avg_loss = total_loss / max(1, n_batches)
    acc_a = float(accuracy_score(y_a_true, y_a_pred))
    acc_s = float(accuracy_score(y_s_true, y_s_pred))
    f1_a = float(f1_score(y_a_true, y_a_pred, average="macro", zero_division=0))
    f1_s = float(f1_score(y_s_true, y_s_pred, average="macro", zero_division=0))
    return avg_loss, acc_a, acc_s, f1_a, f1_s


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
    patience: int = 10,
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

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    no_improve = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        n_batches = 0
        for batch in tqdm(dl_train, desc=f"{arch} train epoch {epoch}"):
            n_batches += 1
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
            tr_loss += loss.item()

        tr_loss /= max(1, n_batches)

        # Validation
        val_loss, val_acc_a, val_acc_s, val_f1_a, val_f1_s = _eval_model(model, dl_val, arch, device)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_loss,
            "val_acc_aspect": val_acc_a,
            "val_acc_sent": val_acc_s,
            "val_f1_aspect": val_f1_a,
            "val_f1_sent": val_f1_s,
        })

        print(
            f"  [{arch}] epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_loss:.4f} "
            f"val_acc_a={val_acc_a:.4f} val_acc_s={val_acc_s:.4f} "
            f"val_f1_a={val_f1_a:.4f} val_f1_s={val_f1_s:.4f}"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  [{arch}] Early stopping at epoch {epoch} (best epoch={best_epoch})")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation with best model
    val_loss, val_acc_a, val_acc_s, val_f1_a, val_f1_s = _eval_model(model, dl_val, arch, device)

    return RunResult(
        arch=arch,
        best_epoch=best_epoch,
        total_epochs=epoch,
        val_acc_aspect=val_acc_a,
        val_f1_aspect_macro=val_f1_a,
        val_acc_sent=val_acc_s,
        val_f1_sent_macro=val_f1_s,
        train_history=history,
    )


def _arch_title(arch: str) -> str:
    return {
        "paper": "Model A (paper): CNN(3) + BiLSTM + pooling + 2 heads",
        "transformer_mtl": "Model B (Transformer-MTL): RoBERTa/BERT encoder + 2 heads",
    }.get(arch, arch)


def _write_report(
    path: str,
    *,
    csv: str,
    max_len: int,
    batch_size: int,
    lr: float,
    dropout: float,
    patience: int,
    max_epochs: int,
    results: List[RunResult],
) -> None:
    lines: List[str] = []
    lines.append("# Report so sánh 2 model (multitask aspect + sentiment)\n")
    lines.append(f"- **Dataset**: `{csv}`")
    lines.append(f"- **max_len**: {max_len}")
    lines.append(f"- **batch_size**: {batch_size}")
    lines.append(f"- **optimizer**: RMSprop, lr={lr}")
    lines.append(f"- **dropout**: {dropout}")
    lines.append(f"- **max_epochs**: {max_epochs}")
    lines.append(f"- **early_stopping patience**: {patience}")
    lines.append(f"- **time**: {datetime.now().isoformat(timespec='seconds')}\n")

    lines.append("---\n")

    for r in results:
        lines.append(f"## {_arch_title(r.arch)}\n")
        lines.append(f"- **Best epoch**: {r.best_epoch} / {r.total_epochs} (stopped)")
        lines.append(f"- **Val Aspect Accuracy**: {r.val_acc_aspect:.4f}")
        lines.append(f"- **Val Aspect macro-F1**: {r.val_f1_aspect_macro:.4f}")
        lines.append(f"- **Val Sentiment Accuracy**: {r.val_acc_sent:.4f}")
        lines.append(f"- **Val Sentiment macro-F1**: {r.val_f1_sent_macro:.4f}\n")

        # Training curve summary
        lines.append("### Training History (selected epochs)\n")
        lines.append("| Epoch | Train Loss | Val Loss | Val Acc Aspect | Val Acc Sent | Val F1 Aspect | Val F1 Sent |")
        lines.append("|-------|-----------|----------|---------------|-------------|--------------|------------|")
        # Show first 5, every 10th, and last 5
        h = r.train_history
        indices_to_show = set()
        for i in range(min(5, len(h))):
            indices_to_show.add(i)
        for i in range(0, len(h), 10):
            indices_to_show.add(i)
        for i in range(max(0, len(h) - 5), len(h)):
            indices_to_show.add(i)
        for i in sorted(indices_to_show):
            row = h[i]
            lines.append(
                f"| {row['epoch']:>5} | {row['train_loss']:.4f}    | {row['val_loss']:.4f}   | "
                f"{row['val_acc_aspect']:.4f}         | {row['val_acc_sent']:.4f}       | "
                f"{row['val_f1_aspect']:.4f}        | {row['val_f1_sent']:.4f}      |"
            )
        lines.append("")

    # Comparison table
    lines.append("---\n")
    lines.append("## So sánh tổng hợp\n")
    lines.append("| Metric | Model A (CNN+BiLSTM) | Model B (Transformer) | Winner |")
    lines.append("|--------|---------------------|----------------------|--------|")

    if len(results) == 2:
        ra, rb = results[0], results[1]
        metrics = [
            ("Val Acc Aspect", ra.val_acc_aspect, rb.val_acc_aspect),
            ("Val F1 Aspect (macro)", ra.val_f1_aspect_macro, rb.val_f1_aspect_macro),
            ("Val Acc Sentiment", ra.val_acc_sent, rb.val_acc_sent),
            ("Val F1 Sentiment (macro)", ra.val_f1_sent_macro, rb.val_f1_sent_macro),
            ("Best Epoch", float(ra.best_epoch), float(rb.best_epoch)),
        ]
        for name, va, vb in metrics:
            if name == "Best Epoch":
                winner = "-"
            elif va > vb:
                winner = "Model A"
            elif vb > va:
                winner = "Model B"
            else:
                winner = "Tie"
            lines.append(f"| {name} | {va:.4f} | {vb:.4f} | {winner} |")
    lines.append("")

    # Conclusion
    lines.append("## Kết luận\n")
    if len(results) == 2:
        ra, rb = results[0], results[1]
        best_aspect = "Model A (CNN+BiLSTM)" if ra.val_f1_aspect_macro >= rb.val_f1_aspect_macro else "Model B (Transformer)"
        best_sent = "Model A (CNN+BiLSTM)" if ra.val_f1_sent_macro >= rb.val_f1_sent_macro else "Model B (Transformer)"
        lines.append(f"- **Aspect classification tốt nhất**: {best_aspect}")
        lines.append(f"- **Sentiment classification tốt nhất**: {best_sent}")
        lines.append(f"- Model A dừng ở epoch {ra.best_epoch}/{ra.total_epochs}")
        lines.append(f"- Model B dừng ở epoch {rb.best_epoch}/{rb.total_epochs}")
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
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience")
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
    print(f"Device: {device}")
    print(f"Dataset: {len(texts)} samples -> train={len(idx_train)}, val={len(idx_val)}")
    print(f"Aspect classes: {len(aspect_vocab)}, Sentiment classes: {len(sentiment_vocab)}")
    print(f"Max epochs: {args.epochs}, Early stopping patience: {args.patience}")
    print("=" * 70)

    results: List[RunResult] = []

    # ===== Model A: paper flow (CNN+BiLSTM) =====
    print("\n>>> Training Model A: CNN(3) + BiLSTM (paper flow)")
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
            patience=args.patience,
        )
    )

    # ===== Data cho Transformer-MTL =====
    print("\n>>> Training Model B: Transformer-MTL (RoBERTa)")
    if AutoTokenizer is None:
        raise SystemExit("Thiếu dependency `transformers` (AutoTokenizer). Hãy cài lại.")
    tok = AutoTokenizer.from_pretrained(args.transformer_name)

    def tf_collate(batch_items: List[Tuple[str, int, int]]) -> Dict[str, torch.Tensor]:
        bt = [t for t, _, _ in batch_items]
        ya = torch.tensor([a for _, a, _ in batch_items], dtype=torch.long)
        ys = torch.tensor([s for _, _, s in batch_items], dtype=torch.long)
        enc = tok(bt, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        enc["y_aspect"] = ya
        enc["y_sentiment"] = ys
        return enc

    class TfDs(torch.utils.data.Dataset):
        def __init__(self, t: List[str], a: List[str], s: List[str]) -> None:
            self.t = t
            self.a = a
            self.s = s

        def __len__(self) -> int:
            return len(self.t)

        def __getitem__(self, i: int) -> Tuple[str, int, int]:
            return self.t[i], aspect_vocab.get(str(self.a[i]).strip(), 0), sentiment_vocab.get(str(self.s[i]).strip(), 0)

    tf_train = TfDs(texts_train, aspects_train, sentiments_train)
    tf_val = TfDs(texts_val, aspects_val, sentiments_val)
    dl_train_tf = DataLoader(tf_train, batch_size=args.batch_size, shuffle=True, collate_fn=tf_collate)
    dl_val_tf = DataLoader(tf_val, batch_size=args.batch_size, shuffle=False, collate_fn=tf_collate)

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
            patience=args.patience,
        )
    )

    _write_report(
        args.out_md,
        csv=args.csv,
        max_len=args.max_len,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout=args.dropout,
        patience=args.patience,
        max_epochs=args.epochs,
        results=results,
    )
    print(f"\n{'=' * 70}")
    print(f"Đã ghi report: {args.out_md}")
    print(f"Model A best epoch: {results[0].best_epoch}, Model B best epoch: {results[1].best_epoch}")


if __name__ == "__main__":
    main()
