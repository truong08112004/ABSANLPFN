from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
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
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = use all)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default="outputs/train", help="Directory to save report and artifacts")
    args = p.parse_args(argv)

    # Setup output directory
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.aspect_col, args.sentiment_col])
    total_rows = len(df)
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=42).reset_index(drop=True)

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

    # Track history for report
    history: List[Dict[str, float]] = []

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

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc_aspect": tr_acc_a,
            "val_acc_aspect": va_acc_a,
            "train_acc_sent": tr_acc_s,
            "val_acc_sent": va_acc_s,
        })

        print(
            f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"train_acc_aspect={tr_acc_a:.4f} val_acc_aspect={va_acc_a:.4f} "
            f"train_acc_sent={tr_acc_s:.4f} val_acc_sent={va_acc_s:.4f}"
        )

    # --- Final evaluation with detailed metrics ---
    model.eval()
    all_y_aspect: List[int] = []
    all_pred_aspect: List[int] = []
    all_y_sent: List[int] = []
    all_pred_sent: List[int] = []
    with torch.no_grad():
        for batch in dl_val:
            word_ids = batch.word_ids.to(device)
            pos_ids = batch.pos_ids.to(device)
            y_aspect = batch.y_aspect.to(device)
            y_sent = batch.y_sentiment.to(device)
            out = model(word_ids, pos_ids)
            all_y_aspect.extend(y_aspect.cpu().numpy().tolist())
            all_pred_aspect.extend(out.logits_aspect.argmax(dim=-1).cpu().numpy().tolist())
            all_y_sent.extend(y_sent.cpu().numpy().tolist())
            all_pred_sent.extend(out.logits_sentiment.argmax(dim=-1).cpu().numpy().tolist())

    final_metrics = {
        "aspect_accuracy": float(accuracy_score(all_y_aspect, all_pred_aspect)),
        "aspect_precision": float(precision_score(all_y_aspect, all_pred_aspect, average="macro", zero_division=0)),
        "aspect_recall": float(recall_score(all_y_aspect, all_pred_aspect, average="macro", zero_division=0)),
        "aspect_f1": float(f1_score(all_y_aspect, all_pred_aspect, average="macro", zero_division=0)),
        "sentiment_accuracy": float(accuracy_score(all_y_sent, all_pred_sent)),
        "sentiment_precision": float(precision_score(all_y_sent, all_pred_sent, average="macro", zero_division=0)),
        "sentiment_recall": float(recall_score(all_y_sent, all_pred_sent, average="macro", zero_division=0)),
        "sentiment_f1": float(f1_score(all_y_sent, all_pred_sent, average="macro", zero_division=0)),
    }

    # --- Save artifacts ---
    # 1. Training history CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(out_dir / "training_history.csv", index=False)

    # 2. Final metrics JSON
    (out_dir / "final_metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")

    # 3. Training curves plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    epochs_range = df_hist["epoch"]

    axes[0].plot(epochs_range, df_hist["train_loss"], label="train_loss")
    axes[0].plot(epochs_range, df_hist["val_loss"], label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs_range, df_hist["train_acc_aspect"], label="train_acc_aspect")
    axes[1].plot(epochs_range, df_hist["val_acc_aspect"], label="val_acc_aspect")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Aspect Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(epochs_range, df_hist["train_acc_sent"], label="train_acc_sent")
    axes[2].plot(epochs_range, df_hist["val_acc_sent"], label="val_acc_sent")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Accuracy")
    axes[2].set_title("Sentiment Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    # 4. Run config
    run_config = {
        "csv": args.csv,
        "arch": args.arch,
        "max_samples": args.max_samples,
        "total_csv_rows": total_rows,
        "used_samples": len(examples),
        "train_samples": len(ex_train),
        "val_samples": len(ex_val),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "device": str(device),
        "vocab_size": len(vp.word),
        "pos_vocab_size": len(vp.pos),
        "aspect_classes": len(vp.aspect),
        "sentiment_classes": len(vp.sentiment),
        "run_id": run_id,
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # 5. Markdown report
    report_lines = [
        "# ABSA Training Report",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Architecture**: `{args.arch}`",
        f"- **Device**: `{device}`",
        f"- **CSV**: `{args.csv}` ({total_rows} total rows)",
        f"- **Samples used**: {len(examples)} (max_samples={args.max_samples})",
        f"- **Train/Val split**: {len(ex_train)} / {len(ex_val)}",
        f"- **Epochs**: {args.epochs}",
        f"- **Batch size**: {args.batch_size}",
        f"- **Learning rate**: {args.lr}",
        f"- **Dropout**: {args.dropout}",
        f"- **Max len**: {args.max_len}",
        "",
        "## Final Metrics (Validation Set)",
        "",
        "| Task | Accuracy | Precision | Recall | F1 |",
        "|------|----------|-----------|--------|-----|",
        f"| Aspect | {final_metrics['aspect_accuracy']:.4f} | {final_metrics['aspect_precision']:.4f} | {final_metrics['aspect_recall']:.4f} | {final_metrics['aspect_f1']:.4f} |",
        f"| Sentiment | {final_metrics['sentiment_accuracy']:.4f} | {final_metrics['sentiment_precision']:.4f} | {final_metrics['sentiment_recall']:.4f} | {final_metrics['sentiment_f1']:.4f} |",
        "",
        "## Training Curves",
        "",
        "![Training Curves](training_curves.png)",
        "",
        "## Last 5 Epochs",
        "",
        "| Epoch | Train Loss | Val Loss | Train Acc Aspect | Val Acc Aspect | Train Acc Sent | Val Acc Sent |",
        "|-------|-----------|---------|-----------------|---------------|---------------|-------------|",
    ]
    for row in history[-5:]:
        report_lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | "
            f"{row['train_acc_aspect']:.4f} | {row['val_acc_aspect']:.4f} | "
            f"{row['train_acc_sent']:.4f} | {row['val_acc_sent']:.4f} |"
        )
    report_lines.append("")

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Training complete! Report saved to: {out_dir}")
    print(f"  - report.md")
    print(f"  - training_history.csv")
    print(f"  - training_curves.png")
    print(f"  - final_metrics.json")
    print(f"  - run_config.json")
    print(f"{'='*60}")
    print(f"\nFinal metrics:")
    print(f"  Aspect    - Acc: {final_metrics['aspect_accuracy']:.4f}  F1: {final_metrics['aspect_f1']:.4f}")
    print(f"  Sentiment - Acc: {final_metrics['sentiment_accuracy']:.4f}  F1: {final_metrics['sentiment_f1']:.4f}")



if __name__ == "__main__":
    main()
