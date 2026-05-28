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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import LABEL_UNK, Absadataset, build_vocab, collate_fn
from .model import CnnOnlyMultitaskModel, MultitaskAbsaModel
from .textproc import build_examples
from .transformer_model import TransformerMtlModel

try:
    from transformers import AutoTokenizer
except Exception:
    AutoTokenizer = None


@dataclass(frozen=True)
class VocabPack:
    word: Dict[str, int]
    pos: Dict[str, int]
    aspect: Dict[str, int]
    sentiment: Dict[str, int]


def build_label_vocab(labels: List[str]) -> Dict[str, int]:
    norm = [str(x).strip() for x in labels]
    uniq = sorted(set(norm))
    out: Dict[str, int] = {LABEL_UNK: 0}
    for lab in uniq:
        if lab == LABEL_UNK:
            continue
        out.setdefault(lab, len(out))
    return out


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == y).float().mean().item())


def evaluate_on_loader(model, dl, is_transformer: bool, device: torch.device, loss_fn):
    """Evaluate model on a dataloader, return metrics dict."""
    model.eval()
    all_y_aspect: List[int] = []
    all_pred_aspect: List[int] = []
    all_y_sent: List[int] = []
    all_pred_sent: List[int] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in dl:
            n_batches += 1
            if is_transformer:
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
            total_loss += float(loss.item())
            all_y_aspect.extend(y_aspect.cpu().numpy().tolist())
            all_pred_aspect.extend(out.logits_aspect.argmax(dim=-1).cpu().numpy().tolist())
            all_y_sent.extend(y_sent.cpu().numpy().tolist())
            all_pred_sent.extend(out.logits_sentiment.argmax(dim=-1).cpu().numpy().tolist())

    avg_loss = total_loss / max(1, n_batches)
    metrics = {
        "loss": avg_loss,
        "aspect_accuracy": float(accuracy_score(all_y_aspect, all_pred_aspect)),
        "aspect_precision": float(precision_score(all_y_aspect, all_pred_aspect, average="macro", zero_division=0)),
        "aspect_recall": float(recall_score(all_y_aspect, all_pred_aspect, average="macro", zero_division=0)),
        "aspect_f1": float(f1_score(all_y_aspect, all_pred_aspect, average="macro", zero_division=0)),
        "sentiment_accuracy": float(accuracy_score(all_y_sent, all_pred_sent)),
        "sentiment_precision": float(precision_score(all_y_sent, all_pred_sent, average="macro", zero_division=0)),
        "sentiment_recall": float(recall_score(all_y_sent, all_pred_sent, average="macro", zero_division=0)),
        "sentiment_f1": float(f1_score(all_y_sent, all_pred_sent, average="macro", zero_division=0)),
    }
    return metrics, all_y_aspect, all_pred_aspect, all_y_sent, all_pred_sent


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", default="reviewText")
    p.add_argument("--aspect-col", default="aspect")
    p.add_argument("--sentiment-col", default="sentiment")
    p.add_argument("--arch", default="paper", choices=["paper", "cnn_only", "bert"])
    p.add_argument("--transformer-name", default="bert-base-uncased", help="HuggingFace model name for bert arch")
    p.add_argument("--max-len", type=int, default=62)
    p.add_argument("--min-word-freq", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-samples", type=int, default=0, help="Limit number of samples (0 = use all)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.25)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--out-dir", default="outputs/train", help="Directory to save report and artifacts")
    p.add_argument("--patience", type=int, default=10, help="Early stopping patience (0 = disabled)")
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

    is_transformer = args.arch == "bert"

    # ===== SPLIT: train 70% / val 15% / test 15% =====
    idx = np.arange(len(texts))
    idx_train_val, idx_test = train_test_split(idx, test_size=0.15, random_state=42, shuffle=True)
    idx_train, idx_val = train_test_split(idx_train_val, test_size=0.176, random_state=42, shuffle=True)
    # 0.176 of 85% ≈ 15% of total

    texts_train = [texts[i] for i in idx_train]
    texts_val = [texts[i] for i in idx_val]
    texts_test = [texts[i] for i in idx_test]
    aspects_train = [aspects[i] for i in idx_train]
    aspects_val = [aspects[i] for i in idx_val]
    aspects_test = [aspects[i] for i in idx_test]
    sentiments_train = [sentiments[i] for i in idx_train]
    sentiments_val = [sentiments[i] for i in idx_val]
    sentiments_test = [sentiments[i] for i in idx_test]

    n_train = len(texts_train)
    n_val = len(texts_val)
    n_test = len(texts_test)
    n_examples = len(texts)

    if is_transformer:
        if AutoTokenizer is None:
            raise SystemExit("transformers library not installed. Run: pip install transformers")
        tokenizer = AutoTokenizer.from_pretrained(args.transformer_name)

        aspect_vocab = build_label_vocab(aspects)
        sentiment_vocab = build_label_vocab(sentiments)

        class TransformerDataset(torch.utils.data.Dataset):
            def __init__(self, t: List[str], a: List[str], s: List[str]) -> None:
                self.t = t
                self.a = a
                self.s = s

            def __len__(self) -> int:
                return len(self.t)

            def __getitem__(self, i: int) -> Tuple[str, int, int]:
                return (
                    self.t[i],
                    aspect_vocab.get(str(self.a[i]).strip(), 0),
                    sentiment_vocab.get(str(self.s[i]).strip(), 0),
                )

        def tf_collate(batch_items: List[Tuple[str, int, int]]) -> Dict[str, torch.Tensor]:
            bt = [t for t, _, _ in batch_items]
            ya = torch.tensor([a for _, a, _ in batch_items], dtype=torch.long)
            ys = torch.tensor([s for _, _, s in batch_items], dtype=torch.long)
            enc = tokenizer(bt, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
            enc["y_aspect"] = ya
            enc["y_sentiment"] = ys
            return enc

        ds_train = TransformerDataset(texts_train, aspects_train, sentiments_train)
        ds_val = TransformerDataset(texts_val, aspects_val, sentiments_val)
        ds_test = TransformerDataset(texts_test, aspects_test, sentiments_test)
        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=tf_collate)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=tf_collate)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=tf_collate)

    else:
        examples_train = build_examples(texts_train, aspects_train, sentiments_train)
        examples_val = build_examples(texts_val, aspects_val, sentiments_val)
        examples_test = build_examples(texts_test, aspects_test, sentiments_test)

        if len(examples_train) < 10:
            raise SystemExit(f"Quá ít sample hợp lệ sau preprocessing: {len(examples_train)}")

        word_vocab = build_vocab([ex.tokens for ex in examples_train], min_freq=args.min_word_freq)
        pos_vocab = build_vocab([ex.pos for ex in examples_train], min_freq=1)
        aspect_vocab = build_label_vocab([ex.aspect for ex in examples_train])
        sentiment_vocab = build_label_vocab([ex.sentiment for ex in examples_train])

        vp = VocabPack(word=word_vocab, pos=pos_vocab, aspect=aspect_vocab, sentiment=sentiment_vocab)

        ds_train = Absadataset(examples_train, word_vocab=vp.word, pos_vocab=vp.pos,
                               aspect_vocab=vp.aspect, sentiment_vocab=vp.sentiment, max_len=args.max_len)
        ds_val = Absadataset(examples_val, word_vocab=vp.word, pos_vocab=vp.pos,
                             aspect_vocab=vp.aspect, sentiment_vocab=vp.sentiment, max_len=args.max_len)
        ds_test = Absadataset(examples_test, word_vocab=vp.word, pos_vocab=vp.pos,
                              aspect_vocab=vp.aspect, sentiment_vocab=vp.sentiment, max_len=args.max_len)

        dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
        dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        dl_test = DataLoader(ds_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cuda" if args.device == "cuda" else "cpu")
    )

    if is_transformer:
        model = TransformerMtlModel(
            model_name=args.transformer_name,
            aspect_classes=len(aspect_vocab),
            sentiment_classes=len(sentiment_vocab),
            dropout=args.dropout,
        ).to(device)
        effective_lr = args.lr if args.lr < 1e-3 else 2e-5
    elif args.arch == "paper":
        model = MultitaskAbsaModel(
            vocab_size=len(vp.word),
            pos_vocab_size=len(vp.pos),
            aspect_classes=len(vp.aspect),
            sentiment_classes=len(vp.sentiment),
            max_len=args.max_len,
            dropout=args.dropout,
        ).to(device)
        effective_lr = args.lr
    else:
        model = CnnOnlyMultitaskModel(
            vocab_size=len(vp.word),
            pos_vocab_size=len(vp.pos),
            aspect_classes=len(vp.aspect),
            sentiment_classes=len(vp.sentiment),
            max_len=args.max_len,
            dropout=args.dropout,
        ).to(device)
        effective_lr = args.lr

    if is_transformer:
        opt = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=0.01)
    else:
        opt = torch.optim.RMSprop(model.parameters(), lr=effective_lr)
    loss_fn = nn.CrossEntropyLoss()

    # Track history for report
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        tr_loss = 0.0
        tr_acc_a = 0.0
        tr_acc_s = 0.0
        n_batches = 0
        for batch in tqdm(dl_train, desc=f"train epoch {epoch}"):
            n_batches += 1
            if is_transformer:
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

        # Validation
        val_metrics_epoch, _, _, _, _ = evaluate_on_loader(model, dl_val, is_transformer, device, loss_fn)

        # Test (evaluate mỗi epoch để vẽ đường test)
        test_metrics_epoch, _, _, _, _ = evaluate_on_loader(model, dl_test, is_transformer, device, loss_fn)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_metrics_epoch["loss"],
            "test_loss": test_metrics_epoch["loss"],
            "train_acc_aspect": tr_acc_a,
            "val_acc_aspect": val_metrics_epoch["aspect_accuracy"],
            "test_acc_aspect": test_metrics_epoch["aspect_accuracy"],
            "train_acc_sent": tr_acc_s,
            "val_acc_sent": val_metrics_epoch["sentiment_accuracy"],
            "test_acc_sent": test_metrics_epoch["sentiment_accuracy"],
            "val_f1_aspect": val_metrics_epoch["aspect_f1"],
            "test_f1_aspect": test_metrics_epoch["aspect_f1"],
            "val_f1_sent": val_metrics_epoch["sentiment_f1"],
            "test_f1_sent": test_metrics_epoch["sentiment_f1"],
        })

        print(
            f"epoch={epoch} train_loss={tr_loss:.4f} val_loss={val_metrics_epoch['loss']:.4f} test_loss={test_metrics_epoch['loss']:.4f} "
            f"train_acc_aspect={tr_acc_a:.4f} val_acc_aspect={val_metrics_epoch['aspect_accuracy']:.4f} test_acc_aspect={test_metrics_epoch['aspect_accuracy']:.4f} "
            f"train_acc_sent={tr_acc_s:.4f} val_acc_sent={val_metrics_epoch['sentiment_accuracy']:.4f} test_acc_sent={test_metrics_epoch['sentiment_accuracy']:.4f}"
        )

        # Early stopping
        if val_metrics_epoch["loss"] < best_val_loss:
            best_val_loss = val_metrics_epoch["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if args.patience > 0 and no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # ===== EVALUATE ON TEST SET =====
    print("\n--- Evaluating on TEST set ---")
    test_metrics, test_y_aspect, test_pred_aspect, test_y_sent, test_pred_sent = evaluate_on_loader(
        model, dl_test, is_transformer, device, loss_fn
    )

    # Also evaluate on val for comparison
    val_metrics, _, _, _, _ = evaluate_on_loader(model, dl_val, is_transformer, device, loss_fn)

    print(f"TEST  - Aspect Acc: {test_metrics['aspect_accuracy']:.4f}  F1: {test_metrics['aspect_f1']:.4f}  "
          f"Sentiment Acc: {test_metrics['sentiment_accuracy']:.4f}  F1: {test_metrics['sentiment_f1']:.4f}")

    # ===== SAVE ARTIFACTS =====
    # 1. Training history CSV
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(out_dir / "training_history.csv", index=False)

    # 2. Final metrics JSON (both val and test)
    final_output = {
        "validation": val_metrics,
        "test": test_metrics,
    }
    (out_dir / "final_metrics.json").write_text(json.dumps(final_output, indent=2), encoding="utf-8")

    # 3. Training curves plot (3 đường: train, val, test)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    epochs_range = df_hist["epoch"]

    # Row 1: Loss, Aspect Acc, Sentiment Acc
    axes[0, 0].plot(epochs_range, df_hist["train_loss"], "b-o", label="train", markersize=4)
    axes[0, 0].plot(epochs_range, df_hist["val_loss"], "r-o", label="val", markersize=4)
    axes[0, 0].plot(epochs_range, df_hist["test_loss"], "g-s", label="test", markersize=4)
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Loss (Train / Val / Test)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs_range, df_hist["train_acc_aspect"], "b-o", label="train", markersize=4)
    axes[0, 1].plot(epochs_range, df_hist["val_acc_aspect"], "r-o", label="val", markersize=4)
    axes[0, 1].plot(epochs_range, df_hist["test_acc_aspect"], "g-s", label="test", markersize=4)
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Aspect Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs_range, df_hist["train_acc_sent"], "b-o", label="train", markersize=4)
    axes[0, 2].plot(epochs_range, df_hist["val_acc_sent"], "r-o", label="val", markersize=4)
    axes[0, 2].plot(epochs_range, df_hist["test_acc_sent"], "g-s", label="test", markersize=4)
    axes[0, 2].set_xlabel("Epoch")
    axes[0, 2].set_ylabel("Accuracy")
    axes[0, 2].set_title("Sentiment Accuracy")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # Row 2: F1 scores
    axes[1, 0].plot(epochs_range, df_hist["val_f1_aspect"], "r-o", label="val_f1_aspect", markersize=4)
    axes[1, 0].plot(epochs_range, df_hist["test_f1_aspect"], "g-s", label="test_f1_aspect", markersize=4)
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Aspect F1 (Val / Test)")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs_range, df_hist["val_f1_sent"], "r-o", label="val_f1_sent", markersize=4)
    axes[1, 1].plot(epochs_range, df_hist["test_f1_sent"], "g-s", label="test_f1_sent", markersize=4)
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("F1 Score")
    axes[1, 1].set_title("Sentiment F1 (Val / Test)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Summary text in last subplot
    axes[1, 2].axis("off")
    summary_text = (
        f"Final Test Metrics:\n\n"
        f"Aspect Acc:  {test_metrics['aspect_accuracy']:.4f}\n"
        f"Aspect F1:   {test_metrics['aspect_f1']:.4f}\n"
        f"Sent Acc:    {test_metrics['sentiment_accuracy']:.4f}\n"
        f"Sent F1:     {test_metrics['sentiment_f1']:.4f}\n\n"
        f"Split: {n_train}/{n_val}/{n_test}\n"
        f"(train/val/test)"
    )
    axes[1, 2].text(0.1, 0.5, summary_text, transform=axes[1, 2].transAxes,
                    fontsize=12, verticalalignment="center", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    # 4. Confusion matrix plots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Aspect confusion matrix
    inv_aspect = {v: k for k, v in aspect_vocab.items()}
    aspect_labels = [inv_aspect.get(i, f"cls_{i}") for i in sorted(set(test_y_aspect + test_pred_aspect))]
    cm_aspect = confusion_matrix(test_y_aspect, test_pred_aspect)
    im0 = axes[0].imshow(cm_aspect, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix - Aspect (Test)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    plt.colorbar(im0, ax=axes[0])

    # Sentiment confusion matrix
    inv_sent = {v: k for k, v in sentiment_vocab.items()}
    sent_labels = [inv_sent.get(i, f"cls_{i}") for i in sorted(set(test_y_sent + test_pred_sent))]
    cm_sent = confusion_matrix(test_y_sent, test_pred_sent)
    im1 = axes[1].imshow(cm_sent, interpolation="nearest", cmap=plt.cm.Blues)
    axes[1].set_title("Confusion Matrix - Sentiment (Test)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_xticks(range(len(sent_labels)))
    axes[1].set_yticks(range(len(sent_labels)))
    axes[1].set_xticklabels(sent_labels, rotation=45, ha="right")
    axes[1].set_yticklabels(sent_labels)
    plt.colorbar(im1, ax=axes[1])

    plt.tight_layout()
    plt.savefig(out_dir / "confusion_matrix_test.png", dpi=150)
    plt.close()

    # 5. Run config
    run_config = {
        "csv": args.csv,
        "arch": args.arch,
        "transformer_name": args.transformer_name if is_transformer else None,
        "max_samples": args.max_samples,
        "total_csv_rows": total_rows,
        "used_samples": n_examples,
        "train_samples": n_train,
        "val_samples": n_val,
        "test_samples": n_test,
        "split": "70/15/15 (train/val/test)",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": effective_lr,
        "dropout": args.dropout,
        "max_len": args.max_len,
        "device": str(device),
        "aspect_classes": len(aspect_vocab),
        "sentiment_classes": len(sentiment_vocab),
        "run_id": run_id,
    }
    if not is_transformer:
        run_config["vocab_size"] = len(vp.word)
        run_config["pos_vocab_size"] = len(vp.pos)
    (out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    # 6. Markdown report
    arch_display = args.arch if not is_transformer else f"{args.arch} ({args.transformer_name})"
    report_lines = [
        "# ABSA Training Report",
        "",
        f"- **Run ID**: `{run_id}`",
        f"- **Architecture**: `{arch_display}`",
        f"- **Device**: `{device}`",
        f"- **CSV**: `{args.csv}` ({total_rows} total rows)",
        f"- **Samples used**: {n_examples} (max_samples={args.max_samples})",
        f"- **Split**: Train {n_train} / Val {n_val} / Test {n_test} (70/15/15)",
        f"- **Epochs**: {args.epochs}",
        f"- **Batch size**: {args.batch_size}",
        f"- **Learning rate**: {effective_lr}",
        f"- **Dropout**: {args.dropout}",
        f"- **Max len**: {args.max_len}",
        "",
        "## Results on TEST Set",
        "",
        "| Task | Accuracy | Precision | Recall | F1 |",
        "|------|----------|-----------|--------|-----|",
        f"| Aspect | {test_metrics['aspect_accuracy']:.4f} | {test_metrics['aspect_precision']:.4f} | {test_metrics['aspect_recall']:.4f} | {test_metrics['aspect_f1']:.4f} |",
        f"| Sentiment | {test_metrics['sentiment_accuracy']:.4f} | {test_metrics['sentiment_precision']:.4f} | {test_metrics['sentiment_recall']:.4f} | {test_metrics['sentiment_f1']:.4f} |",
        "",
        "## Results on Validation Set",
        "",
        "| Task | Accuracy | Precision | Recall | F1 |",
        "|------|----------|-----------|--------|-----|",
        f"| Aspect | {val_metrics['aspect_accuracy']:.4f} | {val_metrics['aspect_precision']:.4f} | {val_metrics['aspect_recall']:.4f} | {val_metrics['aspect_f1']:.4f} |",
        f"| Sentiment | {val_metrics['sentiment_accuracy']:.4f} | {val_metrics['sentiment_precision']:.4f} | {val_metrics['sentiment_recall']:.4f} | {val_metrics['sentiment_f1']:.4f} |",
        "",
        "## Training Curves",
        "",
        "![Training Curves](training_curves.png)",
        "",
        "## Confusion Matrix (Test Set)",
        "",
        "![Confusion Matrix](confusion_matrix_test.png)",
        "",
        "## Training History (Last 5 Epochs)",
        "",
        "| Epoch | Train Loss | Val Loss | Test Loss | Val Acc Asp | Test Acc Asp | Val F1 Asp | Test F1 Asp | Val Acc Sent | Test Acc Sent | Val F1 Sent | Test F1 Sent |",
        "|-------|-----------|---------|----------|------------|-------------|-----------|------------|-------------|--------------|------------|-------------|",
    ]
    for row in history[-5:]:
        report_lines.append(
            f"| {row['epoch']} | {row['train_loss']:.4f} | {row['val_loss']:.4f} | {row['test_loss']:.4f} | "
            f"{row['val_acc_aspect']:.4f} | {row['test_acc_aspect']:.4f} | "
            f"{row['val_f1_aspect']:.4f} | {row['test_f1_aspect']:.4f} | "
            f"{row['val_acc_sent']:.4f} | {row['test_acc_sent']:.4f} | "
            f"{row['val_f1_sent']:.4f} | {row['test_f1_sent']:.4f} |"
        )
    report_lines.append("")

    (out_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"Training complete! Report saved to: {out_dir}")
    print(f"  - report.md")
    print(f"  - training_history.csv")
    print(f"  - training_curves.png")
    print(f"  - confusion_matrix_test.png")
    print(f"  - final_metrics.json")
    print(f"  - run_config.json")
    print(f"{'='*60}")
    print(f"\nTEST SET metrics:")
    print(f"  Aspect    - Acc: {test_metrics['aspect_accuracy']:.4f}  F1: {test_metrics['aspect_f1']:.4f}")
    print(f"  Sentiment - Acc: {test_metrics['sentiment_accuracy']:.4f}  F1: {test_metrics['sentiment_f1']:.4f}")


if __name__ == "__main__":
    main()
