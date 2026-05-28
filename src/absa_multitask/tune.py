from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from mealpy import FloatVar, PSO
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from tqdm import tqdm

from .compare import build_label_vocab
from .data import Absadataset, build_vocab, collate_fn
from .model import MultitaskAbsaModel
from .textproc import build_examples
from .transformer_model import TransformerMtlModel

try:
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover
    AutoTokenizer = None


@dataclass(frozen=True)
class TaskMetrics:
    acc: float
    precision: float
    recall: float
    f1: float
    auc: float


@dataclass(frozen=True)
class EvalResult:
    aspect: TaskMetrics
    sentiment: TaskMetrics


@dataclass(frozen=True)
class RunArtifacts:
    eval_result: EvalResult
    history: List[Dict[str, float]]
    best_epoch: int
    total_epochs: int


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_multiclass_auc(y_true: Sequence[int], probs: np.ndarray) -> float:
    classes = sorted(set(int(x) for x in y_true))
    if len(classes) < 2:
        return float("nan")
    if probs.ndim != 2 or probs.shape[1] < 2:
        return float("nan")
    try:
        y_bin = label_binarize(list(y_true), classes=classes)
        return float(roc_auc_score(y_bin, probs[:, classes], average="macro", multi_class="ovr"))
    except Exception:
        return float("nan")


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int], probs: np.ndarray) -> TaskMetrics:
    return TaskMetrics(
        acc=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        recall=float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        auc=safe_multiclass_auc(y_true, probs),
    )


def evaluate_model(model, dl_val, arch: str, device: torch.device) -> EvalResult:
    model.eval()
    y_a_true: List[int] = []
    y_a_pred: List[int] = []
    y_s_true: List[int] = []
    y_s_pred: List[int] = []
    y_a_probs: List[np.ndarray] = []
    y_s_probs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dl_val:
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

            pa = torch.softmax(out.logits_aspect, dim=-1).cpu().numpy()
            ps = torch.softmax(out.logits_sentiment, dim=-1).cpu().numpy()
            y_a_probs.append(pa)
            y_s_probs.append(ps)
            y_a_true.extend(ya.cpu().numpy().tolist())
            y_s_true.extend(ys.cpu().numpy().tolist())
            y_a_pred.extend(np.argmax(pa, axis=1).tolist())
            y_s_pred.extend(np.argmax(ps, axis=1).tolist())

    arr_a = np.concatenate(y_a_probs, axis=0) if y_a_probs else np.zeros((0, 1), dtype=float)
    arr_s = np.concatenate(y_s_probs, axis=0) if y_s_probs else np.zeros((0, 1), dtype=float)
    return EvalResult(
        aspect=compute_metrics(y_a_true, y_a_pred, arr_a),
        sentiment=compute_metrics(y_s_true, y_s_pred, arr_s),
    )


def _device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _round_to(options: Sequence[int], x: float) -> int:
    return min(options, key=lambda z: abs(z - x))


def decode_params(arch: str, x: np.ndarray) -> Dict[str, float | int]:
    # x order: lr, dropout, batch_size_raw, max_len_raw, patience_raw
    batch_opts = [16, 32, 64, 128]
    max_len_opts = [32, 48, 62, 80, 96, 128]
    params: Dict[str, float | int] = {
        "lr": float(np.clip(x[0], 5e-5, 5e-3)),
        "dropout": float(np.clip(x[1], 0.05, 0.5)),
        "batch_size": int(_round_to(batch_opts, float(x[2]))),
        "max_len": int(_round_to(max_len_opts, float(x[3]))),
        "patience": int(np.clip(round(float(x[4])), 2, 12)),
    }
    if arch == "transformer_mtl":
        params["lr"] = float(np.clip(x[0], 1e-5, 3e-4))
        params["dropout"] = float(np.clip(x[1], 0.05, 0.3))
    return params


def make_tf_loaders(
    *,
    texts_train: List[str],
    texts_val: List[str],
    aspects_train: List[str],
    aspects_val: List[str],
    sentiments_train: List[str],
    sentiments_val: List[str],
    aspect_vocab: Dict[str, int],
    sentiment_vocab: Dict[str, int],
    batch_size: int,
    max_len: int,
    transformer_name: str,
) -> Tuple[DataLoader, DataLoader]:
    if AutoTokenizer is None:
        raise RuntimeError("Missing transformers AutoTokenizer.")
    tok = AutoTokenizer.from_pretrained(transformer_name)

    def tf_collate(batch_items: List[Tuple[str, int, int]]) -> Dict[str, torch.Tensor]:
        bt = [t for t, _, _ in batch_items]
        ya = torch.tensor([a for _, a, _ in batch_items], dtype=torch.long)
        ys = torch.tensor([s for _, _, s in batch_items], dtype=torch.long)
        enc = tok(bt, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
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
            return (
                self.t[i],
                aspect_vocab.get(str(self.a[i]).strip(), 0),
                sentiment_vocab.get(str(self.s[i]).strip(), 0),
            )

    tf_train = TfDs(texts_train, aspects_train, sentiments_train)
    tf_val = TfDs(texts_val, aspects_val, sentiments_val)
    dl_train = DataLoader(tf_train, batch_size=batch_size, shuffle=True, collate_fn=tf_collate)
    dl_val = DataLoader(tf_val, batch_size=batch_size, shuffle=False, collate_fn=tf_collate)
    return dl_train, dl_val


def make_paper_loaders(
    *,
    texts_train: List[str],
    texts_val: List[str],
    aspects_train: List[str],
    aspects_val: List[str],
    sentiments_train: List[str],
    sentiments_val: List[str],
    aspect_vocab: Dict[str, int],
    sentiment_vocab: Dict[str, int],
    max_len: int,
    min_word_freq: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, int, int]:
    examples_train = build_examples(texts_train, aspects_train, sentiments_train)
    examples_val = build_examples(texts_val, aspects_val, sentiments_val)
    word_vocab = build_vocab([ex.tokens for ex in examples_train], min_freq=min_word_freq)
    pos_vocab = build_vocab([ex.pos for ex in examples_train], min_freq=1)

    ds_train = Absadataset(
        examples_train,
        word_vocab=word_vocab,
        pos_vocab=pos_vocab,
        aspect_vocab=aspect_vocab,
        sentiment_vocab=sentiment_vocab,
        max_len=max_len,
    )
    ds_val = Absadataset(
        examples_val,
        word_vocab=word_vocab,
        pos_vocab=pos_vocab,
        aspect_vocab=aspect_vocab,
        sentiment_vocab=sentiment_vocab,
        max_len=max_len,
    )
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return dl_train, dl_val, len(word_vocab), len(pos_vocab)


def train_once(
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
    patience: int,
) -> RunArtifacts:
    if arch == "paper":
        model = MultitaskAbsaModel(
            vocab_size=vocab_size,
            pos_vocab_size=pos_vocab_size,
            aspect_classes=aspect_classes,
            sentiment_classes=sentiment_classes,
            max_len=max_len,
            dropout=dropout,
        ).to(device)
    else:
        model = TransformerMtlModel(
            model_name=transformer_name,
            aspect_classes=aspect_classes,
            sentiment_classes=sentiment_classes,
            dropout=dropout,
        ).to(device)

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
        for batch in tqdm(dl_train, desc=f"{arch} train epoch {epoch}", leave=False):
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
            tr_loss += float(loss.item())
        tr_loss /= max(1, n_batches)

        model.eval()
        val_loss = 0.0
        va_batches = 0
        with torch.no_grad():
            for batch in dl_val:
                va_batches += 1
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
                val_loss += float(
                    loss_fn(out.logits_aspect, y_aspect).item() + loss_fn(out.logits_sentiment, y_sent).item()
                )
        val_loss /= max(1, va_batches)
        eval_metrics = evaluate_model(model, dl_val, arch, device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": tr_loss,
                "val_loss": val_loss,
                "val_f1_aspect": eval_metrics.aspect.f1,
                "val_f1_sentiment": eval_metrics.sentiment.f1,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    final_eval = evaluate_model(model, dl_val, arch, device)
    return RunArtifacts(eval_result=final_eval, history=history, best_epoch=best_epoch, total_epochs=epoch)


def run_pso_for_arch(
    *,
    arch: str,
    texts_train: List[str],
    texts_val: List[str],
    aspects_train: List[str],
    aspects_val: List[str],
    sentiments_train: List[str],
    sentiments_val: List[str],
    aspect_vocab: Dict[str, int],
    sentiment_vocab: Dict[str, int],
    min_word_freq: int,
    pso_epoch: int,
    pso_pop_size: int,
    train_epochs: int,
    transformer_name: str,
    device: torch.device,
    seed: int,
) -> Tuple[Dict[str, float | int], List[float]]:
    def objective(solution: np.ndarray) -> float:
        params = decode_params(arch, solution)
        try:
            if arch == "paper":
                dl_train, dl_val, vocab_size, pos_vocab_size = make_paper_loaders(
                    texts_train=texts_train,
                    texts_val=texts_val,
                    aspects_train=aspects_train,
                    aspects_val=aspects_val,
                    sentiments_train=sentiments_train,
                    sentiments_val=sentiments_val,
                    aspect_vocab=aspect_vocab,
                    sentiment_vocab=sentiment_vocab,
                    max_len=int(params["max_len"]),
                    min_word_freq=min_word_freq,
                    batch_size=int(params["batch_size"]),
                )
            else:
                dl_train, dl_val = make_tf_loaders(
                    texts_train=texts_train,
                    texts_val=texts_val,
                    aspects_train=aspects_train,
                    aspects_val=aspects_val,
                    sentiments_train=sentiments_train,
                    sentiments_val=sentiments_val,
                    aspect_vocab=aspect_vocab,
                    sentiment_vocab=sentiment_vocab,
                    batch_size=int(params["batch_size"]),
                    max_len=int(params["max_len"]),
                    transformer_name=transformer_name,
                )
                vocab_size, pos_vocab_size = 0, 0

            run = train_once(
                arch=arch,
                dl_train=dl_train,
                dl_val=dl_val,
                vocab_size=vocab_size,
                pos_vocab_size=pos_vocab_size,
                aspect_classes=len(aspect_vocab),
                sentiment_classes=len(sentiment_vocab),
                max_len=int(params["max_len"]),
                dropout=float(params["dropout"]),
                epochs=train_epochs,
                lr=float(params["lr"]),
                device=device,
                transformer_name=transformer_name,
                patience=int(params["patience"]),
            )
            obj = (run.eval_result.aspect.f1 + run.eval_result.sentiment.f1) / 2.0
            return -obj
        except torch.cuda.OutOfMemoryError:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return 1e6
        except Exception:
            return 1e6

    set_seed(seed)
    bounds = FloatVar(
        lb=(5e-5, 0.05, 16.0, 32.0, 2.0),
        ub=(5e-3, 0.5, 128.0, 128.0, 12.0),
    )
    if arch == "transformer_mtl":
        bounds = FloatVar(
            lb=(1e-5, 0.05, 16.0, 32.0, 2.0),
            ub=(3e-4, 0.3, 128.0, 128.0, 12.0),
        )
    problem = {
        "bounds": bounds,
        "obj_func": objective,
        "minmax": "min",
        "name": f"PSO_tuning_{arch}",
        "log_to": None,
        "save_population": False,
    }
    model = PSO.OriginalPSO(epoch=pso_epoch, pop_size=pso_pop_size)
    g_best = model.solve(problem)
    best_vec = np.array(g_best.solution, dtype=float)
    best_params = decode_params(arch, best_vec)
    convergence = [float(x) for x in model.history.list_global_best_fit]
    return best_params, convergence


def save_convergence(path_csv: Path, path_png: Path, values: List[float]) -> None:
    df = pd.DataFrame({"step": np.arange(1, len(values) + 1), "best_objective": values})
    df.to_csv(path_csv, index=False)
    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["best_objective"], marker="o", linewidth=1.5)
    plt.xlabel("PSO Step")
    plt.ylabel("Best Objective (min)")
    plt.title("PSO Convergence")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_png, dpi=180)
    plt.close()


def save_training_curve(path_png: Path, history: List[Dict[str, float]], arch: str) -> None:
    if not history:
        return
    df = pd.DataFrame(history)
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df["val_loss"], label="val_loss")
    plt.plot(df["epoch"], df["val_f1_aspect"], label="val_f1_aspect")
    plt.plot(df["epoch"], df["val_f1_sentiment"], label="val_f1_sentiment")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title(f"Training Diagram - {arch}")
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_png, dpi=180)
    plt.close()


def metrics_rows(model_name: str, result: EvalResult) -> List[Dict[str, str | float]]:
    rows: List[Dict[str, str | float]] = []
    for task_name, task in (("aspect", result.aspect), ("sentiment", result.sentiment)):
        rows.append(
            {
                "model": model_name,
                "task": task_name,
                "acc": task.acc,
                "precision": task.precision,
                "recall": task.recall,
                "f1": task.f1,
                "auc": task.auc,
            }
        )
    return rows


def write_report(path: Path, run_dir: Path, rows: List[Dict[str, str | float]], notes: str) -> None:
    df = pd.DataFrame(rows)
    md_table = dataframe_to_markdown(df)
    lines = [
        "# ABSA PSO Tuning Report",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Objective: average macro-F1 (aspect, sentiment) maximized via PSO",
        "",
        "## Final Metrics",
        "",
        md_table,
        "",
        "## Notes",
        notes,
        "",
        "## AUC Handling",
        "AUC is computed as macro OvR multiclass ROC-AUC; if unavailable in an edge case, it is saved as NaN.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows: List[str] = [header, sep]
    for _, row in df.iterrows():
        vals: List[str] = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("NaN")
                else:
                    vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        rows.append("| " + " | ".join(vals) + " |")
    return "\n".join(rows)


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    p.add_argument("--text-col", default="reviewText")
    p.add_argument("--aspect-col", default="aspect")
    p.add_argument("--sentiment-col", default="sentiment")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--min-word-freq", type=int, default=1)
    p.add_argument("--train-epochs", type=int, default=10)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--transformer-name", default="roberta-base")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pso-epoch", type=int, default=15)
    p.add_argument("--pso-pop-size", type=int, default=12)
    p.add_argument("--out-dir", default="outputs/tuning")
    args = p.parse_args(argv)
    if args.pso_pop_size < 5:
        raise SystemExit("--pso-pop-size must be >= 5 for mealpy PSO.")

    set_seed(args.seed)
    out_root = Path(args.out_dir)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    df = df.dropna(subset=[args.text_col, args.aspect_col, args.sentiment_col]).reset_index(drop=True)
    if args.max_samples and args.max_samples > 0:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed).reset_index(drop=True)

    texts = df[args.text_col].astype(str).tolist()
    aspects = df[args.aspect_col].astype(str).map(lambda x: str(x).strip()).tolist()
    sentiments = df[args.sentiment_col].astype(str).map(lambda x: str(x).strip()).tolist()
    idx = np.arange(len(texts))
    idx_train, idx_val = train_test_split(idx, test_size=0.15, random_state=args.seed, shuffle=True)
    texts_train = [texts[i] for i in idx_train]
    texts_val = [texts[i] for i in idx_val]
    aspects_train = [aspects[i] for i in idx_train]
    aspects_val = [aspects[i] for i in idx_val]
    sentiments_train = [sentiments[i] for i in idx_train]
    sentiments_val = [sentiments[i] for i in idx_val]

    aspect_vocab = build_label_vocab(aspects_train)
    sentiment_vocab = build_label_vocab(sentiments_train)
    device = _device(args.device)

    all_rows: List[Dict[str, str | float]] = []
    model_cfgs = [("paper", "model_a"), ("transformer_mtl", "model_b")]
    notes: List[str] = []

    for arch, alias in model_cfgs:
        best_params, convergence = run_pso_for_arch(
            arch=arch,
            texts_train=texts_train,
            texts_val=texts_val,
            aspects_train=aspects_train,
            aspects_val=aspects_val,
            sentiments_train=sentiments_train,
            sentiments_val=sentiments_val,
            aspect_vocab=aspect_vocab,
            sentiment_vocab=sentiment_vocab,
            min_word_freq=args.min_word_freq,
            pso_epoch=args.pso_epoch,
            pso_pop_size=args.pso_pop_size,
            train_epochs=args.train_epochs,
            transformer_name=args.transformer_name,
            device=device,
            seed=args.seed,
        )

        (run_dir / f"best_params_{alias}.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")
        save_convergence(
            run_dir / f"pso_convergence_{alias}.csv",
            run_dir / f"pso_convergence_{alias}.png",
            convergence,
        )

        if arch == "paper":
            dl_train, dl_val, vocab_size, pos_vocab_size = make_paper_loaders(
                texts_train=texts_train,
                texts_val=texts_val,
                aspects_train=aspects_train,
                aspects_val=aspects_val,
                sentiments_train=sentiments_train,
                sentiments_val=sentiments_val,
                aspect_vocab=aspect_vocab,
                sentiment_vocab=sentiment_vocab,
                max_len=int(best_params["max_len"]),
                min_word_freq=args.min_word_freq,
                batch_size=int(best_params["batch_size"]),
            )
        else:
            dl_train, dl_val = make_tf_loaders(
                texts_train=texts_train,
                texts_val=texts_val,
                aspects_train=aspects_train,
                aspects_val=aspects_val,
                sentiments_train=sentiments_train,
                sentiments_val=sentiments_val,
                aspect_vocab=aspect_vocab,
                sentiment_vocab=sentiment_vocab,
                batch_size=int(best_params["batch_size"]),
                max_len=int(best_params["max_len"]),
                transformer_name=args.transformer_name,
            )
            vocab_size, pos_vocab_size = 0, 0

        run = train_once(
            arch=arch,
            dl_train=dl_train,
            dl_val=dl_val,
            vocab_size=vocab_size,
            pos_vocab_size=pos_vocab_size,
            aspect_classes=len(aspect_vocab),
            sentiment_classes=len(sentiment_vocab),
            max_len=int(best_params["max_len"]),
            dropout=float(best_params["dropout"]),
            epochs=args.train_epochs,
            lr=float(best_params["lr"]),
            device=device,
            transformer_name=args.transformer_name,
            patience=int(best_params["patience"]),
        )
        save_training_curve(run_dir / f"training_curves_{alias}.png", run.history, arch)
        rows = metrics_rows(alias, run.eval_result)
        all_rows.extend(rows)
        notes.append(
            f"- `{alias}` ({arch}): best_epoch={run.best_epoch}, total_epochs={run.total_epochs}, "
            f"best_params={best_params}"
        )

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(run_dir / "final_metrics.csv", index=False)
    (run_dir / "final_metrics.md").write_text(dataframe_to_markdown(final_df), encoding="utf-8")
    write_report(run_dir / "report.md", run_dir, all_rows, "\n".join(notes))
    print(f"Tuning completed. Artifacts saved at: {run_dir}")


if __name__ == "__main__":
    main()
