#!/usr/bin/env python3
"""Generate remaining Excalidraw diagram JSON files for the presentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

OUT = Path(__file__).resolve().parents[1] / "assets" / "diagrams"


def rect(
    eid: str,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    bg: str = "#ffffff",
    stroke: str = "#1e1e1e",
    font_size: int = 16,
) -> dict[str, Any]:
    return {
        "id": eid,
        "type": "rectangle",
        "x": x,
        "y": y,
        "width": w,
        "height": h,
        "backgroundColor": bg,
        "strokeColor": stroke,
        "label": {"text": text},
        "fontSize": font_size,
        "version": 1,
    }


def arrow(eid: str, x: float, y: float, dx: float, dy: float, start: str, end: str, stroke: str = "#1e1e1e") -> dict[str, Any]:
    return {
        "id": eid,
        "type": "arrow",
        "x": x,
        "y": y,
        "strokeColor": stroke,
        "points": [[0, 0], [dx, dy]],
        "start": {"id": start},
        "end": {"id": end},
        "endArrowhead": "arrow",
        "version": 1,
    }


def scene(elements: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "type": "excalidraw",
        "version": 2,
        "source": "absa-presentation-gen",
        "elements": elements,
        "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
    }


def diagram_lstm_layers() -> dict[str, Any]:
    els = [
        rect("in", 40, 80, 130, 70, "Input\nword_ids + pos_ids", bg="#a5d8ff", stroke="#1971c2"),
        rect("emb", 200, 80, 150, 70, "Embedding 100d\n+ POS one-hot"),
        rect("cnn1", 380, 80, 100, 70, "CNN-1\n128 ch", bg="#b2f2bb", stroke="#2f9e44"),
        rect("cnn2", 500, 80, 100, 70, "CNN-2", bg="#b2f2bb", stroke="#2f9e44"),
        rect("cnn3", 620, 80, 100, 70, "CNN-3", bg="#b2f2bb", stroke="#2f9e44"),
        rect("lstm", 740, 70, 120, 90, "BiLSTM\nhidden=128", bg="#eebefa", stroke="#9c36b5"),
        rect("pool", 880, 80, 110, 70, "Avg Pool", bg="#ffd8a8", stroke="#e8590c"),
        rect("dense", 1010, 80, 110, 70, "FC + Dropout"),
        rect("ha", 900, 220, 140, 60, "Head Aspect", bg="#ffc9c9", stroke="#e03131"),
        rect("hs", 1060, 220, 140, 60, "Head Sentiment", bg="#ffc9c9", stroke="#e03131"),
        arrow("a1", 170, 115, 30, 0, "in", "emb"),
        arrow("a2", 350, 115, 30, 0, "emb", "cnn1"),
        arrow("a3", 480, 115, 20, 0, "cnn1", "cnn2"),
        arrow("a4", 600, 115, 20, 0, "cnn2", "cnn3"),
        arrow("a5", 720, 115, 20, 0, "cnn3", "lstm"),
        arrow("a6", 860, 115, 20, 0, "lstm", "pool"),
        arrow("a7", 990, 115, 20, 0, "pool", "dense"),
        arrow("a8", 1065, 150, 0, 70, "dense", "ha"),
        arrow("a9", 1080, 150, 0, 70, "dense", "hs"),
    ]
    return scene(els)


def diagram_bert_layers() -> dict[str, Any]:
    els = [
        rect("txt", 40, 100, 160, 70, "reviewText\n(raw string)", bg="#a5d8ff", stroke="#1971c2"),
        rect("tok", 240, 100, 180, 70, "BERT Tokenizer\nmax_len=128"),
        rect("enc", 460, 90, 200, 90, "bert-base-uncased\n12 Transformer layers", bg="#eebefa", stroke="#9c36b5"),
        rect("cls", 700, 100, 140, 70, "CLS / Pooler\n[768-d]"),
        rect("drop", 880, 100, 120, 70, "Dropout 0.25"),
        rect("ha", 720, 240, 150, 60, "Head Aspect\nsoftmax", bg="#ffc9c9", stroke="#e03131"),
        rect("hs", 900, 240, 150, 60, "Head Sentiment\nsoftmax", bg="#ffc9c9", stroke="#e03131"),
        arrow("b1", 200, 135, 40, 0, "txt", "tok"),
        arrow("b2", 420, 135, 40, 0, "tok", "enc"),
        arrow("b3", 660, 135, 40, 0, "enc", "cls"),
        arrow("b4", 840, 135, 40, 0, "cls", "drop"),
        arrow("b5", 940, 170, -100, 70, "drop", "ha"),
        arrow("b6", 960, 170, 0, 70, "drop", "hs"),
    ]
    return scene(els)


def diagram_preprocessing() -> dict[str, Any]:
    els = [
        rect("raw", 80, 60, 180, 60, "reviewText\n(CSV row)", bg="#a5d8ff", stroke="#1971c2"),
        rect("clean", 320, 60, 200, 60, "Cleaning\nlower, lemmatize"),
        rect("tok", 560, 60, 180, 60, "Tokenize"),
        rect("pos", 320, 180, 200, 60, "POS tagging\n(NLTK)", bg="#b2f2bb", stroke="#2f9e44"),
        rect("vocab", 560, 180, 180, 60, "Build vocab\nPAD / UNK"),
        rect("pad", 800, 120, 200, 80, "Pad / Truncate\nmax_len", bg="#ffd8a8", stroke="#e8590c"),
        rect("out_a", 80, 320, 220, 70, "Model A:\nword_ids, pos_ids", bg="#e9ecef"),
        rect("out_b", 400, 320, 220, 70, "Model B:\ninput_ids, mask", bg="#e9ecef"),
        arrow("p1", 260, 90, 60, 0, "raw", "clean"),
        arrow("p2", 520, 90, 40, 0, "clean", "tok"),
        arrow("p3", 420, 120, 0, 60, "clean", "pos"),
        arrow("p4", 640, 120, 0, 60, "tok", "vocab"),
        arrow("p5", 740, 150, 60, -30, "vocab", "pad"),
        arrow("p6", 900, 200, -500, 120, "pad", "out_a"),
        arrow("p7", 900, 200, -300, 120, "pad", "out_b"),
    ]
    return scene(els)


def diagram_multitask_loss() -> dict[str, Any]:
    els = [
        rect("back", 300, 80, 280, 80, "Shared backbone\n(CNN+BiLSTM hoặc BERT)", bg="#eebefa", stroke="#9c36b5"),
        rect("ha", 120, 240, 200, 70, "Head Aspect\nCrossEntropy", bg="#ffc9c9", stroke="#e03131"),
        rect("hs", 560, 240, 200, 70, "Head Sentiment\nCrossEntropy", bg="#ffc9c9", stroke="#e03131"),
        rect("la", 120, 380, 200, 60, "L_aspect", bg="#ffd8a8", stroke="#e8590c"),
        rect("ls", 560, 380, 200, 60, "L_sentiment", bg="#ffd8a8", stroke="#e8590c"),
        rect("lt", 340, 500, 240, 70, "L = L_aspect + L_sentiment", bg="#b2f2bb", stroke="#2f9e44", font_size=18),
        rect("opt", 340, 620, 240, 60, "Backprop + Optimizer"),
        arrow("m1", 380, 160, -200, 80, "back", "ha"),
        arrow("m2", 500, 160, 200, 80, "back", "hs"),
        arrow("m3", 220, 310, 0, 70, "ha", "la"),
        arrow("m4", 660, 310, 0, 70, "hs", "ls"),
        arrow("m5", 220, 440, 200, 60, "la", "lt"),
        arrow("m6", 660, 440, -200, 60, "ls", "lt"),
        arrow("m7", 460, 570, 0, 50, "lt", "opt"),
    ]
    return scene(els)


DIAGRAMS = {
    "02-model-lstm-layers.excalidraw": diagram_lstm_layers,
    "03-data-preprocessing.excalidraw": diagram_preprocessing,
    "04-model-bert-layers.excalidraw": diagram_bert_layers,
    "05-multitask-loss.excalidraw": diagram_multitask_loss,
}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, builder in DIAGRAMS.items():
        path = OUT / name
        path.write_text(json.dumps(builder(), indent=2), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
