# ABSA Multitask Presentation

Reveal.js slides comparing **CNN + BiLSTM** and **BERT** on multitask aspect + sentiment classification.

## Speaker notes

Script thuyết trình từng slide (tiếng Việt): [`SPEAKER_NOTES.md`](SPEAKER_NOTES.md)

## Run locally

```bash
cd presentation
python -m http.server 8080
# Open http://localhost:8080
```

## Appendix (Q&A backup)

After **Takeaways**, slides tagged **Appendix** cover:

- **Main deck**: “Multitask learning in this project” (after Dataset) — 2 heads, shared encoder, joint loss
- **Appendix A**: `MultitaskAbsaModel` — inputs, layer stack (2 slides, full-width table), hyperparameters
- **Appendix B**: `TransformerMtlModel` / `bert-base-uncased` — 12 layers, 768-d, CLS pooling
- **Comparison table**: CNN+BiLSTM vs BERT side-by-side

Numbers (vocab 12884, 13 aspect / 4 sentiment classes, L=62/128) come from `outputs/train-*/run_config.json`.

## Assets

| File | Slide |
|------|--------|
| `assets/diagrams/01-pipeline-flow.png` | End-to-end pipeline |
| `assets/diagrams/03-data-preprocessing.png` | Preprocessing |
| `assets/diagrams/02-model-lstm-layers.png` | CNN + BiLSTM |
| `assets/diagrams/04-model-bert-layers.png` | BERT |
| `assets/diagrams/05-multitask-loss.png` | Multitask loss |
| `assets/lstm_*.png`, `assets/bert_*.png` | Training curves & confusion matrices |

## Excalidraw canvas (optional edits)

```bash
cd ~/.local/share/mcp_excalidraw && node dist/server.js
```

Open http://127.0.0.1:3000 — keep the tab open for MCP export.
