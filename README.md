## ABSA multitask: 2 model để so sánh

Project này có 2 model để train và so sánh trên cùng dataset:

- **Model A (paper-like)**: CNN(3) + BiLSTM theo flow paper (có POS + embedding).
- **Model B (Transformer-MTL, English)**: encoder Transformer (vd `roberta-base`) + 2 head multitask.

Model A bám flow chính trong paper:

- cleaning + tokenize
- word embedding dim 100
- POS tag (mặc định NLTK; nếu thiếu resource sẽ fallback)
- concat embedding + POS vector
- 3-layer CNN → BiLSTM → average pooling → dense → 2 softmax heads
  - head 1: `aspect`
  - head 2: `sentiment`

Lưu ý: paper có nhánh IOB2 để trích opinion target; CSV hiện chưa có nhãn IOB2 nên project này tập trung vào multitask classification (aspect + sentiment).

### Chạy với uv

Tại thư mục `/home/enterprise/Desktop/ABSA`:

```bash
uv sync
uv run absa-train --csv ok050824.csv --text-col reviewText --aspect-col aspect --sentiment-col sentiment
uv run absa-compare --csv ok050824.csv --epochs 1 --out-md report-compare.md
```

### Các tham số hay dùng

- `--max-len`: mặc định 62 (giống paper; bạn có thể tăng/giảm theo dataset)
- `--epochs`: mặc định 1 để smoke test; tăng lên để train thật
- `--batch-size`: mặc định 128 (giống paper)

