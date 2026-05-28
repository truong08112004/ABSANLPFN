## Flow pipeline + vị trí Multitask (2 model để so sánh)

```mermaid
flowchart TD
  A[ok050824.csv] --> B[Chọn cột\nX=reviewText\nY1=aspect, Y2=sentiment]
  B --> C{Chọn model}

  %% ===== Model A: paper-like =====
  C -->|Model A: Paper-like| A1[Cleaning + tokenize\n(lowercase, bỏ ký tự lạ...)]
  A1 --> A2[POS tagging\n(NLTK, fallback 'X')]
  A1 --> A3[Word IDs\n(vocab + pad/truncate)]
  A2 --> A4[POS IDs\n(vocab + pad/truncate)]
  A3 --> A5[Word embedding\n100-dim]
  A4 --> A6[POS one-hot\n(|POS| dim)]
  A5 --> A7[Concat\n[word_emb ; pos_vec]]
  A6 --> A7
  A7 --> A8[CNN layer 1]
  A8 --> A9[CNN layer 2]
  A9 --> A10[CNN layer 3]
  A10 --> A11[BiLSTM]
  A11 --> A12[Average pooling]
  A12 --> A13[Dense + Dropout]
  A13 --> A14[Head aspect]
  A13 --> A15[Head sentiment]
  A14 --> A16[Loss_aspect]
  A15 --> A17[Loss_sentiment]
  A16 --> A18[Tổng loss]
  A17 --> A18

  %% ===== Model B: Transformer-MTL =====
  C -->|Model B: Transformer-MTL (English)| B1[Tokenizer (vd roberta-base)\npadding + truncation]
  B1 --> B2[Transformer encoder]
  B2 --> B3[CLS pooling]
  B3 --> B4[Dense/Dropout]
  B4 --> B5[Head aspect]
  B4 --> B6[Head sentiment]
  B5 --> B7[Loss_aspect]
  B6 --> B8[Loss_sentiment]
  B7 --> B9[Tổng loss]
  B8 --> B9

  %% ===== Optimizer =====
  A18 --> OPT[Backprop + optimizer]
  B9 --> OPT
```

- **Multitask ở đâu**: ở cả hai model, sau backbone chung (Model A: tới `Dense`; Model B: tới `CLS/Dense`) tách thành **2 head song song** (`aspect` và `sentiment`) và tối ưu **đồng thời** bằng tổng loss.

