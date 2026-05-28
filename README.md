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

### PSO tuning + back-translation balancing (khuyến nghị)

Pipeline mới dùng `absa-tune` để:
- tune hyperparameters bằng PSO cho cả 2 model
- train lại final model với hyperparams tốt nhất
- lưu đầy đủ metrics/charts/report/artifacts
- có thể cân bằng train data bằng back-translation theo `sentiment`

Chạy đầy đủ:

```bash
uv sync
uv run absa-tune \
  --csv ok050824.csv \
  --text-col reviewText \
  --aspect-col aspect \
  --sentiment-col sentiment \
  --enable-back-translation \
  --balance-target sentiment \
  --bt-from-model Helsinki-NLP/opus-mt-en-fr \
  --bt-to-model Helsinki-NLP/opus-mt-fr-en \
  --pso-epoch 15 \
  --pso-pop-size 12 \
  --train-epochs 10 \
  --out-dir outputs/tuning
```

Smoke test chạy nhanh:

```bash
uv run absa-tune \
  --csv ok050824.csv \
  --max-samples 80 \
  --train-epochs 1 \
  --pso-epoch 1 \
  --pso-pop-size 5 \
  --enable-back-translation \
  --out-dir outputs/tuning-smoke
```

Artifacts quan trọng sau mỗi run (trong `outputs/tuning/<run_id>/`):
- `best_params_model_a.json`, `best_params_model_b.json`
- `run_config.json`
- `class_distribution_before_balance.csv`
- `class_distribution_after_balance.csv`
- `augmentation_log.csv`
- `balance_summary.json`
- `final_metrics.csv`, `final_metrics.md`
- `pso_convergence_model_a.png`, `pso_convergence_model_b.png`
- `training_curves_model_a.png`, `training_curves_model_b.png`
- `report.md`

