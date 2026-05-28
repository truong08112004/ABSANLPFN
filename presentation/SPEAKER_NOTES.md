# Gợi ý thuyết trình — Sentiment Analysis on Review Text

Tài liệu này đi song song với `index.html` (Reveal.js). Mỗi mục gồm **nội dung slide**, **gợi ý nói** (~1–2 phút/slide phần chính), và **điểm nhấn** nếu bị hỏi.

**Chạy slide:** `cd presentation && python -m http.server 8080` → mở http://localhost:8080

---

## Phần chính (Main deck)

### Slide 1 — Title · *Sentiment Analysis on Review Text*

**Trên slide:** Tiêu đề, joint aspect + sentiment, dòng phụ CNN + BiLSTM vs BERT.

**Gợi ý nói:**
> Xin chào. Hôm nay em trình bày đề tài **Sentiment Analysis on Review Text** — tức phân tích cảm xúc và khía cạnh được nhắc tới **trên văn bản review**.
>
> Mục tiêu: từ một câu review tiếng Anh, mô hình dự đoán đồng thời **review đang nói về khía cạnh nào** (aspect) và **thái độ tích cực / trung tính / tiêu cực** (sentiment).
>
> Em so sánh hai hướng: **CNN + BiLSTM** (học từ đầu, có đặc trưng POS) và **BERT** (fine-tune encoder tiếng Anh có sẵn).

**Điểm nhấn:** Một câu review → hai nhãn; multitask; hai backbone để so sánh công bằng.

**Thời gian gợi ý:** ~45 giây

---

### Slide 2 — Problem

**Trên slide:** ABSA, input `reviewText`, task aspect & sentiment, lý do multitask.

**Gợi ý nói:**
> Bài toán thuộc nhóm **Aspect-Based Sentiment Analysis (ABSA)**: không chỉ hỏi “câu này tích cực hay tiêu cực”, mà còn “đang nói về **cái gì**”.
>
> **Input** là cột `reviewText` — câu review tiếng Anh, ví dụ lĩnh vực giao thông / dịch vụ.
>
> **Hai nhãn:** `aspect` (nhiều lớp: Convenience, Staff quality, Facilities, …) và `sentiment` (Positive, Neutral, Negative).
>
> Em dùng **multitask**: một encoder đọc câu một lần, hai đầu ra song song — context dùng chung giúp cả hai task, thay vì train hai model riêng.

**Điểm nhấn:** ABSA ≠ sentiment đơn thuần; multitask = shared encoder.

**Thời gian gợi ý:** ~1 phút

---

### Slide 3 — Dataset

**Trên slide:** Bảng cột CSV; size, split, max_len, thách thức.

**Gợi ý nói:**
> Dữ liệu file **`ok050824.csv`**: mỗi dòng một review kèm nhãn aspect và sentiment.
>
> Tổng khoảng **104 nghìn dòng**; thực nghiệm em lấy **30 nghìn mẫu** để train nhanh hơn, vẫn đủ đại diện.
>
> **Chia dữ liệu:** 70% train, 15% validation, 15% test — cùng seed cho cả hai model.
>
> **Độ dài chuỗi:** LSTM pad/truncate **62 token**; BERT **128 subword** vì tokenizer WordPiece.
>
> **Khó hơn** là aspect — nhiều lớp, dễ nhầm giữa các khía cạnh gần nghĩa; sentiment chỉ 3–4 lớp nên thường dễ hơn.

**Điểm nhấn:** Cùng split → so sánh model công bằng; aspect khó hơn sentiment.

**Thời gian gợi ý:** ~1 phút

---

### Slide 4 — Multitask learning in this project

**Trên slide:** Sơ đồ loss multitask + 5 bullet.

**Gợi ý nói:**
> Đây là **cách em triển khai multitask** trong project.
>
> Một **backbone chung** đọc toàn bộ review → vector đặc trưng dùng chung.
>
> Từ đó tách **hai head:** một softmax cho aspect, một cho sentiment.
>
> Loss huấn luyện: **L = L_aspect + L_sentiment** — mỗi head cross-entropy, cộng lại rồi backprop một lần qua backbone.
>
> **CNN + BiLSTM** và **BERT** khác nhau ở phần encoder; **phần hai head và ý tưởng multitask giống nhau**.

**Điểm nhấn:** Một forward pass → hai dự đoán; gradient từ cả hai task cập nhật encoder.

**Thời gian gợi ý:** ~1–1,5 phút

---

### Slide 5 — End-to-end pipeline

**Trên slide:** Sơ đồ pipeline (CSV → preprocess → 2 model → 2 heads → loss).

**Gợi ý nói:**
> Toàn bộ luồng xử lý như trên sơ đồ.
>
> Bắt đầu từ CSV → **tiền xử lý** (clean, tokenize; LSTM thêm POS).
>
> Rẽ nhánh **Model A:** CNN + BiLSTM — đặc trưng local + ngữ cảnh dài.
>
> **Model B:** BERT — embedding ngữ cảnh sâu, pretrained.
>
> Cả hai đều kết thúc bằng **hai head** và **tổng loss** — đây là chỗ so sánh công bằng nhất.

**Điểm nhấn:** Preprocess khác nhánh; backbone khác; phần cuối giống nhau.

**Thời gian gợi ý:** ~1 phút

---

### Slide 6 — Preprocessing

**Trên slide:** Sơ đồ preprocessing.

**Gợi ý nói:**
> Chi tiết **tiền xử lý**:
>
> **Clean:** lowercase, bỏ ký tự lạ, lemmatize.
>
> **Tokenize** → vocab có PAD/UNK cho LSTM.
>
> **POS tagging** (NLTK) — chỉ nhánh LSTM: gắn thêm thông tin ngữ pháp từng token.
>
> **Pad/truncate** về độ dài cố định để batch trên GPU.
>
> Nhánh **BERT** bỏ POS; dùng tokenizer `bert-base-uncased` + **attention mask** — chuẩn fine-tune Transformer.

**Điểm nhấn:** LSTM = word + POS; BERT = subword, không POS.

**Thời gian gợi ý:** ~1 phút

---

### Slide 7 — Model A — CNN + BiLSTM

**Trên slide:** Sơ đồ layer LSTM + bullet + hyperparams.

**Gợi ý nói:**
> **Model A** — kiến trúc chính trong code là `MultitaskAbsaModel`:
>
> Embedding từ **100 chiều** + vector **POS one-hot** → concat theo từng token.
>
> **Ba lớp Conv1d** (kernel 3) bắt pattern cục bộ (n-gram) quanh mỗi vị trí.
>
> **BiLSTM** một lớp, hai chiều, hidden 128 → mỗi token 256 chiều; sau đó **average pooling** gom cả câu thành một vector.
>
> **FC + dropout** rồi **hai linear head** → logits aspect và sentiment.
>
> Train: **100 epoch**, batch 128, lr 0.001, dropout 0.25 — học từ đầu, không pretrained.

**Điểm nhấn:** CNN local + LSTM global; pool → 2 heads.

**Thời gian gợi ý:** ~1,5 phút

---

### Slide 8 — Model B — BERT

**Trên slide:** Sơ đồ BERT + hyperparams.

**Gợi ý nói:**
> **Model B** — class `TransformerMtlModel`, backbone **`bert-base-uncased`**:
>
> **12 lớp Transformer**, hidden **768**, 12 attention heads — kiến thức tiếng Anh có sẵn.
>
> Lấy representation **token CLS** (hoặc pooler) làm vector cả câu.
>
> **Dropout** trước head; **cùng hai head** aspect/sentiment như Model A.
>
> Fine-tune end-to-end: chỉ **5 epoch**, batch 64, lr **2e-5** — điển hình fine-tune BERT, nhanh hơn LSTM về số epoch nhưng model lớn hơn.

**Điểm nhấn:** Pretrained English; CLS pooling; ít epoch, lr nhỏ.

**Thời gian gợi ý:** ~1,5 phút

---

### Slide 9 — Multitask loss

**Trên slide:** Lại sơ đồ loss + công thức và chi tiết optimizer.

**Gợi ý nói:**
> Nhắc lại **cách train** (có thể nói ngắn vì đã có slide 4):
>
> **L = L_aspect + L_sentiment**, trọng số λ = 1.
>
> Sai aspect hoặc sai sentiment đều **kéo gradient** về backbone.
>
> LSTM run dùng kiểu **RMSprop / lr 1e-3**; BERT dùng **Adam, 2e-5**.
>
> Project còn có script `absa-tune` để tune hyperparameter bằng PSO nếu cần — không trình bày sâu hôm nay.

**Điểm nhấn:** Joint training; shared gradient.

**Thời gian gợi ý:** ~45 giây – 1 phút

---

### Slide 10 — Results (test set)

**Trên slide:** Bảng metrics test, 3 bullet nhận xét.

**Gợi ý nói:**
> Kết quả trên **tập test** (cùng 30k mẫu, cùng split):
>
> | | Aspect Acc | Aspect F1 | Sent Acc | Sent F1 |
> |---|:---:|:---:|:---:|:---:|
> | CNN+BiLSTM | 60.1% | 45.2% | 86.9% | 78.1% |
> | **BERT** | **70.7%** | **54.6%** | **93.0%** | **87.4%** |
>
> **BERT mạnh hơn rõ ở aspect** — khoảng **+10.6 điểm accuracy**; đúng với task khó hơn.
>
> **Sentiment** cả hai đã cao; BERT thêm ~6 điểm accuracy.
>
> **F1 aspect** vẫn thấp hơn sentiment → nhiều lớp aspect, mất cân bằng lớp.

**Điểm nhấn:** BERT thắng chủ yếu ở aspect; sentiment gần trần.

**Thời gian gợi ý:** ~1,5 phút

---

### Slide 11 — Training — CNN + BiLSTM

**Trên slide:** `lstm_training_curves.png` + 3 bullet.

**Gợi ý nói:**
> Biểu đồ huấn luyện **LSTM**:
>
> **Train loss** giảm đều qua 100 epoch.
>
> **Validation** có lúc chững / dao động — dấu hiệu **overfit nhẹ** sau giữa training.
>
> **Accuracy aspect** thấp hơn sentiment trên mọi epoch — khớp với bảng kết quả.

**Điểm nhấn:** Overfit nhẹ; aspect luôn khó hơn.

**Thời gian gợi ý:** ~45 giây – 1 phút

---

### Slide 12 — Confusion matrix — CNN + BiLSTM

**Trên slide:** `lstm_confusion_matrix.png`.

**Gợi ý nói:**
> **Confusion matrix** LSTM — thường hai ma trận: trái **aspect**, phải **sentiment**.
>
> **Aspect:** nhiều điểm ngoài đường chéo → nhầm giữa các lớp aspect tương tự (ví dụ Facilities vs Accessibility).
>
> **Sentiment:** đường chéo rõ hơn; hay nhầm **Neutral** với Positive/Negative.
>
> Ma trận giúp biết **lớp nào cần thêm dữ liệu** hoặc feature.

**Điểm nhấn:** Aspect confused nhiều; sentiment ổn hơn.

**Thời gian gợi ý:** ~1 phút

---

### Slide 13 — Training — BERT

**Trên slide:** `bert_training_curves.png`.

**Gợi ý nói:**
> **BERT** chỉ **5 epoch** đã hội tụ tốt.
>
> Cải thiện mạnh **3 epoch đầu** — đặc trưng fine-tune Transformer.
>
> **Val loss** ổn định → generalize tốt hơn LSTM trong run này.
>
> Cuối cùng **vượt LSTM** cả aspect lẫn sentiment với ít epoch hơn nhiều.

**Điểm nhấn:** Fine-tune nhanh; val ổn.

**Thời gian gợi ý:** ~45 giây – 1 phút

---

### Slide 14 — Confusion matrix — BERT

**Trên slide:** `bert_confusion_matrix.png`.

**Gợi ý nói:**
> **BERT confusion matrix:**
>
> **Đường chéo aspect** đậm hơn LSTM → phân tách lớp aspect tốt hơn.
>
> **Sentiment** gần như tách sạch; ít nhầm Neutral.
>
> Lỗi còn lại chủ yếu **lớp aspect hiếm** hoặc câu **mơ hồ** (nhiều khía cạnh trong một review).

**Điểm nhấn:** BERT giảm confusion aspect rõ rệt.

**Thời gian gợi ý:** ~1 phút

---

### Slide 15 — Takeaways

**Trên slide:** 4 bullet (hiện hết một lần, không fragment).

**Gợi ý nói:**
> **Tóm lại:**
>
> 1. **Multitask khả thi** — một encoder, hai task, inference một lần.
> 2. **BERT** là baseline mạnh hơn khi có GPU và weight pretrained.
> 3. **CNN + BiLSTM** vẫn hữu ích: nhẹ hơn, dễ giải thích (POS, conv local).
> 4. **Aspect** là nút thắt; **sentiment** gần đạt trần trên subset này.
>
> Cảm ơn mọi người đã lắng nghe.

**Điểm nhấn:** Kết luận ngắn, rõ; không mở rộng future work trừ khi bị hỏi.

**Thời gian gợi ý:** ~1 phút

---

### Slide 16 — Closing · *Thank you*

**Trên slide:** Thank you, Questions & discussion.

**Gợi ý nói:**
> Em sẵn sàng nhận câu hỏi.

*(Nếu không có câu hỏi, kết thúc lịch sự.)*

**Thời gian gợi ý:** ~15 giây

---

## Phụ lục (Appendix) — dùng khi bị hỏi sâu

> **Lưu ý:** Appendix nằm **sau** slide Thank you trong file HTML. Khi thuyết trình, có thể **nhảy tới slide Appendix** (phím điều hướng / số slide) hoặc mở sẵn trên máy. Không bắt buộc trình bày hết.

### Appendix divider — Model architecture reference

> “Phần sau là chi tiết kỹ thuật nếu thầy/cô hỏi về layer, shape tensor, hoặc hyperparameter.”

---

### Appendix A — CNN + BiLSTM overview

**Gợi ý nói ngắn:**
> Class `MultitaskAbsaModel`, input `word_ids` + `pos_ids` shape `[B, L]`, L=62, vocab ~12.8k, 13 aspect / 4 sentiment classes, loss CE cộng.

---

### Appendix A — Layer stack (1/2) & (2/2)

**Gợi ý nói khi bị hỏi “từng layer”:**
> Đi theo bảng: emb 100d + POS → concat 103d → 3× CNN 128 kênh → BiLSTM 128×2 → pool 256d → FC → head 13 và 4. Shape từng bước em có ghi trên slide.

---

### Appendix A — Why these sizes?

**Gợi ý nói:**
> L=62 theo độ dài pad dataset; 100d embedding và 128 CNN là default trong `model.py`; 13/4 class đếm từ nhãn trong subset 30k. Có thêm `CnnOnlyMultitaskModel` (bỏ BiLSTM) để ablation.

---

### Appendix B — BERT overview & encoder internals

**Gợi ý nói:**
> `bert-base-uncased`: 12 layer, 12 heads, hidden 768, input max 128 token, CLS 768d → hai linear head. Fine-tune full encoder + heads, 5 epoch, lr 2e-5.

---

### Appendix — Side-by-side comparison

**Gợi ý nói:**
> So sánh nhanh: LSTM word+POS, 62 token, ~1–2M params trainable; BERT WordPiece, 128 token, ~110M params pretrained. LSTM 100 epoch; BERT 5 epoch fine-tune.

---

## Câu hỏi thường gặp (FAQ ngắn)

| Câu hỏi | Gợi ý trả lời |
|--------|----------------|
| Sao không train riêng 2 model? | Multitask dùng chung context; ít tham số hơn; một forward pass. |
| Vì sao BERT tốt hơn? | Pretrained ngữ cảnh tiếng Anh; subword OOV tốt; representation 768d sâu hơn. |
| Aspect F1 thấp? | Nhiều lớp, imbalance; có thể cần balance data / focal loss. |
| POS vocab chỉ 3? | POS được collapse/fallback trong pipeline; chi tiết trong `textproc.py`. |
| Loss có trọng số λ không? | Mặc định λ=1; có thể tune qua `absa-tune` / PSO. |
| Dataset 30k có đủ? | Đủ để so sánh hai model; full 104k có thể cải thêm nếu có thời gian train. |

---

## Gợi ý thời lượng tổng (phần chính)

| Phần | Slide | ~Thời gian |
|------|-------|-----------|
| Mở đầu | 1 | 0:45 |
| Bài toán & data | 2–3 | 2:00 |
| Multitask + pipeline | 4–6 | 3:30 |
| Models + loss | 7–9 | 4:00 |
| Kết quả | 10 | 1:30 |
| Biểu đồ & CM | 11–14 | 3:30 |
| Kết luận | 15–16 | 1:15 |
| **Tổng** | **16** | **~17–20 phút** |

*(Appendix không tính; Q&A riêng.)*

---

*Tài liệu tạo theo `presentation/index.html` · metrics từ `outputs/train-lstm/20260528-235501` và `outputs/train-bert/20260528-232134`.*
