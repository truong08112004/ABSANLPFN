## Mục tiêu

Áp dụng đúng “flow” trong paper **“Multitask Aspect_Based Sentiment Analysis with Integrated BiLSTM & CNN Model”** lên dataset CSV của bạn (`ok050824.csv`), đồng thời dựng **2 model để train và so sánh**:

- **Model A (paper-like)**: GloVe/POS → CNN(3) → BiLSTM → pooling → dense → **2 heads** (aspect + sentiment)
- **Model B (Transformer-MTL, English)**: Transformer encoder (vd `roberta-base`) → pooling CLS → **2 heads** (aspect + sentiment)

## Input/label hiện có trong CSV của bạn

Từ header + các dòng mẫu:

- **Input text**: `reviewText` (văn bản review)
- **Các nhãn/metadata sẵn có**:
  - `sentiment` (ví dụ: Positive/Neutral/Negative) → dùng cho **polarity classification**
  - `aspect` (ví dụ: Convenience, Facilities, Staff quality, Data availability, Accessibility, …) → dùng cho **aspect/category classification** (tuỳ cách bạn định nghĩa “aspect” vs “category”)
  - `rating` (số sao) → metadata, có thể dùng bổ trợ hoặc bỏ qua (paper không bắt buộc)
  - `user_location`, `month`, `year`, `social` → metadata (tuỳ chọn)
  - `originalText` (nhiều dòng đang trống) → nếu có, có thể dùng làm input thay/so sánh với `reviewText`
  - `Unnamed: 0` → index cũ (thường nên bỏ)

Điểm khác với paper:

- Paper có **opinion target extraction** bằng IOB2 (sequence tagging). CSV hiện **chưa thấy cột target span/IOB2**, nên nhánh này sẽ cần **tạo nhãn mới** hoặc **bỏ qua**.

## Flow theo paper (chuẩn hoá thành pipeline cho dataset của bạn)

### 1) Chuẩn bị dữ liệu (Data preprocessing)

Theo paper (mục 3.3):

- **Data cleaning**:
  - loại ký tự lạ/emoticon
  - lowercase
  - bỏ số (tuỳ bạn, paper có)
  - bỏ câu rỗng
  - thay slang/viết tắt (nếu có từ điển)
  - lemmatize
- **Tokenize** `reviewText` → danh sách token theo thứ tự.

Đầu ra bước này:

- `tokens`: chuỗi token đã làm sạch cho từng review

### 2) Tạo đặc trưng theo paper

Theo paper (3.3 và 3.4.1/3.4.2):

- **Word embedding (GloVe)**:
  - paper dùng GloVe vector 100 chiều (train trên domain Restaurant SemEval 2016)
  - với dataset của bạn: có thể train GloVe trên toàn bộ `reviewText` hoặc dùng pretrained (tuỳ yêu cầu thực nghiệm)
- **POS tag**:
  - paper dùng Stanford POS Tagger
  - tạo **POS tag sequence** cho từng token
  - biến POS thành vector (paper nêu 34 chiều)
- **Concatenate**:
  - nối vector GloVe (100-d) + POS vector (34-d) → vector đặc trưng mỗi token
- **Padding/truncation**:
  - paper fix độ dài câu về max length của dataset (paper dùng 62)
  - dùng **0-padding** cho câu ngắn

Đầu ra bước này:

- `X`: tensor \([batch, seq_len, 134]\) nếu theo paper 100+34

### 3) Backbone CNN trích đặc trưng (3-layer CNN)

Theo paper: đưa `X` qua **3 lớp CNN** để lấy local features quanh mỗi token.

Đầu ra:

- `H_cnn`: biểu diễn sau CNN (vẫn theo chuỗi token hoặc đã biến đổi theo kiến trúc bạn hiện thực)

### 4) Hai nhánh nhiệm vụ theo paper

Paper mô tả 2 mô hình:

#### 4A) Nhánh 1 — Opinion target extraction (CNN-IOB2)

Flow paper (3.4.1):

- `tokens` → (GloVe + POS) → 3-layer CNN → **IOB2 tagging** (B/I/O) để trích “opinion target” (word/phrase)

Áp dụng lên CSV của bạn:

- Hiện CSV **chưa có nhãn target span/IOB2**, nên có 2 lựa chọn triển khai:
  - **Option A (đúng paper, đầy đủ multitask)**: tạo thêm dữ liệu gán nhãn target (span) rồi sinh IOB2.
  - **Option B (thực dụng, không có target labels)**: bỏ nhánh IOB2, chỉ làm classification tasks (4B).

Đầu ra nhánh này (nếu làm):

- `y_iob2`: chuỗi nhãn B/I/O theo token

#### 4B) Nhánh 2 — Multitask classification (Category + Aspect + Polarity)

Flow paper (3.4.2):

- (GloVe + POS) → CNN → **BiLSTM** (trên đỉnh CNN)
- → **average pooling**
- → **dense**
- → **softmax** cho các nhiệm vụ phân loại

Map sang CSV:

- **Polarity**: dùng `sentiment` làm nhãn (Positive/Neutral/Negative)
- **Aspect/Category**: paper tách “category (entity-aspect)” và “aspect”. CSV bạn có 1 cột `aspect`.
  - Nếu bạn muốn “đúng 3 subtasks còn lại” như paper (category + aspect + polarity), bạn cần:
    - **Cách 1**: coi `aspect` chính là **category** (entity-aspect) và bỏ task “aspect” (chỉ 2-task).
    - **Cách 2**: tách `aspect` thành 2 phần (entity, aspect) nếu dữ liệu có cấu trúc/chuẩn hoá được, rồi tạo 2 nhãn.
    - **Cách 3**: định nghĩa “aspect classification” = chính `aspect`, và “category classification” = nhóm hoá `aspect` theo taxonomy bạn đặt (nhãn mới).

Đầu ra nhánh này:

- `y_polarity`: nhãn từ `sentiment`
- `y_aspect_or_category`: nhãn từ `aspect` (hoặc nhãn suy ra)

## Multitask “chuẩn theo flow” khi huấn luyện

### Loss

Paper dùng softmax cho các task classification; khi multitask, bạn tối thiểu sẽ dùng:

- **Classification loss** cho `sentiment`
- **Classification loss** cho `aspect`/`category`
- (tuỳ chọn) **Sequence labeling loss** cho IOB2 nếu bạn có nhãn target

Tổng loss dạng:

\[
L = \lambda_1 L_{polarity} + \lambda_2 L_{aspect/category} + \lambda_3 L_{IOB2}
\]

Trong đó \(\lambda_i\) là trọng số (paper không nêu chi tiết trọng số; bạn có thể đặt mặc định = 1 và tuning).

### Hyperparameters được paper nêu (để bám sát khi tái hiện)

Paper báo cáo (mục 4.2):

- **GloVe dim**: 100
- **POS vector**: 34
- **max sequence length**: 62 (trên SemEval restaurant)
- **optimizer**: rmsprop, learning rate 0.001
- **dropout**: 0.25
- **batch size**: 128
- **epochs**:
  - CNN-IOB2: 125
  - MABSA: 300

Khi áp dụng lên CSV của bạn, các giá trị này là điểm khởi đầu; cần chỉnh theo độ dài câu và kích thước dataset thực tế.

## “Multitask chuẩn” trên dataset của bạn: khuyến nghị cấu hình tối thiểu

Vì CSV hiện đã có `aspect` và `sentiment`, flow khả thi nhất (không cần gán nhãn mới) là:

- **Input**: `reviewText`
- **Task A**: dự đoán `aspect`
- **Task B**: dự đoán `sentiment`
- **Backbone đúng paper**: GloVe + POS → CNN → BiLSTM → pooling → dense → 2 softmax heads

Ngoài ra (vì review là tiếng Anh), để có 1 model mạnh và dễ fine-tune, có thể dùng Transformer encoder cho multitask:

- **Backbone Transformer-MTL**: Tokenizer → Transformer encoder (vd `roberta-base`) → CLS pooling → dense/dropout → 2 softmax heads

Nếu bạn muốn “đúng 4 subtasks” như paper (có opinion target):

- Bổ sung annotation để tạo `y_iob2` (sequence labels) cho `reviewText`
- Dùng chung embedding/POS/CNN backbone, thêm head IOB2 song song với các head classification

## Checklist thực thi (đúng trình tự flow)

- **Bước 0**: lọc cột thừa (`Unnamed: 0`), chọn text input (`reviewText` hoặc `originalText` nếu có đủ)
- **Bước 1**: cleaning + tokenize
- **Bước 2**: train/lookup GloVe embeddings
- **Bước 3**: POS tagging + encode POS vector
- **Bước 4**: pad/truncate về `max_len` (tính từ dataset; paper dùng 62)
- **Bước 5**: build CNN (3-layer) → BiLSTM → avg pooling → dense
- **Bước 6**: heads
  - head sentiment: softmax over {Negative, Neutral, Positive}
  - head aspect/category: softmax over tập nhãn `aspect`
  - (tuỳ chọn) head IOB2: token-level softmax over {B, I, O}
- **Bước 7**: train multitask với tổng loss (weighted sum)
- **Bước 8**: evaluate
  - paper dùng Accuracy và F1; bạn có thể report F1 macro cho `sentiment`/`aspect`, và token-level F1 cho IOB2 nếu có

## Checklist để train 2 model và report so sánh

- **Chung cho cả hai**:
  - dùng cùng **split train/val** (cố định random seed)
  - report **Accuracy + macro-F1** cho cả `aspect` và `sentiment`
- **Model A (paper-like)**:
  - preprocessing theo paper (clean/tokenize/POS) + pad/truncate
  - CNN(3) → BiLSTM → pooling → 2 heads
- **Model B (Transformer-MTL)**:
  - dùng tokenizer của Transformer, truncation theo `max_len`
  - encoder chung → 2 heads

