# Hệ thống Gợi ý Tin tức Cá nhân hóa - MIND Dataset

Dự án xây dựng hệ thống gợi ý tin tức cá nhân hóa sử dụng Deep Learning trên bộ dữ liệu MIND (Microsoft News Dataset).


## 🎯 Tổng quan

### Bài toán
Xây dựng hệ thống **Personalized News Recommendation** - gợi ý tin tức cá nhân hóa cho người dùng dựa trên:
- Lịch sử đọc tin
- Hành vi tương tác
- Nội dung bài viết (title, abstract, category, entities)

### Mô hình
**Hybrid Model** kết hợp:
- **Fastformer**: Attention hiệu quả O(n)
- **Entity Knowledge**: Khai thác tri thức từ WikiData
- **Multi-interest Modeling**: Mô hình hóa nhiều sở thích người dùng

### Bộ dữ liệu
**MIND (Microsoft News Dataset)**
- ~1M users, ~161K articles, ~24M clicks
- Download: https://msnews.github.io/

## 🚀 Cài đặt

### 1. Tạo môi trường ảo
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows PowerShell
```

### 2. Cài đặt dependencies
```bash
pip install -r requirements.txt
```

### 3. Tải dữ liệu MIND
Tải và giải nén MIND dataset vào thư mục `data/`:
```
data/
├── news.tsv
├── train/
│   └── behaviors.tsv
├── val/
│   └── behaviors.tsv
└── test/
    └── behaviors.tsv
```

## 📁 Cấu trúc Dự án

```
NewsRecommender/
├── src/
│   ├── data/
│   │   ├── mind_dataset.py      # Dataset loader
│   │   └── text_encoder.py      # Text encoding (BERT/GloVe)
│   ├── models/
│   │   ├── news_encoder.py      # News Encoder module
│   │   ├── user_encoder.py      # User Encoder module
│   │   └── hybrid_model.py      # Complete Hybrid model
│   ├── utils/
│   │   ├── config.py            # Configuration
│   │   └── metrics.py           # Evaluation metrics
│   └── training/
│       ├── train.py             # Training script
│       └── evaluate.py          # Evaluation script
├── CHECKLIST.md                 # Checklist thực hiện
├── MO_TA_DU_AN.md              # Mô tả chi tiết dự án
├── requirements.txt
└── README.md
```

## 💻 Sử dụng

### 1. Huấn luyện

```bash
python -m src.training.train \
    --data_dir data \
    --output_dir outputs \
    --device cuda
```

**Tùy chọn:**
- `--config`: Path to JSON config file
- `--device`: cuda hoặc cpu

### 2. Đánh giá

```bash
python -m src.training.evaluate \
    --checkpoint outputs/checkpoints/best_model.pt \
    --data_dir data \
    --split test \
    --device cuda
```

### 3. Cấu hình

Tạo file `config.json` để tùy chỉnh:

```json
{
  "model": {
    "text_encoder_type": "bert",
    "bert_model_name": "bert-base-uncased",
    "max_title_len": 30,
    "max_abstract_len": 100,
    "num_interests": 5,
    "news_encoder_dim": 768,
    "user_encoder_dim": 768
  },
  "training": {
    "batch_size": 64,
    "num_epochs": 20,
    "learning_rate": 1e-4,
    "num_negatives": 4
  },
  "data": {
    "data_dir": "data",
    "num_workers": 4
  }
}
```

## 📊 Chỉ số Đánh giá

- **AUC**: Area Under ROC Curve
- **MRR**: Mean Reciprocal Rank
- **nDCG@5**: Normalized Discounted Cumulative Gain @ 5
- **nDCG@10**: Normalized Discounted Cumulative Gain @ 10

### Baseline Results (Reference)
| Model | AUC | MRR | nDCG@5 | nDCG@10 |
|-------|-----|-----|--------|---------|
| NAML | 0.6686 | 0.3249 | 0.3524 | 0.4091 |
| NRMS | 0.6776 | 0.3305 | 0.3594 | 0.4163 |
| MINER | 0.7275 | 0.3724 | 0.4102 | 0.4661 |
| Fastformer | 0.7268 | 0.3745 | 0.4151 | 0.4684 |
