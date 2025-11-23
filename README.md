# Hệ thống Gợi ý Tin tức Cá nhân hóa - MIND Dataset

Dự án xây dựng hệ thống gợi ý tin tức cá nhân hóa sử dụng Deep Learning trên bộ dữ liệu MIND (Microsoft News Dataset).


## 🎯 Tổng quan

### Bài toán
Xây dựng hệ thống **Personalized News Recommendation** - gợi ý tin tức cá nhân hóa cho người dùng dựa trên:
- Lịch sử đọc tin
- Hành vi tương tác
- Nội dung bài viết (title, abstract, category, entities)

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
Dự án sử dụng bộ dữ liệu MIND-small. Bạn có thể tải về từ Hugging Face:

```bash
# Tạo thư mục data
mkdir -p data

# Tải và giải nén MIND-small (Training set)
wget https://huggingface.co/datasets/yjw1029/MIND/resolve/main/MINDsmall_train.zip -O data/MINDsmall_train.zip
cd data
unzip MINDsmall_train.zip -d MINDsmall_train
rm MINDsmall_train.zip
cd ..
```

Cấu trúc thư mục sau khi giải nén:
```
data/
└── MINDsmall_train/
    ├── behaviors.tsv
    ├── news.tsv
    ├── entity_embedding.vec
    └── relation_embedding.vec
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
    --data_dir data/MINDsmall_train \
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