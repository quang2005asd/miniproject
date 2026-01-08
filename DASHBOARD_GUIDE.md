# Hướng dẫn chạy Dashboard

## 1. Chạy Pipeline để tạo dữ liệu kết quả

Trước khi chạy dashboard, cần chạy pipeline để tạo các file kết quả:

```bash
python run_papermill.py
```

Pipeline này sẽ tạo các file trong `data/processed/`:
- `metrics.json` - Kết quả supervised baseline
- `metrics_self_training.json` - Kết quả self-training
- `metrics_co_training.json` - Kết quả co-training
- Các file predictions và alerts tương ứng

## 2. Cài đặt thư viện (nếu chưa có)

```bash
pip install streamlit seaborn
```

Hoặc cài đặt lại toàn bộ từ requirements.txt:

```bash
pip install -r requirements.txt
```

## 3. Chạy Dashboard

```bash
streamlit run app.py
```

Dashboard sẽ mở tự động trên browser tại `http://localhost:8501`

## 4. Tính năng Dashboard

### 📈 Overview
- Thông tin tổng quan về dataset
- Số lượng methods đã train
- Thông tin về 6 mức AQI

### 🔬 Model Comparison
- So sánh metrics (Accuracy, F1-Macro) giữa các methods
- Confusion matrices
- Bảng so sánh chi tiết

### 🔄 Self-Training Analysis
- Metrics tổng quan (Accuracy, F1, số pseudo-labels, iterations)
- Biểu đồ diễn biến qua các vòng lặp
- Chi tiết từng iteration
- Confusion matrix
- Classification report

### 🔀 Co-Training Analysis
- Metrics cho cả 2 models
- Biểu đồ trao đổi pseudo-labels giữa 2 models
- Performance evolution của 2 models
- Confusion matrix (ensemble)
- Chi tiết từng iteration

### 🚨 Predictions & Alerts
- Xem predictions của từng method
- Download predictions
- Phân tích alerts (tổng số, tỷ lệ)
- Alerts theo từng trạm
- Class distribution (True vs Predicted)

## 5. Visualization Utilities

File `src/visualization_utils.py` cung cấp các hàm vẽ biểu đồ:

```python
from src.visualization_utils import (
    plot_confusion_matrix,
    plot_self_training_progress,
    plot_cotraining_progress,
    plot_method_comparison,
    plot_class_performance,
    plot_alert_analysis,
    create_report_summary
)
```

Có thể sử dụng trong notebooks để tạo các biểu đồ phân tích.

## 6. Lưu ý

- Dashboard yêu cầu đã chạy pipeline và có dữ liệu trong `data/processed/`
- Nếu chưa có dữ liệu, dashboard sẽ hiển thị thông báo yêu cầu chạy pipeline
- Có thể custom dashboard trong file `app.py`
- Các plot functions có thể dùng độc lập trong notebooks
