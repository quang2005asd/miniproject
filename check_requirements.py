import json
from pathlib import Path

print("=" * 80)
print("KIỂM TRA YÊU CẦU MINI PROJECT")
print("=" * 80)

# Load metrics
DATA_PROCESSED = Path("data/processed")

baseline = json.load(open(DATA_PROCESSED / "metrics.json"))
self_training = json.load(open(DATA_PROCESSED / "metrics_self_training.json"))
co_training = json.load(open(DATA_PROCESSED / "metrics_co_training.json"))

print("\n" + "="*80)
print("YÊU CẦU 1: SELF-TRAINING")
print("="*80)

st_cfg = self_training.get('st_cfg', {})
st_history = self_training.get('history', [])
st_test = self_training.get('test_metrics', {})

print(f"\n✅ Đã huấn luyện Self-Training:")
print(f"   - Tau (τ): {st_cfg.get('tau', 'N/A')}")
print(f"   - Iterations: {len(st_history)}")
print(f"   - Test Accuracy: {st_test.get('accuracy', 0):.4f}")
print(f"   - Test F1-Macro: {st_test.get('f1_macro', 0):.4f}")

print(f"\n⚠️  THIẾU - Thay đổi ngưỡng τ:")
print(f"   - Hiện chỉ có τ=0.9")
print(f"   - Cần: Thử τ ∈ {{0.8, 0.85, 0.9, 0.95}} và so sánh")

print(f"\n⚠️  THIẾU - Biểu đồ diễn biến:")
print(f"   - Có history data: ✓")
print(f"   - Có visualization code: ✓ (trong visualization_utils.py)")
print(f"   - Cần: Notebook hoặc script tạo biểu đồ")

print(f"\n✅ So sánh với baseline:")
baseline_acc = baseline.get('test_accuracy', 0)
baseline_f1 = baseline.get('test_f1_macro', 0)
st_acc = st_test.get('accuracy', 0)
st_f1 = st_test.get('f1_macro', 0)

print(f"   Baseline:      Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
print(f"   Self-Training: Acc={st_acc:.4f}, F1={st_f1:.4f}")
print(f"   Improvement:   Acc={st_acc-baseline_acc:+.4f}, F1={st_f1-baseline_f1:+.4f}")

print(f"\n⚠️  THIẾU - Phân tích per-class:")
print(f"   - Có classification report: ✓")
print(f"   - Cần: So sánh từng class với baseline")

print("\n" + "="*80)
print("YÊU CẦU 2: CO-TRAINING")
print("="*80)

ct_cfg = co_training.get('ct_cfg', {})
ct_history = co_training.get('history', [])
ct_test = co_training.get('test_metrics', {})

print(f"\n✅ Đã huấn luyện Co-Training:")
print(f"   - Tau (τ): {ct_cfg.get('tau', 'N/A')}")
print(f"   - Iterations: {len(ct_history)}")
print(f"   - Test Accuracy: {ct_test.get('accuracy', 0):.4f}")
print(f"   - Test F1-Macro: {ct_test.get('f1_macro', 0):.4f}")

print(f"\n❌ THIẾU - Mô tả 2 views:")
print(f"   - Cần document: View 1 (features gì), View 2 (features gì)")
print(f"   - Giải thích tại sao 2 views độc lập")

if ct_history:
    total_m1 = sum(h.get('n_added_m1', 0) for h in ct_history)
    total_m2 = sum(h.get('n_added_m2', 0) for h in ct_history)
    print(f"\n⚠️  VẤN ĐỀ - Pseudo-label exchange:")
    print(f"   - M1→M2: {total_m1} samples")
    print(f"   - M2→M1: {total_m2} samples")
    if total_m1 == 0 and total_m2 == 0:
        print(f"   ❌ KHÔNG CÓ TRAO ĐỔI! Cần kiểm tra lại config")

print(f"\n⚠️  THIẾU - Biểu đồ diễn biến:")
print(f"   - Có history data: ✓")
print(f"   - Có visualization code: ✓")
print(f"   - Cần: Notebook hoặc script tạo biểu đồ")

print(f"\n✅ So sánh với baseline & self-training:")
ct_acc = ct_test.get('accuracy', 0)
ct_f1 = ct_test.get('f1_macro', 0)

print(f"   Baseline:      Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}")
print(f"   Self-Training: Acc={st_acc:.4f}, F1={st_f1:.4f}")
print(f"   Co-Training:   Acc={ct_acc:.4f}, F1={ct_f1:.4f}")

print("\n" + "="*80)
print("YÊU CẦU 3: SO SÁNH THAM SỐ")
print("="*80)

# Check for multiple tau configs
metrics_files = list(DATA_PROCESSED.glob("metrics*.json"))
print(f"\n❌ THIẾU - Thử nghiệm nhiều τ:")
print(f"   - Hiện có: {len(metrics_files)} config")
print(f"   - Cần: Ít nhất 3 giá trị τ khác nhau (0.8, 0.9, 0.95)")

print(f"\n⚠️  Các thử nghiệm khác (optional):")
print(f"   - Thay đổi label fraction: Có thể làm ✓")
print(f"   - Thử model khác: Chưa làm")
print(f"   - Thử view khác: Chưa làm")

print("\n" + "="*80)
print("YÊU CẦU 4: DASHBOARD STREAMLIT")
print("="*80)

dashboard_file = Path("app.py")
viz_utils = Path("src/visualization_utils.py")

print(f"\n✅ Dashboard Streamlit:")
print(f"   - File app.py: {'✓' if dashboard_file.exists() else '❌'}")
print(f"   - Visualization utils: {'✓' if viz_utils.exists() else '❌'}")
print(f"   - 5 pages: Overview, Comparison, Self-Training, Co-Training, Predictions")

if dashboard_file.exists():
    print(f"\n   Chạy: streamlit run app.py")

print("\n" + "="*80)
print("TỔNG KẾT")
print("="*80)

completed = []
partial = []
missing = []

completed.append("✅ Self-Training đã chạy (τ=0.9)")
completed.append("✅ Co-Training đã chạy (τ=0.9)")  
completed.append("✅ So sánh baseline vs semi-supervised")
completed.append("✅ Dashboard Streamlit đã có")
completed.append("✅ Visualization utilities đã có")

partial.append("⚠️  Self-Training: Có history nhưng thiếu biểu đồ")
partial.append("⚠️  Co-Training: Có history nhưng thiếu biểu đồ")
partial.append("⚠️  Co-Training: Pseudo-label exchange = 0")

missing.append("❌ Thử nghiệm nhiều τ (0.8, 0.85, 0.95)")
missing.append("❌ Biểu đồ diễn biến (plots/charts)")
missing.append("❌ Phân tích per-class performance")
missing.append("❌ Document 2 views cho Co-Training")

print(f"\n📊 HOÀN THÀNH ({len(completed)} items):")
for item in completed:
    print(f"   {item}")

print(f"\n⚠️  PARTIAL ({len(partial)} items):")
for item in partial:
    print(f"   {item}")

print(f"\n❌ CẦN BỔ SUNG ({len(missing)} items):")
for item in missing:
    print(f"   {item}")

progress = len(completed) / (len(completed) + len(partial) + len(missing)) * 100
print(f"\n{'='*80}")
print(f"TIẾN ĐỘ: {progress:.0f}% HOÀN THÀNH")
print(f"{'='*80}")
