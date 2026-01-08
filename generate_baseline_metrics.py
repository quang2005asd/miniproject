import pandas as pd
import json
from pathlib import Path
from src.classification_library import time_split, train_classifier

print("=" * 60)
print("GENERATING BASELINE METRICS")
print("=" * 60)

# Load dataset
dataset_path = Path('data/processed/dataset_for_clf.parquet')
df = pd.read_parquet(dataset_path)
print(f"\n✓ Loaded dataset: {df.shape}")

# Split
cutoff = '2017-01-01'
train_df, test_df = time_split(df, cutoff=cutoff)
print(f"✓ Train: {train_df.shape}, Test: {test_df.shape}")

# Train classifier
print("\n🚀 Training classifier...")
out = train_classifier(train_df, test_df, target_col='aqi_class')
metrics = out['metrics']
pred_df = out['pred_df']

print(f"\n📊 RESULTS:")
print(f"   Accuracy: {metrics['accuracy']:.4f}")
print(f"   F1-macro: {metrics['f1_macro']:.4f}")

# Save metrics
metrics_path = Path('data/processed/metrics.json')
metrics_data = {
    'method': 'supervised_baseline',
    'model': 'HistGradientBoostingClassifier',
    'cutoff': cutoff,
    'test_accuracy': metrics['accuracy'],
    'test_f1_macro': metrics['f1_macro'],
    'test_confusion_matrix': metrics['confusion_matrix'],
    'test_classification_report': metrics['report'],
    'labels': metrics['labels']
}

with open(metrics_path, 'w', encoding='utf-8') as f:
    json.dump(metrics_data, f, indent=2, ensure_ascii=False)

print(f"\n✅ Saved: {metrics_path}")

# Save predictions sample
pred_path = Path('data/processed/predictions_sample.csv')
pred_df.head(5000).to_csv(pred_path, index=False)
print(f"✅ Saved: {pred_path}")

print("\n" + "=" * 60)
print("BASELINE METRICS COMPLETE!")
print("=" * 60)
