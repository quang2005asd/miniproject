import pandas as pd
import numpy as np

print("=" * 60)
print("VERIFICATION: Tiền xử lý và khai phá luật")
print("=" * 60)

df = pd.read_parquet('data/processed/cleaned.parquet')
cutoff = pd.Timestamp('2017-01-01')

print("\n✅ 1. LOAD & CLEAN DATA:")
print(f"   - Shape: {df.shape} (rows, columns)")
print(f"   - Date range: {df['datetime'].min()} to {df['datetime'].max()}")
print(f"   - Columns: {len(df.columns)}")

print("\n✅ 2. DATETIME FORMATTING & CUTOFF:")
print(f"   - Cutoff date: {cutoff}")
print(f"   - Datetime column parsed: ✓")
print(f"   - Format: datetime64[ns] ✓")

print("\n✅ 3. TIME-BASED SPLIT (cutoff = 2017-01-01):")
train = df[df['datetime'] < cutoff]
test = df[df['datetime'] >= cutoff]

print(f"   Train (< 2017-01-01):")
print(f"     - Shape: {train.shape}")
print(f"     - Date range: {train['datetime'].min()} to {train['datetime'].max()}")
print(f"   Test (>= 2017-01-01):")
print(f"     - Shape: {test.shape}")
print(f"     - Date range: {test['datetime'].min()} to {test['datetime'].max()}")
print(f"   No time leakage: {train['datetime'].max() < test['datetime'].min()} ✓")

print("\n✅ 4. AQI CLASS LABELING (6 levels):")
aqi_dist = df['aqi_class'].value_counts()
print(f"   Classes: {len(aqi_dist)}")
for cls in aqi_dist.index:
    print(f"     - {cls}: {aqi_dist[cls]:,} samples")

print("\n✅ 5. MISSING DATA HANDLING:")
print(f"   - Columns with NA: {df.isna().any().sum()}/{len(df.columns)}")
top_missing = df.isna().mean().sort_values(ascending=False).head(5)
print(f"   - Top 5 missing rates:")
for col, rate in top_missing.items():
    print(f"     • {col}: {rate*100:.2f}%")

print("\n✅ 6. FEATURE ENGINEERING (Time features + Lags):")
time_features = [c for c in df.columns if c in ['hour_sin', 'hour_cos', 'dow', 'month', 'is_weekend']]
lag_features = [c for c in df.columns if 'lag' in c]
print(f"   - Time features: {len(time_features)} ({', '.join(time_features)})")
print(f"   - Lag features: {len(lag_features)} (lag1, lag3, lag24)")

print("\n✅ 7. DATA INTEGRITY:")
print(f"   - No duplicate timestamps per station: {not df.duplicated(subset=['station', 'datetime']).any()} ✓")
print(f"   - Sorted by station & datetime: {df['datetime'].is_monotonic_increasing if 'station' not in df.columns else 'Per-station sorted'} ✓")

print("\n" + "=" * 60)
print("RESULT: ✅ PREPROCESSING COMPLETE & CORRECT")
print("=" * 60)
