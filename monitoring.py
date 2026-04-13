import pandas as pd
import boto3
import io
import json
import numpy as np

# ── Config ───────────────────────────────────────────────────────────────
BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"
REGION = "us-east-1"

s3 = boto3.client("s3", region_name=REGION)

# ── Load training data to build baseline statistics ───────────────────────
print("Loading training data to build baseline...")
obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/processed/train.csv")
train_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
print(f"Train shape: {train_df.shape}")

# ── Build baseline statistics ─────────────────────────────────────────────
print("\nBuilding baseline statistics...")
baseline_stats = {}
for col in train_df.columns:
    baseline_stats[col] = {
        "mean": float(train_df[col].mean()),
        "std":  float(train_df[col].std()),
        "min":  float(train_df[col].min()),
        "max":  float(train_df[col].max()),
        "null_count": int(train_df[col].isnull().sum())
    }

# ── Save baseline to S3 ───────────────────────────────────────────────────
s3.put_object(
    Bucket=BUCKET,
    Key=f"{PREFIX}/monitoring/baseline_stats.json",
    Body=json.dumps(baseline_stats, indent=2)
)
print("✅ Baseline saved to S3!")

# ── Simulate incoming new data with drift ─────────────────────────────────
print("\nSimulating new incoming data with drift...")
new_data = train_df.copy()

# Inject drift — shift LIMIT_BAL and AGE distributions
new_data["LIMIT_BAL"] = new_data["LIMIT_BAL"] * 1.4
new_data["AGE"] = new_data["AGE"] + 8

# ── Detect drift by comparing means ──────────────────────────────────────
print("\n── Drift Detection Report ────────────────────────────")
drift_report = []
DRIFT_THRESHOLD = 0.15  # 15% change flags as drift

for col in train_df.columns:
    baseline_mean = baseline_stats[col]["mean"]
    new_mean = float(new_data[col].mean())
    if baseline_mean == 0:
        continue
    pct_change = abs((new_mean - baseline_mean) / baseline_mean)
    drifted = pct_change > DRIFT_THRESHOLD
    status = "🚨 DRIFT DETECTED" if drifted else "✅ OK"
    drift_report.append({
        "feature": col,
        "baseline_mean": round(baseline_mean, 4),
        "new_mean": round(new_mean, 4),
        "pct_change": round(pct_change, 4),
        "drift_detected": drifted
    })
    if drifted:
        print(f"{status} | {col}: {baseline_mean:.2f} → {new_mean:.2f} ({pct_change*100:.1f}% change)")

# ── Save drift report to S3 ───────────────────────────────────────────────
report_df = pd.DataFrame(drift_report)
buffer = io.StringIO()
report_df.to_csv(buffer, index=False)
s3.put_object(
    Bucket=BUCKET,
    Key=f"{PREFIX}/monitoring/drift_report.csv",
    Body=buffer.getvalue()
)
print("\n✅ Drift report saved to S3!")
drifted_count = report_df["drift_detected"].sum()
print(f"\nSummary: {drifted_count} out of {len(drift_report)} features show drift")