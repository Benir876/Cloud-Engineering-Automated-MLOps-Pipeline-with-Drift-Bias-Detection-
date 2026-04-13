import pandas as pd
import boto3
import io

# ── Config ───────────────────────────────────────────────────────────────
BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"
REGION = "us-east-1"

s3 = boto3.client("s3", region_name=REGION)

# ── Load test data ────────────────────────────────────────────────────────
print("Loading test data from S3...")
obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/processed/test.csv")
test_df = pd.read_csv(io.BytesIO(obj["Body"].read()))
print(f"Test shape: {test_df.shape}")

# ── Load model ────────────────────────────────────────────────────────────
import pickle
print("Loading model from S3...")
obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/model-output/model.pkl")
model = pickle.loads(obj["Body"].read())
print("✅ Model loaded!")

# ── Generate predictions ──────────────────────────────────────────────────
X_test = test_df.drop(columns=["target"])
y_test = test_df["target"]
test_df["prediction"] = model.predict(X_test)

# ── Bias Analysis by SEX ──────────────────────────────────────────────────
# SEX: 1 = Male, 2 = Female
print("\n── Bias Analysis by SEX ──────────────────────────────")
for sex, label in [(1, "Male"), (2, "Female")]:
    group = test_df[test_df["SEX"] == sex]
    actual_default_rate = group["target"].mean()
    predicted_default_rate = group["prediction"].mean()
    accuracy = (group["target"] == group["prediction"]).mean()
    print(f"\n{label} (n={len(group)}):")
    print(f"  Actual default rate:    {actual_default_rate:.4f}")
    print(f"  Predicted default rate: {predicted_default_rate:.4f}")
    print(f"  Accuracy:               {accuracy:.4f}")

# ── Bias Analysis by EDUCATION ────────────────────────────────────────────
print("\n── Bias Analysis by EDUCATION ────────────────────────")
edu_labels = {1: "Graduate", 2: "University", 3: "High School", 4: "Other"}
for edu, label in edu_labels.items():
    group = test_df[test_df["EDUCATION"] == edu]
    if len(group) == 0:
        continue
    actual_default_rate = group["target"].mean()
    predicted_default_rate = group["prediction"].mean()
    accuracy = (group["target"] == group["prediction"]).mean()
    print(f"\n{label} (n={len(group)}):")
    print(f"  Actual default rate:    {actual_default_rate:.4f}")
    print(f"  Predicted default rate: {predicted_default_rate:.4f}")
    print(f"  Accuracy:               {accuracy:.4f}")

# ── Save bias report to S3 ────────────────────────────────────────────────
print("\nSaving bias report to S3...")
report = test_df.groupby("SEX").apply(
    lambda g: pd.Series({
        "actual_default_rate": g["target"].mean(),
        "predicted_default_rate": g["prediction"].mean(),
        "accuracy": (g["target"] == g["prediction"]).mean(),
        "count": len(g)
    })
).reset_index()

buffer = io.StringIO()
report.to_csv(buffer, index=False)
s3.put_object(
    Bucket=BUCKET,
    Key=f"{PREFIX}/bias-report/bias_by_sex.csv",
    Body=buffer.getvalue()
)
print(f"✅ Bias report saved to S3!")