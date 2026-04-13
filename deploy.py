import boto3
import pickle
import io
import json
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────
BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"
REGION = "us-east-1"
ENDPOINT_NAME = "credit-default-endpoint"

s3 = boto3.client("s3", region_name=REGION)

# ── Step 1: Load model from S3 ────────────────────────────────────────────
print("Loading model from S3...")
obj = s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/model-output/model.pkl")
model = pickle.loads(obj["Body"].read())
print("✅ Model loaded!")

# ── Step 2: Test the model locally with a sample prediction ──────────────
print("\nTesting model with a sample applicant...")

# Sample credit card applicant (24 features, no target column)
sample = pd.DataFrame([{
    "LIMIT_BAL": 50000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1,
    "AGE": 35, "PAY_0": 0, "PAY_2": 0, "PAY_3": 0, "PAY_4": 0,
    "PAY_5": 0, "PAY_6": 0, "BILL_AMT1": 10000, "BILL_AMT2": 9500,
    "BILL_AMT3": 9000, "BILL_AMT4": 8500, "BILL_AMT5": 8000,
    "BILL_AMT6": 7500, "PAY_AMT1": 1000, "PAY_AMT2": 1000,
    "PAY_AMT3": 1000, "PAY_AMT4": 1000, "PAY_AMT5": 1000,
    "PAY_AMT6": 1000
}])

prediction = model.predict(sample)
probability = model.predict_proba(sample)

print(f"Prediction:  {'⚠️  Will Default' if prediction[0] == 1 else '✅ Will NOT Default'}")
print(f"Confidence:  {max(probability[0]) * 100:.1f}%")

# ── Step 3: Save prediction function to S3 as deployment config ───────────
print("\nSaving deployment config to S3...")
deploy_config = {
    "endpoint_name": ENDPOINT_NAME,
    "model_s3_path": f"s3://{BUCKET}/{PREFIX}/model-output/model.pkl",
    "instance_type": "ml.m5.large",
    "region": REGION,
    "features": list(sample.columns),
    "target": "default payment next month",
    "model_accuracy": 0.8160
}

s3.put_object(
    Bucket=BUCKET,
    Key=f"{PREFIX}/deployment/deploy_config.json",
    Body=json.dumps(deploy_config, indent=2)
)
print("✅ Deployment config saved to S3!")
print(f"\nEndpoint name ready: {ENDPOINT_NAME}")
print("✅ Model is ready for deployment!")