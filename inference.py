import boto3
import joblib
import json
import io
import pandas as pd

# ── Config ───────────────────────────────────────────────────────────────
BUCKET = "cloud-machine-dataset-429293899944-us-east-1-an"
REGION = "us-east-1"
THRESHOLD = 0.5

session = boto3.Session(profile_name="teammate", region_name=REGION)
s3 = session.client("s3")

# ── Load model ────────────────────────────────────────────────────────────
print("Loading model...")
obj = s3.get_object(Bucket=BUCKET, Key="ml_models/xgboost_final_model.joblib")
model = joblib.load(io.BytesIO(obj["Body"].read()))
print("✅ Model loaded!")

# ── Sample applicant (29 features matching model config) ──────────────────
sample = pd.DataFrame([{
    "LIMIT_BAL": 50000, "AGE": 35,
    "PAY_0": 0, "PAY_2": 0, "PAY_3": 0,
    "PAY_4": 0, "PAY_5": 0, "PAY_6": 0,
    "BILL_AMT1": 10000, "BILL_AMT2": 9500, "BILL_AMT3": 9000,
    "BILL_AMT4": 8500, "BILL_AMT5": 8000, "BILL_AMT6": 7500,
    "PAY_AMT1": 1000, "PAY_AMT2": 1000, "PAY_AMT3": 1000,
    "PAY_AMT4": 1000, "PAY_AMT5": 1000, "PAY_AMT6": 1000,
    "SEX_male": 0, "SEX_female": 1,
    "EDU_grad_school": 0, "EDU_university": 1,
    "EDU_high_school": 0, "EDU_others": 0,
    "MARRIED": 1, "SINGLE": 0, "MARRIAGE_others": 0
}])

# ── Run prediction ────────────────────────────────────────────────────────
print("\nRunning inference...")
probability = model.predict_proba(sample)[0][1]
prediction = 1 if probability >= THRESHOLD else 0

print(f"\n── Prediction Result ─────────────────────────────────")
print(f"Default Probability: {probability:.4f} ({probability*100:.1f}%)")
print(f"Threshold:           {THRESHOLD}")
print(f"Prediction:          {'⚠️  Will Default' if prediction == 1 else '✅ Will NOT Default'}")

# ── Save result to your S3 bucket ────────────────────────────────────────
print("\nSaving inference result to S3...")
YOUR_BUCKET = "sagemaker-us-east-1-350238019374"
your_s3 = boto3.client("s3", region_name=REGION)

result = {
    "prediction": int(prediction),
    "probability": round(float(probability), 4),
    "threshold": THRESHOLD,
    "result": "Will Default" if prediction == 1 else "Will NOT Default"
}

your_s3.put_object(
    Bucket=YOUR_BUCKET,
    Key="credit-default/inference/sample_prediction.json",
    Body=json.dumps(result, indent=2)
)
print("✅ Result saved to your S3!")