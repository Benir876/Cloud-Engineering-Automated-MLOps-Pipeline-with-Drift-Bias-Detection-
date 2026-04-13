import pandas as pd
import boto3
import io
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── Config ───────────────────────────────────────────────────────────────
BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"
REGION = "us-east-1"

s3 = boto3.client("s3", region_name=REGION)

# ── Step 1: Load train and test data from S3 ─────────────────────────────
print("Loading processed data from S3...")

def read_csv_from_s3(key):
    obj = s3.get_object(Bucket=BUCKET, Key=key)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))

train_df = read_csv_from_s3(f"{PREFIX}/processed/train.csv")
test_df  = read_csv_from_s3(f"{PREFIX}/processed/test.csv")

print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# ── Step 2: Split features and target ────────────────────────────────────
X_train = train_df.drop(columns=["target"])
y_train = train_df["target"]
X_test  = test_df.drop(columns=["target"])
y_test  = test_df["target"]

# ── Step 3: Train the model ──────────────────────────────────────────────
print("\nTraining model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("✅ Model trained!")

# ── Step 4: Evaluate ─────────────────────────────────────────────────────
print("\nEvaluating model...")
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# ── Step 5: Save and upload model to S3 ──────────────────────────────────
print("Saving model to S3...")
model_buffer = io.BytesIO()
pickle.dump(model, model_buffer)
model_buffer.seek(0)

s3.put_object(
    Bucket=BUCKET,
    Key=f"{PREFIX}/model-output/model.pkl",
    Body=model_buffer.getvalue()
)
print(f"✅ Model saved to: s3://{BUCKET}/{PREFIX}/model-output/model.pkl")