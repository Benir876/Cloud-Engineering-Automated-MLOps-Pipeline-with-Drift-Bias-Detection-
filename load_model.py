import boto3
import joblib
import json
import io

# ── Their S3 bucket ───────────────────────────────────────────────────────
BUCKET = "cloud-machine-dataset-429293899944-us-east-1-an"
REGION = "us-east-1"

# ── Use teammate profile ──────────────────────────────────────────────────
session = boto3.Session(profile_name="teammate", region_name=REGION)
s3 = session.client("s3")

# ── Step 1: Load the model config ─────────────────────────────────────────
print("Loading model config...")
obj = s3.get_object(
    Bucket=BUCKET,
    Key="ml_models/xgboost_model_config.json"
)
config = json.loads(obj["Body"].read())
print("✅ Config loaded!")
print(json.dumps(config, indent=2))

# ── Step 2: Load the XGBoost model ────────────────────────────────────────
print("\nLoading XGBoost model...")
obj = s3.get_object(
    Bucket=BUCKET,
    Key="ml_models/xgboost_final_model.joblib"
)
model = joblib.load(io.BytesIO(obj["Body"].read()))
print("✅ Model loaded!")
print(f"Model type: {type(model)}")