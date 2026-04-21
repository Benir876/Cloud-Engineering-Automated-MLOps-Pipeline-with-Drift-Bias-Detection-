import boto3
import json
import io
import tarfile
import joblib
import os

# ── Config ───────────────────────────────────────────────────────────────
REGION = "us-east-1"
TEAM_BUCKET = "cloud-machine-dataset-429293899944-us-east-1-an"
YOUR_BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"
ROLE = "arn:aws:iam::350238019374:role/service-role/AmazonSageMaker-ExecutionRole-20260413T134223"
ENDPOINT_NAME = "credit-default-xgboost-endpoint"

# ── Step 1: Download model from teammate's S3 ─────────────────────────────
print("Downloading model from teammate's S3...")
team_session = boto3.Session(profile_name="teammate", region_name=REGION)
team_s3 = team_session.client("s3")

obj = team_s3.get_object(Bucket=TEAM_BUCKET, Key="ml_models/xgboost_final_model.joblib")
model_bytes = obj["Body"].read()
print("✅ Model downloaded!")

# ── Step 2: Repackage model as model.tar.gz for SageMaker ─────────────────
print("\nPackaging model for SageMaker...")
with open("xgboost_final_model.joblib", "wb") as f:
    f.write(model_bytes)

with tarfile.open("model.tar.gz", "w:gz") as tar:
    tar.add("xgboost_final_model.joblib")
print("✅ Model packaged!")

# ── Step 3: Upload model.tar.gz to your S3 ───────────────────────────────
print("\nUploading model to your S3...")
your_s3 = boto3.client("s3", region_name=REGION)
your_s3.upload_file(
    "model.tar.gz",
    YOUR_BUCKET,
    f"{PREFIX}/model-output/model.tar.gz"
)
MODEL_S3_URI = f"s3://{YOUR_BUCKET}/{PREFIX}/model-output/model.tar.gz"
print(f"✅ Model uploaded to: {MODEL_S3_URI}")

# ── Step 4: Deploy to SageMaker endpoint ─────────────────────────────────
print("\nDeploying to SageMaker endpoint...")
sm = boto3.client("sagemaker", region_name=REGION)

# Create model
print("Creating SageMaker model...")
sm.create_model(
    ModelName="credit-xgboost-model",
    PrimaryContainer={
        "Image": "683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.2-1-cpu-py3",
        "ModelDataUrl": MODEL_S3_URI,
        "Environment": {
            "SAGEMAKER_PROGRAM": "inference_handler.py"
        }
    },
    ExecutionRoleArn=ROLE
)
print("✅ Model created!")

# Create endpoint config
print("Creating endpoint config...")
sm.create_endpoint_config(
    EndpointConfigName="credit-xgboost-config",
    ProductionVariants=[{
        "VariantName": "default",
        "ModelName": "credit-xgboost-model",
        "InitialInstanceCount": 1,
        "InstanceType": "ml.m5.large"
    }]
)
print("✅ Endpoint config created!")

# Create endpoint
print("Creating endpoint (this takes 5-10 mins)...")
sm.create_endpoint(
    EndpointName=ENDPOINT_NAME,
    EndpointConfigName="credit-xgboost-config"
)
print(f"✅ Endpoint deployment started!")
print(f"Endpoint name: {ENDPOINT_NAME}")
print("\nCheck status in AWS Console → SageMaker → Endpoints")

# ── Save deployment info ──────────────────────────────────────────────────
deployment_info = {
    "endpoint_name": ENDPOINT_NAME,
    "model_s3_uri": MODEL_S3_URI,
    "status": "deploying",
    "instance_type": "ml.m5.large"
}
your_s3.put_object(
    Bucket=YOUR_BUCKET,
    Key=f"{PREFIX}/deployment/endpoint_info.json",
    Body=json.dumps(deployment_info, indent=2)
)
print("✅ Deployment info saved to S3!")

# ── Cleanup local files ───────────────────────────────────────────────────
os.remove("xgboost_final_model.joblib")
os.remove("model.tar.gz")