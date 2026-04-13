import boto3
import sagemaker
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
ROLE = "arn:aws:iam::350238019374:role/service-role/AmazonSageMaker-ExecutionRole-20260413T134223"
REGION = "us-east-1"
BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"

RAW_DATA_S3 = f"s3://{BUCKET}/{PREFIX}/raw-data/"
PROCESSED_DATA_S3 = f"s3://{BUCKET}/{PREFIX}/processed/"
MODEL_OUTPUT_S3 = f"s3://{BUCKET}/{PREFIX}/model-output/"

# ── Step 1: Read the XLS file ────────────────────────────────────────────
print("Reading dataset...")
df = pd.read_excel(r"D:\Classes\Winter_2026\COMP264 (CLOUD MACHINE)\GroupProject\default+of+credit+card+clients\default of credit card clients.xls", header=1)
print(f"Dataset shape: {df.shape}")
print(df.head())

# ── Step 2: Upload raw file to S3 ────────────────────────────────────────
print("\nUploading to S3...")
s3 = boto3.client("s3", region_name=REGION)
s3.upload_file(
    Filename=r"D:\Classes\Winter_2026\COMP264 (CLOUD MACHINE)\GroupProject\default+of+credit+card+clients\default of credit card clients.xls",
    Bucket=BUCKET,
    Key=f"{PREFIX}/raw-data/default of credit card clients.xls"
)
print(f"✅ Upload complete! File is at: {RAW_DATA_S3}")