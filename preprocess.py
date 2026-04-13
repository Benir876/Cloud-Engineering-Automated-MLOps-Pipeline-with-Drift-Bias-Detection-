import pandas as pd
import boto3
import io

# ── Config ───────────────────────────────────────────────────────────────
BUCKET = "sagemaker-us-east-1-350238019374"
PREFIX = "credit-default"
REGION = "us-east-1"

# ── Step 1: Read XLS from S3 ─────────────────────────────────────────────
print("Reading raw data from S3...")
s3 = boto3.client("s3", region_name=REGION)

obj = s3.get_object(
    Bucket=BUCKET,
    Key=f"{PREFIX}/raw-data/default of credit card clients.xls"
)
df = pd.read_excel(io.BytesIO(obj["Body"].read()), header=1)
print(f"Raw shape: {df.shape}")

# ── Step 2: Clean the data ───────────────────────────────────────────────
print("Cleaning data...")

# Drop the ID column — not useful for training
df = df.drop(columns=["ID"])

# Rename target column to something cleaner
df = df.rename(columns={"default payment next month": "target"})

# Drop any rows with missing values
df = df.dropna()

print(f"Cleaned shape: {df.shape}")
print(df.head())

# ── Step 3: Split into train and test ────────────────────────────────────
from sklearn.model_selection import train_test_split

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"\nTrain size: {train_df.shape}")
print(f"Test size:  {test_df.shape}")

# ── Step 4: Upload CSVs to S3 ────────────────────────────────────────────
print("\nUploading processed data to S3...")

def upload_df(df, key):
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    s3.put_object(Bucket=BUCKET, Key=key, Body=buffer.getvalue())
    print(f"✅ Uploaded: s3://{BUCKET}/{key}")

upload_df(train_df, f"{PREFIX}/processed/train.csv")
upload_df(test_df,  f"{PREFIX}/processed/test.csv")

print("\nPreprocessing complete!")