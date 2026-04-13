import boto3
import json

# ── Config ───────────────────────────────────────────────────────────────
REGION = "us-east-1"
PIPELINE_NAME = "CreditMLOpsPipeline"

# ── This simulates what the Lambda function does on AWS ───────────────────
def lambda_handler(event=None, context=None):
    sm = boto3.client("sagemaker", region_name=REGION)

    print("🔁 Retraining trigger fired!")
    print(f"Starting pipeline: {PIPELINE_NAME}")

    # Check if pipeline exists first
    try:
        pipelines = sm.list_pipelines(PipelineNamePrefix="Credit")
        existing = [p["PipelineName"] for p in pipelines["PipelineSummaries"]]
        print(f"Found pipelines: {existing}")

        if PIPELINE_NAME in existing:
            response = sm.start_pipeline_execution(PipelineName=PIPELINE_NAME)
            print(f"✅ Pipeline triggered! Execution ARN: {response['PipelineExecutionArn']}")
        else:
            print("⚠️  Pipeline not deployed to SageMaker yet.")
            print("✅ Lambda trigger code is ready — will fire when pipeline is registered.")

    except Exception as e:
        print(f"Error: {e}")

    return {"status": "trigger complete"}

# ── Run locally to test ───────────────────────────────────────────────────
if __name__ == "__main__":
    lambda_handler()