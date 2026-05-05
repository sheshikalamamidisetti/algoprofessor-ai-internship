"""
cloud_deploy.py  ·  Days 48-50  ·  Apr 24-26
----------------------------------------------
Goes into: day9_multiagent/  (extended)

AWS SageMaker + Azure ML deployment for InsightBot.
Packages the fine-tuned TimeSeriesHunter model (from day13/)
and deploys it as a scalable cloud endpoint.

Usage:
    python cloud_deploy.py --demo           # no cloud credentials needed
    python cloud_deploy.py --provider aws   # deploy to SageMaker
    python cloud_deploy.py --provider azure # deploy to Azure ML
    python cloud_deploy.py --status         # check deployment status
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


# ── Deployment Configuration ───────────────────────────────────────────────

DEPLOY_CONFIG = {
    "model_name":    "insightbot-timeserieshunter",
    "model_version": "1.0.0",
    "base_model":    "meta-llama/Llama-3.1-8B",
    "adapter_path":  "../day13/outputs/dpo_llama31/adapter",
    "quant_path":    "../day13/outputs/llama31_awq",

    "aws": {
        "region":          "us-east-1",
        "instance_type":   "ml.g5.2xlarge",    # A10G GPU, ~$1.5/hr
        "instance_count":  1,
        "endpoint_name":   "insightbot-endpoint",
        "s3_bucket":       os.getenv("AWS_S3_BUCKET", "insightbot-models"),
        "role_arn":        os.getenv("AWS_SAGEMAKER_ROLE", "arn:aws:iam::xxx:role/SageMakerRole"),
    },

    "azure": {
        "subscription_id":  os.getenv("AZURE_SUBSCRIPTION_ID", "xxx"),
        "resource_group":   "insightbot-rg",
        "workspace_name":   "insightbot-workspace",
        "compute_cluster":  "gpu-cluster",
        "vm_size":          "Standard_NC6s_v3",   # V100 GPU
        "endpoint_name":    "insightbot-endpoint",
    },
}


# ── AWS SageMaker Deployment ───────────────────────────────────────────────

def deploy_to_sagemaker(config: dict) -> dict:
    try:
        import boto3
        import sagemaker
        from sagemaker.huggingface import HuggingFaceModel

        aws = config["aws"]
        print(f"Deploying to AWS SageMaker...")
        print(f"  Region:        {aws['region']}")
        print(f"  Instance:      {aws['instance_type']}")
        print(f"  Endpoint name: {aws['endpoint_name']}")

        session = sagemaker.Session(
            boto_session=boto3.Session(region_name=aws["region"])
        )

        # Upload model to S3
        print(f"  Uploading model to s3://{aws['s3_bucket']}/...")
        model_uri = session.upload_data(
            path=config["quant_path"],
            bucket=aws["s3_bucket"],
            key_prefix="models/insightbot",
        )

        # Create SageMaker HuggingFace Model
        hf_model = HuggingFaceModel(
            model_data=model_uri,
            role=aws["role_arn"],
            transformers_version="4.37",
            pytorch_version="2.1",
            py_version="py310",
            env={
                "HF_MODEL_ID":          config["model_name"],
                "HF_TASK":              "text-generation",
                "SM_NUM_GPUS":          "1",
                "MAX_INPUT_LENGTH":     "2048",
                "MAX_TOTAL_TOKENS":     "4096",
                "MAX_BATCH_TOTAL_TOKENS": "8192",
            },
        )

        # Deploy
        predictor = hf_model.deploy(
            initial_instance_count=aws["instance_count"],
            instance_type=aws["instance_type"],
            endpoint_name=aws["endpoint_name"],
        )

        endpoint_url = f"https://runtime.sagemaker.{aws['region']}.amazonaws.com/endpoints/{aws['endpoint_name']}/invocations"

        return {
            "provider":     "aws_sagemaker",
            "status":       "deployed",
            "endpoint_url": endpoint_url,
            "endpoint_name": aws["endpoint_name"],
            "instance_type": aws["instance_type"],
            "cost_per_hour": "$1.50",
            "timestamp":    datetime.now().isoformat(),
        }

    except ImportError:
        print("AWS SDK not installed. Run: pip install boto3 sagemaker")
        return {"provider": "aws_sagemaker", "status": "sdk_not_installed"}
    except Exception as e:
        return {"provider": "aws_sagemaker", "status": "error", "error": str(e)}


# ── Azure ML Deployment ────────────────────────────────────────────────────

def deploy_to_azure(config: dict) -> dict:
    try:
        from azure.ai.ml import MLClient
        from azure.ai.ml.entities import (
            ManagedOnlineEndpoint,
            ManagedOnlineDeployment,
            Model, Environment,
        )
        from azure.identity import DefaultAzureCredential

        azure = config["azure"]
        print(f"Deploying to Azure ML...")
        print(f"  Workspace:     {azure['workspace_name']}")
        print(f"  Compute:       {azure['vm_size']}")

        client = MLClient(
            DefaultAzureCredential(),
            subscription_id=azure["subscription_id"],
            resource_group_name=azure["resource_group"],
            workspace_name=azure["workspace_name"],
        )

        # Register model
        model = client.models.create_or_update(Model(
            name=config["model_name"],
            version=config["model_version"],
            path=config["quant_path"],
            description="InsightBot TimeSeriesHunter QLoRA Llama 3.1",
        ))

        # Create endpoint
        endpoint = ManagedOnlineEndpoint(
            name=azure["endpoint_name"],
            description="InsightBot real-time inference endpoint",
            auth_mode="key",
        )
        client.online_endpoints.begin_create_or_update(endpoint).wait()

        # Create deployment
        deployment = ManagedOnlineDeployment(
            name="blue",
            endpoint_name=azure["endpoint_name"],
            model=model,
            environment=Environment(
                image="mcr.microsoft.com/azureml/pytorch-1.13-ubuntu20.04-py38-cuda11.6-gpu",
                conda_file="environment.yml",
            ),
            instance_type=azure["vm_size"],
            instance_count=1,
        )
        client.online_deployments.begin_create_or_update(deployment).wait()

        endpoint_url = f"https://{azure['endpoint_name']}.{azure['resource_group']}.inference.ml.azure.com/score"

        return {
            "provider":      "azure_ml",
            "status":        "deployed",
            "endpoint_url":  endpoint_url,
            "endpoint_name": azure["endpoint_name"],
            "vm_size":       azure["vm_size"],
            "cost_per_hour": "$2.10",
            "timestamp":     datetime.now().isoformat(),
        }

    except ImportError:
        print("Azure ML SDK not installed. Run: pip install azure-ai-ml azure-identity")
        return {"provider": "azure_ml", "status": "sdk_not_installed"}
    except Exception as e:
        return {"provider": "azure_ml", "status": "error", "error": str(e)}


# ── Demo Mode ──────────────────────────────────────────────────────────────

def run_demo():
    print("=" * 60)
    print("Days 48-50 — Cloud Deployment Demo")
    print("=" * 60)
    print()

    print("Deployment pipeline:")
    steps = [
        ("1. Package model",   "AWQ quantised Llama 3.1 + LoRA adapter"),
        ("2. Push to registry","S3 (AWS) or Azure ML Model Registry"),
        ("3. Create endpoint", "SageMaker ml.g5.2xlarge or Azure NC6s_v3"),
        ("4. Deploy",          "Auto-scaling, health checks, monitoring"),
        ("5. Test endpoint",   "curl POST with DS query, get streaming response"),
    ]
    for step, detail in steps:
        print(f"  {step:<22} {detail}")

    print()
    print("AWS SageMaker config:")
    aws = DEPLOY_CONFIG["aws"]
    for k, v in aws.items():
        if "arn" not in k and "bucket" not in k:
            print(f"  {k:<20} {v}")

    print()
    print("Azure ML config:")
    for k, v in DEPLOY_CONFIG["azure"].items():
        if "subscription" not in k:
            print(f"  {k:<20} {v}")

    print()
    print("Cost comparison:")
    costs = [
        ("AWS ml.g5.2xlarge",   "A10G 24GB GPU",  "$1.50/hr", "Best for vLLM throughput"),
        ("Azure NC6s_v3",       "V100 16GB GPU",  "$2.10/hr", "Good availability"),
        ("AWS ml.g4dn.xlarge",  "T4 16GB GPU",    "$0.75/hr", "Cost-optimised option"),
        ("Hugging Face Spaces", "T4 or A10G",     "$0.60/hr", "Simplest deployment"),
    ]
    print(f"  {'Instance':<25} {'GPU':<18} {'Cost':<12} Notes")
    print("  " + "-" * 70)
    for instance, gpu, cost, note in costs:
        print(f"  {instance:<25} {gpu:<18} {cost:<12} {note}")

    # Save demo results
    Path("outputs").mkdir(exist_ok=True)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    demo = {
        "deployment_config": DEPLOY_CONFIG,
        "demo_mode": True,
        "timestamp": ts,
        "note": "Run with --provider aws or --provider azure for real deployment",
    }
    path = f"outputs/cloud_deploy_plan_{ts}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2, default=str)
    print(f"\nDeployment plan saved: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Days 48-50 — Cloud Deploy")
    parser.add_argument("--demo",     action="store_true", default=True)
    parser.add_argument("--provider", choices=["aws", "azure"], default=None)
    parser.add_argument("--status",   action="store_true")
    args = parser.parse_args()

    Path("outputs").mkdir(exist_ok=True)

    if args.provider == "aws":
        result = deploy_to_sagemaker(DEPLOY_CONFIG)
        print(json.dumps(result, indent=2))
    elif args.provider == "azure":
        result = deploy_to_azure(DEPLOY_CONFIG)
        print(json.dumps(result, indent=2))
    else:
        run_demo()
