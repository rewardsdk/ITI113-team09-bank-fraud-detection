#!/usr/bin/env python3
"""
CLI entrypoint used by GitHub Actions (and runnable locally) to build,
upsert, execute, and optionally wait on one of the project's two SageMaker
Pipelines.

Usage:
    python scripts/trigger_pipeline.py --pipeline core \
        --n-estimators 500 --max-depth 8 --quality-gate 0.60 --wait

    python scripts/trigger_pipeline.py --pipeline triggered \
        --input-data-url s3://nyp-26s1-iti113/iti113/team09/data/bank-fraud-detection/raw/bank_fraud.csv \
        --n-estimators 500 --max-depth 8 --quality-gate 0.70 --wait

Required environment variables (set as GitHub Actions secrets / vars,
or exported locally):
    SAGEMAKER_EXECUTION_ROLE_ARN   IAM role the pipeline steps run as
    AWS_REGION                     defaults to ap-southeast-1
    SM_BUCKET                      defaults to nyp-26s1-iti113
    TEAM_ID, STUDENT_ID, SEMESTER  default to team09 / s901 / 26S1

AWS credentials themselves are expected to already be configured in the
environment (e.g. by aws-actions/configure-aws-credentials in the calling
workflow) — this script does not read or accept raw keys directly.

Exit codes:
    0  pipeline execution reached "Succeeded"
    1  pipeline execution reached "Failed" or "Stopped"
    2  usage / configuration error
"""
import argparse
import sys
import time


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pipeline", choices=["core", "triggered"], required=True,
                    help="Which pipeline to build/upsert/execute.")
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--quality-gate", type=float, default=None,
                    help="Overrides the pipeline's default QualityGateAUC parameter.")
    p.add_argument("--input-data-url", type=str, default=None,
                    help="Only used by --pipeline triggered; overrides InputDataUrl.")
    p.add_argument("--wait", action="store_true",
                    help="Poll the execution until it finishes, and exit non-zero on failure.")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--timeout-minutes", type=int, default=45)
    return p.parse_args()


def poll_execution(execution, poll_seconds: int, timeout_minutes: int) -> str:
    deadline = time.time() + timeout_minutes * 60
    last_status_by_step = {}

    while time.time() < deadline:
        desc = execution.describe()
        status = desc["PipelineExecutionStatus"]

        steps = execution.list_steps()
        for step in steps:
            name = step["StepName"]
            step_status = step["StepStatus"]
            if last_status_by_step.get(name) != step_status:
                print(f"  [{name}] -> {step_status}")
                last_status_by_step[name] = step_status

        if status in ("Succeeded", "Failed", "Stopped"):
            print(f"\nExecution finished with status: {status}")
            return status

        time.sleep(poll_seconds)

    print(f"\nTimed out after {timeout_minutes} minutes while status was still in progress.")
    return "Timeout"


def print_failed_step_logs(execution, region: str):
    """Best-effort dump of CloudWatch logs for any failed step, so a failed
    GitHub Actions run shows the actual SageMaker error inline in the job log
    instead of only "Failed"."""
    import boto3

    logs_client = boto3.client("logs", region_name=region)
    for step in execution.list_steps():
        if step["StepStatus"] != "Failed":
            continue
        step_name = step["StepName"]
        print(f"\n----- Logs for failed step: {step_name} -----")
        log_group = "/aws/sagemaker/ProcessingJobs" if "Process" in step_name or "Evaluate" in step_name \
            else "/aws/sagemaker/TrainingJobs"
        try:
            streams = logs_client.describe_log_streams(
                logGroupName=log_group, orderBy="LastEventTime", descending=True, limit=5
            )
            for s in streams.get("logStreams", []):
                events = logs_client.get_log_events(
                    logGroupName=log_group, logStreamName=s["logStreamName"], limit=50
                )
                for e in events.get("events", []):
                    print(e["message"].rstrip())
        except Exception as e:  # noqa: BLE001 - best-effort diagnostics only
            print(f"  (could not fetch logs automatically: {e})")


def main():
    args = parse_args()

    if args.pipeline == "core":
        from pipeline import core_pipeline as mod
        params = {"NEstimators": args.n_estimators, "MaxDepth": args.max_depth}
        if args.quality_gate is not None:
            params["QualityGateAUC"] = args.quality_gate
        mod.upload_source_scripts()
        pipeline = mod.build_pipeline()
        role_arn = mod.ROLE_ARN
        region = mod.REGION

    else:  # triggered
        from pipeline import triggered_pipeline as mod
        params = {"NEstimators": args.n_estimators, "MaxDepth": args.max_depth}
        if args.quality_gate is not None:
            params["QualityGateAUC"] = args.quality_gate
        if args.input_data_url is not None:
            params["InputDataUrl"] = args.input_data_url
        mod.ensure_model_package_group()
        pipeline = mod.build_pipeline()
        role_arn = mod.ROLE_ARN
        region = mod.REGION

    print(f'Upserting pipeline "{pipeline.name}"...')
    pipeline.upsert(role_arn=role_arn)
    print("Upsert complete.")

    print(f"Starting execution with parameters: {params}")
    execution = pipeline.start(parameters=params)
    print(f"Execution ARN: {execution.arn}")

    if not args.wait:
        print("Not waiting for completion (--wait not set). Exiting 0.")
        return 0

    status = poll_execution(execution, args.poll_seconds, args.timeout_minutes)

    if status == "Succeeded":
        return 0
    if status in ("Failed", "Stopped"):
        print_failed_step_logs(execution, region)
        return 1
    # Timeout or unknown status
    return 1


if __name__ == "__main__":
    sys.exit(main())
