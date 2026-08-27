"""
Enhanced / S3-triggerable SageMaker Pipeline definition
(5-step: Preprocess -> Train -> Evaluate -> QualityGate -> Register).

Faithful extraction of the pipeline-definition cells from
notebooks/05_fraud_model_b_shap.ipynb, refactored into an importable
function. This is the pipeline the GitHub Actions workflow
.github/workflows/trigger-triggered-pipeline.yml calls — effectively
giving the project a working, git-triggered analogue of the
EventBridge/Lambda S3-upload trigger that was designed but not deployed
as standing AWS infrastructure (see Final Report, Section 8.1).
"""
import os

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import json

import boto3
import botocore
import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# ----------------------------------------------------------------------
# Configuration (mirrors Notebook 05, cell 2)
# ----------------------------------------------------------------------
REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
SEMESTER = os.environ.get("SEMESTER", "26S1")
TEAM_ID = os.environ.get("TEAM_ID", "team09")
STUDENT_ID = os.environ.get("STUDENT_ID", "s901")
BUCKET = os.environ.get("SM_BUCKET", "nyp-26s1-iti113")
ROLE_ARN = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]  # required, no safe default

TRIGGERED_PIPELINE_NAME = f"iti113-{TEAM_ID}-bank-fraud-detection-triggered"
TEAM_PREFIX = f"iti113/{TEAM_ID}/triggered-pipeline"
MODEL_PACKAGE_GROUP_NAME = f"{TEAM_ID}-BankFraudDetection-Triggered"

PROCESSING_INSTANCE_TYPE = os.environ.get("PROCESSING_INSTANCE_TYPE", "ml.m5.large")
TRAINING_INSTANCE_TYPE = os.environ.get("TRAINING_INSTANCE_TYPE", "ml.m5.large")

DEFAULT_INPUT_DATA_URL = os.environ.get(
    "DEFAULT_INPUT_DATA_URL",
    f"s3://{BUCKET}/iti113/{TEAM_ID}/data/bank-fraud-detection/raw/bank_fraud.csv",
)
TRAINING_OUTPUT_PREFIX = f"s3://{BUCKET}/{TEAM_PREFIX}/training-output"
TRAINING_CODE_LOCATION = f"s3://{BUCKET}/{TEAM_PREFIX}/code/training"

LOCAL_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "triggered")


def ensure_model_package_group(sm_client=None):
    """Idempotently creates the Model Package Group, exactly as Notebook 05 cell 9 does."""
    sm_client = sm_client or boto3.client("sagemaker", region_name=REGION)
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP_NAME,
            ModelPackageGroupDescription=(
                f"Model package group for {TEAM_ID} S3-triggered Bank Fraud Detection pipeline"
            ),
        )
        print("Created model package group:", MODEL_PACKAGE_GROUP_NAME)
    except botocore.exceptions.ClientError as e:
        error_code = e.response.get("Error", {}).get("Code", "")
        error_message = e.response.get("Error", {}).get("Message", "")
        if "already exists" in error_message.lower() or error_code == "ValidationException":
            print("Model package group already exists:", MODEL_PACKAGE_GROUP_NAME)
        else:
            raise


def build_pipeline() -> Pipeline:
    """Builds the 5-step enhanced pipeline, matching Notebook 05 cell 10 exactly:
      PreprocessData -> TrainModel -> EvaluateModel -> CheckAUCQualityGate -> RegisterModel
    """
    boto_session = boto3.Session(region_name=REGION)
    pipeline_session = PipelineSession(
        boto_session=boto_session,
        sagemaker_client=boto_session.client("sagemaker"),
        default_bucket=BUCKET,
        default_bucket_prefix=TEAM_PREFIX,
    )

    input_data_url_param = ParameterString(name="InputDataUrl", default_value=DEFAULT_INPUT_DATA_URL)
    n_estimators_param = ParameterInteger(name="NEstimators", default_value=500)
    max_depth_param = ParameterInteger(name="MaxDepth", default_value=8)
    quality_gate_auc_param = ParameterFloat(name="QualityGateAUC", default_value=0.70)

    processor = SKLearnProcessor(
        framework_version="1.4-2",
        role=ROLE_ARN,
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        base_job_name=f"iti113-{TEAM_ID}-process",
        sagemaker_session=pipeline_session,
    )

    estimator = SKLearn(
        entry_point="train.py",
        source_dir=LOCAL_SRC_DIR,
        framework_version="1.4-2",
        py_version="py3",
        role=ROLE_ARN,
        instance_type=TRAINING_INSTANCE_TYPE,
        instance_count=1,
        base_job_name=f"iti113-{TEAM_ID}-train",
        sagemaker_session=pipeline_session,
        code_location=TRAINING_CODE_LOCATION,
        output_path=TRAINING_OUTPUT_PREFIX,
        hyperparameters={
            "n-estimators": n_estimators_param,
            "max-depth": max_depth_param,
            "random-state": 42,
            "team-id": TEAM_ID,
            "student-id": STUDENT_ID,
            "semester": SEMESTER,
            "run-name": "github_actions_triggered_pipeline_run",
        },
    )

    # Step 1: PreprocessData
    step_process = ProcessingStep(
        name="PreprocessData",
        processor=processor,
        inputs=[ProcessingInput(source=input_data_url_param, destination="/opt/ml/processing/input")],
        outputs=[
            ProcessingOutput(output_name="train", source="/opt/ml/processing/output/train"),
            ProcessingOutput(output_name="test", source="/opt/ml/processing/output/test"),
        ],
        code=str(os.path.join(LOCAL_SRC_DIR, "preprocess.py")),
        job_arguments=["--test-size", "0.2", "--random-state", "42"],
    )

    # Step 2: TrainModel
    step_train = TrainingStep(
        name="TrainModel",
        estimator=estimator,
        inputs={
            "train": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["train"].S3Output.S3Uri,
                content_type="text/csv",
            ),
            "test": TrainingInput(
                s3_data=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                content_type="text/csv",
            ),
        },
    )

    # Step 3: EvaluateModel — dedicated structured evaluation step (fixes the
    # log-text-parsing fragility of the core pipeline; see Final Report 5.5/9.5)
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")

    step_eval = ProcessingStep(
        name="EvaluateModel",
        processor=processor,
        inputs=[
            ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                             destination="/opt/ml/processing/model"),
            ProcessingInput(source=step_process.properties.ProcessingOutputConfig.Outputs["test"].S3Output.S3Uri,
                             destination="/opt/ml/processing/test"),
        ],
        outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        code=str(os.path.join(LOCAL_SRC_DIR, "evaluate.py")),
        property_files=[evaluation_report],
    )

    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/evaluation.json".format(
                step_eval.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"]
            ),
            content_type="application/json",
        )
    )

    # Step 4: RegisterModel
    model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir=LOCAL_SRC_DIR,
        framework_version="1.4-2",
        py_version="py3",
        sagemaker_session=pipeline_session,
        env={"HOME": "/tmp", "PYTHONUSERBASE": "/tmp/.local", "PYTHONNOUSERSITE": "0"},
    )

    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["application/json"],
            response_types=["application/json"],
            inference_instances=["ml.m5.large"],
            transform_instances=["ml.m5.large"],
            model_package_group_name=MODEL_PACKAGE_GROUP_NAME,
            approval_status="PendingManualApproval",
            model_metrics=model_metrics,
        ),
    )

    # Step 5: CheckAUCQualityGate — reads the structured evaluation.json via JsonGet
    cond_auc = ConditionGreaterThanOrEqualTo(
        left=JsonGet(
            step_name=step_eval.name,
            property_file=evaluation_report,
            json_path="classification_metrics.auc_roc.value",
        ),
        right=quality_gate_auc_param,
    )
    step_cond = ConditionStep(
        name="CheckAUCQualityGate",
        conditions=[cond_auc],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=TRIGGERED_PIPELINE_NAME,
        parameters=[input_data_url_param, n_estimators_param, max_depth_param, quality_gate_auc_param],
        steps=[step_process, step_train, step_eval, step_cond],
        sagemaker_session=pipeline_session,
    )


if __name__ == "__main__":
    ensure_model_package_group()
    pipeline = build_pipeline()
    definition_json = json.loads(pipeline.definition())
    print("Pipeline definition is valid JSON.")
    upsert_response = pipeline.upsert(role_arn=ROLE_ARN)
    print("Pipeline upsert completed:", json.dumps(upsert_response, indent=2, default=str))
