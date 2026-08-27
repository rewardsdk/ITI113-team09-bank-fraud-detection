"""
Core SageMaker Pipeline definition (4-step: Preprocess -> Train -> QualityGate -> Register).

This is a faithful extraction of the pipeline-definition cells from
notebooks/03_fraud_sagemaker_pipeline.ipynb, refactored into an importable
function so it can be built/upserted/executed from a standalone script
(scripts/trigger_pipeline.py) — which is what the GitHub Actions workflow
in .github/workflows/trigger-core-pipeline.yml calls.

The pipeline itself is unchanged from what was interactively validated in
Notebook 03; only the packaging (module vs. notebook cells) is different.
"""
import os

os.environ.setdefault("SAGEMAKER_SUPPRESS_V2_WARNING", "1")

import boto3
import sagemaker
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterInteger
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

# ----------------------------------------------------------------------
# Configuration (mirrors Notebook 03, cell 2) — override via env vars so
# the same module works locally, in Studio, and in GitHub Actions.
# ----------------------------------------------------------------------
TEAM_ID = os.environ.get("TEAM_ID", "team09")
STUDENT_ID = os.environ.get("STUDENT_ID", "s901")
COURSE = "ITI113"
SEMESTER = os.environ.get("SEMESTER", "26S1")
PROJECT_NAME = "bank-fraud-detection"

BUCKET = os.environ.get("SM_BUCKET", "nyp-26s1-iti113")
REGION = os.environ.get("AWS_REGION", "ap-southeast-1")
ROLE_ARN = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]  # required, no safe default

PREFIX = f"iti113/{TEAM_ID}/data/{PROJECT_NAME}"
PIPELINE_NAME = f"iti113-{TEAM_ID}-bank-fraud-detection"
MODEL_PACKAGE_GROUP = f"{TEAM_ID}-BankFraudDetection"
QUALITY_GATE_AUC_DEFAULT = float(os.environ.get("QUALITY_GATE_AUC", "0.60"))

RAW_DATA_URI = f"s3://{BUCKET}/{PREFIX}/raw/bank_fraud.csv"
PIPELINE_ROOT = f"s3://{BUCKET}/{PREFIX}/pipeline"

SCRIPTS_S3_PREFIX = f"{PREFIX}/pipeline_src"
SCRIPTS_S3_URI = f"s3://{BUCKET}/{SCRIPTS_S3_PREFIX}"

LOCAL_SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "core")

PROCESSING_INSTANCE_TYPE = os.environ.get("PROCESSING_INSTANCE_TYPE", "ml.m5.large")
TRAINING_INSTANCE_TYPE = os.environ.get("TRAINING_INSTANCE_TYPE", "ml.m5.large")


def upload_source_scripts(s3_client=None):
    """Uploads src/core/*.py to the pipeline's S3 script prefix, exactly as
    Notebook 03 cell 10 does, so the pipeline always trains against the
    version of the scripts currently checked in to this repo."""
    s3_client = s3_client or boto3.client("s3", region_name=REGION)
    files = ["preprocess.py", "train.py", "inference.py", "requirements.txt"]
    for filename in files:
        local_path = os.path.join(LOCAL_SRC_DIR, filename)
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Missing local source file: {local_path}")
        s3_key = f"{SCRIPTS_S3_PREFIX}/{filename}"
        s3_client.upload_file(local_path, BUCKET, s3_key)
        print(f"Uploaded {local_path} -> s3://{BUCKET}/{s3_key}")
    return SCRIPTS_S3_URI


def build_pipeline() -> Pipeline:
    """Builds (but does not upsert or execute) the 4-step core pipeline,
    matching Notebook 03 cells 12-17 exactly:
      PreprocessData -> TrainModel -> AUCQualityGate(condition) -> RegisterModel
    """
    pipeline_session = PipelineSession(
        boto_session=boto3.Session(region_name=REGION),
        default_bucket=BUCKET,
    )

    # Pipeline parameters — overridable at execution time
    p_n_est = ParameterInteger(name="NEstimators", default_value=500)
    p_depth = ParameterInteger(name="MaxDepth", default_value=8)
    p_gate = ParameterFloat(name="QualityGateAUC", default_value=QUALITY_GATE_AUC_DEFAULT)

    sklearn_version = "1.2-1"

    # Step 1: PreprocessData
    processor = SKLearnProcessor(
        framework_version=sklearn_version,
        instance_type=PROCESSING_INSTANCE_TYPE,
        instance_count=1,
        role=ROLE_ARN,
        sagemaker_session=pipeline_session,
        base_job_name=f"iti113-{TEAM_ID}-{STUDENT_ID}-process",
    )

    step_process = ProcessingStep(
        name="PreprocessData",
        step_args=processor.run(
            code=f"{LOCAL_SRC_DIR}/preprocess.py",
            inputs=[ProcessingInput(source=RAW_DATA_URI, destination="/opt/ml/processing/input")],
            outputs=[
                ProcessingOutput(
                    output_name="processed",
                    source="/opt/ml/processing/output",
                    destination=f"{PIPELINE_ROOT}/processed/",
                    s3_upload_mode="EndOfJob",
                )
            ],
            arguments=["--input-dir", "/opt/ml/processing/input", "--output-dir", "/opt/ml/processing/output"],
        ),
    )

    # Step 2: TrainModel
    estimator = SKLearn(
        entry_point="train.py",
        source_dir=LOCAL_SRC_DIR,
        framework_version=sklearn_version,
        instance_type=TRAINING_INSTANCE_TYPE,
        instance_count=1,
        role=ROLE_ARN,
        base_job_name=f"iti113-{TEAM_ID}-{STUDENT_ID}-train",
        sagemaker_session=pipeline_session,
        hyperparameters={
            "n-estimators": p_n_est,
            "max-depth": p_depth,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "gamma": 0.1,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "smote_sampling_strategy": 0.5,
            "early_stopping_rounds": 30,
            "random-state": 42,
            "team-id": TEAM_ID,
            "student-id": STUDENT_ID,
            "semester": SEMESTER,
            "run-name": "github_actions_pipeline_run",
        },
        environment={"TEAM_ID": TEAM_ID, "STUDENT_ID": STUDENT_ID, "SEMESTER": SEMESTER},
        metric_definitions=[
            {"Name": "test_auc_roc", "Regex": r"Test AUC-ROC: ([0-9\.]+)"},
            {"Name": "test_accuracy", "Regex": r"test_accuracy: ([0-9\.]+)"},
            {"Name": "test_f1", "Regex": r"test_f1: ([0-9\.]+)"},
            {"Name": "test_recall", "Regex": r"test_recall: ([0-9\.]+)"},
            {"Name": "test_precision", "Regex": r"test_precision: ([0-9\.]+)"},
            {"Name": "test_pr_auc", "Regex": r"test_pr_auc: ([0-9\.]+)"},
            {"Name": "best_threshold", "Regex": r"best_threshold: ([0-9\\.]+)"},
        ],
        tags=[
            {"Key": "Course", "Value": COURSE},
            {"Key": "Semester", "Value": SEMESTER},
            {"Key": "Team", "Value": TEAM_ID},
            {"Key": "Student", "Value": STUDENT_ID},
            {"Key": "Model", "Value": "SMOTE-XGBoost"},
            {"Key": "TriggeredBy", "Value": "GitHubActions"},
        ],
    )

    processed_uri = step_process.properties.ProcessingOutputConfig.Outputs["processed"].S3Output.S3Uri

    step_train = TrainingStep(
        name="TrainModel",
        step_args=estimator.fit(
            inputs={
                "train": sagemaker.inputs.TrainingInput(
                    s3_data=processed_uri, content_type="text/csv",
                    s3_data_type="S3Prefix", distribution="FullyReplicated",
                ),
                "test": sagemaker.inputs.TrainingInput(
                    s3_data=processed_uri, content_type="text/csv",
                    s3_data_type="S3Prefix", distribution="FullyReplicated",
                ),
            },
        ),
        depends_on=[step_process],
    )

    # Step 3: RegisterModel
    model = SKLearnModel(
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir=LOCAL_SRC_DIR,
        framework_version=sklearn_version,
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
            model_package_group_name=MODEL_PACKAGE_GROUP,
            approval_status="PendingManualApproval",
        ),
    )

    # Step 4: AUCQualityGate
    # NOTE: as in Notebook 03, this condition actually gates on the captured
    # "test_pr_auc" metric (not test_auc_roc, despite the parameter name
    # QualityGateAUC/p_gate) — kept identical to the validated notebook logic.
    condition = ConditionGreaterThanOrEqualTo(
        left=step_train.properties.FinalMetricDataList["test_pr_auc"].Value,
        right=p_gate,
    )
    step_condition = ConditionStep(
        name="AUCQualityGate",
        conditions=[condition],
        if_steps=[step_register],
        else_steps=[],
    )

    return Pipeline(
        name=PIPELINE_NAME,
        parameters=[p_n_est, p_depth, p_gate],
        steps=[step_process, step_train, step_condition],
        sagemaker_session=pipeline_session,
    )


if __name__ == "__main__":
    upload_source_scripts()
    pipeline = build_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN)
    print(f'Pipeline "{PIPELINE_NAME}" upserted.')
