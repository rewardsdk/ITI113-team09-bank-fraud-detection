# ITI113 — Team 9 — Bank Transaction Fraud Detection

Real-time bank transaction fraud detection system: EDA, model experimentation
(Logistic Regression / Random Forest / XGBoost), two SageMaker Pipelines,
a Gradio demo, and AI Verify fairness/robustness testing — with the two
SageMaker Pipelines re-triggerable directly from GitHub Actions.

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       ├── trigger-core-pipeline.yml        # runs Notebook 03's 4-step pipeline on demand
│       ├── trigger-triggered-pipeline.yml   # runs Notebook 05's 5-step pipeline on demand
│       └── notebook-ci.yml                  # cheap, no-AWS validation on every push/PR
│
├── notebooks/                                # the 6 submitted notebooks, outputs intact
│   ├── 01_fraud_eda_and_data_preparation.ipynb
│   ├── 02_fraud_baseline_experiments.ipynb
│   ├── 03_fraud_sagemaker_pipeline.ipynb
│   ├── 04_fraud_gradio_serverless.ipynb
│   ├── 05_fraud_model_b.ipynb
│
├── pipeline/                                  # SageMaker Pipeline SDK definitions,
│   ├── __init__.py                            # extracted from the notebooks' pipeline-
│   ├── core_pipeline.py                       # definition cells so they can be built,
│   └── triggered_pipeline.py                  # upserted and executed outside a notebook
│
├── src/                                       # the exact %%writefile script contents
│   ├── core/                                  # from Notebook 03 (preprocess/train/
│   │   ├── preprocess.py                      # train_rf/inference.py + requirements.txt)
│   │   ├── train_rf.py
│   │   ├── train.py
│   │   ├── inference.py
│   │   └── requirements.txt
│   └── triggered/                             # from Notebook 05 (adds evaluate.py)
│       ├── preprocess.py
│       ├── train.py
│       ├── evaluate.py
│       ├── inference.py
│       └── requirements.txt
│
├── scripts/
│   └── trigger_pipeline.py                    # CLI: build -> upsert -> execute -> poll
│                                               # a pipeline; called by both workflows above
│
├── requirements-dev.txt                       # nbformat/flake8/black — CI only
├── .gitignore
└── README.md
```

### Why `pipeline/` and `src/` are separate from `notebooks/`

The pipeline-definition code (`Pipeline(...).upsert()` etc.) and the
training/inference scripts (`%%writefile src/train.py` etc.) already exist
*inside* the notebooks. They are duplicated here as standalone `.py` files
for one reason: **GitHub Actions cannot meaningfully "run a notebook cell."**
A CI/CD trigger needs an importable function and a CLI entrypoint it can
call non-interactively. `pipeline/core_pipeline.py` and
`pipeline/triggered_pipeline.py` are line-for-line extractions of the
validated notebook cells (see the module docstrings), not a rewrite — so
there is one behavioural source of truth, just packaged two ways.


## How the GitHub Actions triggers work

Both `trigger-*-pipeline.yml` workflows do the same four things:

1. **Check out the repo** and install `sagemaker`/`boto3` (no AWS creds yet).
2. **Assume an AWS IAM role via OIDC** (`aws-actions/configure-aws-credentials`)
   — no long-lived AWS access keys are stored in GitHub.
3. **Call `scripts/trigger_pipeline.py --pipeline core|triggered --wait`**,
   which imports `pipeline/core_pipeline.py` or `pipeline/triggered_pipeline.py`,
   calls `.upsert()` (so the pipeline definition in SageMaker always matches
   what's checked in) and `.start(parameters=...)`, then polls
   `describe()`/`list_steps()` every 30s.
4. **Exit non-zero if any step fails**, printing that step's CloudWatch logs
   inline in the Actions log — so a failed pipeline run shows up as a red ❌
   on the commit/PR, the same way a failed test suite would.

Each workflow triggers on:
- **`workflow_dispatch`** — a manual "Run workflow" button in the Actions
  tab, with inputs for hyperparameters (and, for the triggered pipeline,
  which S3 CSV to retrain against).
- **`push` to `main`, path-filtered** to only that pipeline's own
  `src/<core|triggered>/**` and `pipeline/<name>.py` — so pushing a Gradio
  fix or a notebook update does **not** accidentally spend SageMaker
  compute minutes.


## One-time setup

### 1. AWS side — OIDC trust (recommended)

1. Add GitHub's OIDC provider to the account (if not already present):
   `https://token.actions.githubusercontent.com`, audience `sts.amazonaws.com`.
2. Create (or extend) an IAM role — e.g. `GitHubActions-ITI113-Team09` — with
   a trust policy scoped to this repo, for example:

   ```json
   {
     "Effect": "Allow",
     "Principal": { "Federated": "arn:aws:iam::044528205969:oidc-provider/token.actions.githubusercontent.com" },
     "Action": "sts:AssumeRoleWithWebIdentity",
     "Condition": {
       "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
       "StringLike":  { "token.actions.githubusercontent.com:sub": "repo:<org>/iti113-team09-bank-fraud-detection:*" }
     }
   }
   ```

3. Attach permissions equivalent to (or the same as) the team's existing
   `SageMakerExecutionRole-ITI113-Team09` — `sagemaker:*Pipeline*`,
   `sagemaker:*TrainingJob*`, `sagemaker:*ProcessingJob*`,
   `sagemaker:*ModelPackage*`, plus S3 read/write on
   `s3://nyp-26s1-iti113/iti113/team09/*`.


### 2. GitHub side — secrets and variables

Repo → Settings → Secrets and variables → Actions:

| Type | Name | Value |
|---|---|---|
| Secret | `AWS_SAGEMAKER_DEPLOY_ROLE_ARN` | the OIDC-trusted role from step 1 (only if using OIDC) |
| Secret | `SAGEMAKER_EXECUTION_ROLE_ARN` | `arn:aws:iam::044528205969:role/SageMakerExecutionRole-ITI113-Team09` |
| Variable | `AWS_REGION` | `ap-southeast-1` |
| Variable | `SM_BUCKET` | `nyp-26s1-iti113` |
| Variable | `TEAM_ID` | `team09` |
| Variable | `STUDENT_ID` | `s901` |


### 3. Check in the notebooks

```bash
git checkout -b setup-repo-structure
mkdir -p notebooks
cp /path/to/01_fraud_eda_and_data_preparation.ipynb notebooks/
cp /path/to/02_fraud_baseline_experiments.ipynb notebooks/
cp /path/to/03_fraud_sagemaker_pipeline.ipynb notebooks/
cp /path/to/04_fraud_gradio_serverless.ipynb notebooks/
cp /path/to/05_fraud_model_b.ipynb notebooks/
git add notebooks/
git commit -m "Check in all 6 project notebooks"
git push -u origin setup-repo-structure
```

### 4. Try a manual trigger

Actions tab → **Trigger Core SageMaker Pipeline** → **Run workflow** →
leave the defaults → **Run workflow**. Watch the job log; it should show
`upsert` → `Execution ARN: ...` → per-step status lines → a final
`Succeeded`/`Failed`.

