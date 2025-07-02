import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from sklearn.model_selection import train_test_split

# === CONFIG ===
#HISTORICAL_DATA_PATH = "historical_data.csv"
NEW_DATA_PATH = "New Customer Bank_Personal_Loan.csv"
MLFLOW_EXPERIMENT_NAME = "Data Drift Analysis"

# === Load Data ===
historical_df = pd.read_excel("Bank_Personal_Loan_Modelling.xlsx",sheet_name='Data')
new_df = pd.read_csv(NEW_DATA_PATH)

# Drop ID or non-feature columns if needed
if 'ID' in historical_df.columns:
    historical_df = historical_df.drop(columns=['ID'])
if 'ID' in new_df.columns:
    new_df = new_df.drop(columns=['ID'])

# Separate features and target if target column exists
target_column = 'Personal Loan' if 'Personal Loan' in historical_df.columns else None
if target_column:
    X = historical_df.drop(columns=[target_column])
    y = historical_df[target_column]
else:
    X = historical_df
    y = None

# Split for train/test (only for Run 1)
train_df, test_df = train_test_split(X, test_size=0.3, random_state=42)

# === Setup MLflow ===
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

# === Function to Run Report, Save, Log to MLflow ===
def log_data_drift_run(run_name, reference_df, current_df, report_file):
    report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
        TargetDriftPreset()
    ])
    report.run(reference_data=reference_df, current_data=current_df)
    report.save_html(report_file)
    report_dict = report.as_dict()

    with mlflow.start_run(run_name=run_name):
        # Log column-level drift scores
        drift_results = report_dict.get("metrics", [])
        for metric in drift_results:
            if metric.get("metric") == "DataDriftTable":
                for feature in metric.get("result", {}).get("drift_by_columns", {}):
                    score = metric["result"]["drift_by_columns"][feature].get("drift_score")
                    stat_test = metric["result"]["drift_by_columns"][feature].get("stat_test_name")
                    detected = metric["result"]["drift_by_columns"][feature].get("drift_detected")
                    mlflow.log_metric(f"{feature}_drift_score", score)
                    mlflow.log_param(f"{feature}_stat_test", stat_test)
                    mlflow.log_param(f"{feature}_drift_detected", detected)
        # Log HTML report
        mlflow.log_artifact(report_file, artifact_path="evidently_report")
        print(f"✅ Logged run: {run_name}, Report: {report_file}")

# === Run 1: Train vs Test on Historical CSV ===
log_data_drift_run(
    run_name="Train_vs_Test_Historical",
    reference_df=train_df,
    current_df=test_df,
    report_file="report_train_vs_test.html"
)

# === Run 2: Historical CSV (full) vs New Data CSV ===
log_data_drift_run(
    run_name="HistoricalCSV_vs_NewDataCSV",
    reference_df=X,        # full historical features
    current_df=new_df,
    report_file="report_historical_vs_new.html"
)

print("\n✅ Both runs completed and logged to MLflow.")
