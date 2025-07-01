import mlflow
from mlflow.tracking import MlflowClient

def register_best_model_from_experiment(
    experiment_name="Loan_Default_Classification_v3",
    model_registry_name="Loan_Default_Best_Model"
):
    client = MlflowClient()

    # Step 1: Get experiment ID
    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        print(f"❌ Experiment '{experiment_name}' not found.")
        return

    experiment_id = experiment.experiment_id

    # Step 2: Search all completed runs sorted by f1_score_macro DESC
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.f1_score_macro DESC"],
        max_results=1
    )

    if not runs:
        print("❌ No completed runs found with f1_score_macro.")
        return

    best_run = runs[0]

    # Step 3: Get details from best run
    run_id = best_run.info.run_id
    best_model_uri = f"runs:/{run_id}/model"
    best_model_name = best_run.data.tags.get("model_name", "Unknown")
    best_f1_score = best_run.data.metrics.get("f1_score_macro", "N/A")

    print(f"✅ Best model: {best_model_name}")
    print(f"Run ID: {run_id}")
    print(f"f1_score_macro: {best_f1_score}")
    print(f"Model URI: {best_model_uri}")

    # Step 4: Register the model
    print(f"\n📦 Registering model as '{model_registry_name}'...")
    result = mlflow.register_model(model_uri=best_model_uri, name=model_registry_name)

    # Step 5: Transition it to 'Production' (archive previous versions if any)
    client.transition_model_version_stage(
        name=model_registry_name,
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🚀 Model '{model_registry_name}' (version {result.version}) promoted to 'Production'.")
