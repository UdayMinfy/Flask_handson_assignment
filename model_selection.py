import mlflow
import mlflow.sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

def evaluate_models_with_grid_search(X_train, X_test, y_train, y_test):
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(class_weight='balanced', max_iter=1000, solver='liblinear'),
            "params": {
                "C": [0.01, 0.1, 1, 10]
            }
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(class_weight='balanced', random_state=42),
            "params": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {}
        }
    }

    best_score = 0
    best_overall_model = None
    best_overall_params = {}
    best_model_uri = ""
    best_run_id = ""

    mlflow.set_experiment("Loan_Default_Classification_v3")

    for name, item in models.items():
        with mlflow.start_run(run_name=name) as run:
            print(f"\nTraining {name}...")

            # GridSearchCV if params exist, else fit directly
            if item["params"]:
                grid = GridSearchCV(item["model"], item["params"], cv=3, scoring='accuracy', n_jobs=-1)
                grid.fit(X_train, y_train)
                best_model = grid.best_estimator_
                best_params = grid.best_params_
            else:
                item["model"].fit(X_train, y_train)
                best_model = item["model"]
                best_params = {}

            # Predict and evaluate
            preds = best_model.predict(X_test)
            score = accuracy_score(y_test, preds)
            report = classification_report(y_test, preds, output_dict=True)

            # Log best params
            mlflow.log_params(best_params)
            mlflow.log_metric("accuracy", score)

            # Log metrics per class (usually "0" and "1")
            for label in ["0", "1"]:
                mlflow.log_metric(f"precision_{label}", report[label]["precision"])
                mlflow.log_metric(f"recall_{label}", report[label]["recall"])
                mlflow.log_metric(f"f1_score_{label}", report[label]["f1-score"])
                mlflow.log_metric(f"support_{label}", report[label]["support"])

            if "macro avg" in report:
                macro_avg = report["macro avg"]
                mlflow.log_metric("precision_macro", macro_avg["precision"])
                mlflow.log_metric("recall_macro", macro_avg["recall"])
                mlflow.log_metric("f1_score_macro", macro_avg["f1-score"])
            else:
                print("⚠️ 'macro avg' not found in classification report!")

            # Save model
            model_path = "model"
            mlflow.sklearn.log_model(best_model, artifact_path=model_path)
            model_uri = f"runs:/{run.info.run_id}/{model_path}"
            mlflow.set_tag("model_name", name)

            # Output
            print(f"\nBest Params for {name}: {best_params if best_params else 'Default'}")
            print(f"Accuracy: {score:.4f}")
            print(f"Classification Report for {name}:\n{classification_report(y_test, preds)}")
            print("=" * 60)

            # Save best model info
            if score > best_score:
                best_score = score
                best_overall_model = best_model
                best_overall_params = best_params
                best_model_uri = model_uri
                best_run_id = run.info.run_id

    return best_overall_model, best_overall_params
