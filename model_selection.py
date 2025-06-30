from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, accuracy_score

from sklearn.tree import DecisionTreeClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC
 
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

        }

    }
    best_score=0
 
    for name, item in models.items():

        print(f"\n Training {name} with GridSearchCV...")

        grid = GridSearchCV(item["model"], item["params"], cv=3, scoring='accuracy', n_jobs=-1)

        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_

        preds = best_model.predict(X_test)
        score = accuracy_score(y_test,preds)
 
        print(f"\n Best Params for {name}: {grid.best_params_}")

        print(f"Accuracy: {accuracy_score(y_test, preds):.4f}")

        print(f"Classification Report for {name}:")

        print(classification_report(y_test, preds))

        print("=" * 60)
    if score > best_score:
            best_overall_model = best_model
            best_overall_params = grid.best_params_
            best_score = score
 
    return best_overall_model, best_overall_params
 