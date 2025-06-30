from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("best_final_model.pkl")

@app.route('/')
def home():
    return render_template('index 1.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        input_data = {
            'Age': int(request.form['Age']),
            'Experience': int(request.form['Experience']),
            'Income': int(request.form['Income']),
            'ZIP Code': int(request.form['ZIPCode']),
            'Family': int(request.form['Family']),
            'CCAvg': float(request.form['CCAvg']),
            'Education': int(request.form['Education']),
            'Mortgage': int(request.form['Mortgage']),
            'Securities Account': int(request.form['Securities Account']),
            'CD Account': int(request.form['CD Account']),
            'Online': int(request.form['Online']),
            'CreditCard': int(request.form['CreditCard']),
        }

        # Convert to DataFrame
        df = pd.DataFrame([input_data])

        # Prediction
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]  # Probability of being a defaulter

        result = "Defaulter" if prediction == 1 else "Non-Defaulter"
        prob_display = f"{probability * 100:.2f}%"

        return render_template("index 1.html", prediction=result, probability=prob_display)
        #return render_template("form.html", prediction=result, probability=prob_display)

    except Exception as e:
        return f"Error during prediction: {e}"

if __name__ == '__main__':
    app.run(debug=True)
