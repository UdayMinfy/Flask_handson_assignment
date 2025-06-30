from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle
#import psycopg2
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model_path = 'best_final_model.pkl'
import joblib
import os
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'best_final_model.pkl')
model = joblib.load(MODEL_PATH)


@app.route('/')
def home():
    return render_template('index.html')  # Ensure your HTML is saved as templates/index.html

@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        # Get the uploaded file
        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return "No file selected."

        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        uploaded_file.save(file_path)

        # Get PostgreSQL credentials
        #username = request.form['user']
        #password = request.form['pw']
        #dbname = request.form['db']

        try:
            # Read the uploaded CSV
            df = pd.read_csv(file_path)
            df_new=df.drop(columns=['ID'])
            # Predict using the loaded model
            predictions = model.predict(df_new)

            # Add predictions to the dataframe
            df['Prediction'] = predictions

            # Optional: Save to PostgreSQL (if needed)
            # Uncomment if you want to insert predictions
            """
            import sqlalchemy
            engine = sqlalchemy.create_engine(f'postgresql://{username}:{password}@localhost:5432/{dbname}')
            df.to_sql('predictions', engine, if_exists='replace', index=False)
            """

            return df.to_html(classes='table table-striped')

        except Exception as e:
            return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
