from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
import pickle

# Step 1: Initialize Flask app
app = Flask(__name__)

# Step 2: Sample Data (same as the training data in the previous example)
data = {
    'Age': [45, 50, 37, 62, 29],
    'Cholesterol (mg/dL)': [220, 250, 190, 230, 180],
    'Blood Pressure (mm Hg)': [130, 140, 120, 135, 125],
    'Glucose (mg/dL)': [100, 110, 90, 105, 95],
    'BMI': [24.5, 28.7, 22.1, 29.5, 21.0],
    'Disease': [1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Step 3: Preprocess the data (Normalization)
scaler = MinMaxScaler()
X = scaler.fit_transform(df[['Age', 'Cholesterol (mg/dL)', 'Blood Pressure (mm Hg)', 'Glucose (mg/dL)', 'BMI']])
y = df['Disease']

# Step 4: Train the Logistic Regression model (for simplicity, we'll retrain here)
model = LogisticRegression()
model.fit(X, y)

# Step 5: Save the model (Optional, in case you want to save and load it later)
# pickle.dump(model, open("disease_model.pkl", "wb"))

# Step 6: Load the model (if saved)
# model = pickle.load(open("disease_model.pkl", "rb"))

# Step 7: Create the home page with a chatbot-like interface
@app.route('/')
def home():
    return render_template('chat.html')  # This is a simple HTML page we'll create

# Step 8: Prediction route that accepts form inputs
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get user input from form
            age = float(request.form['age'])
            cholesterol = float(request.form['cholesterol'])
            bp = float(request.form['bp'])
            glucose = float(request.form['glucose'])
            bmi = float(request.form['bmi'])

            # Prepare input for the model (scale the input)
            input_data = np.array([[age, cholesterol, bp, glucose, bmi]])
            input_scaled = scaler.transform(input_data)

            # Get the prediction
            prediction = model.predict(input_scaled)[0]
            if prediction == 1:
                result = "Based on your test results, you may be at risk for the disease. Please consult with a doctor for further evaluation."
            else:
                result = "Your test results look good. However, it's always a good idea to consult with a doctor for further evaluation."
        except:
            result = "Error processing your input. Please make sure to enter valid numerical values."

        return render_template('chat.html', prediction_text=result)

# Step 9: Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
