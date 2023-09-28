from flask import Flask, render_template, request, jsonify
import flask
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


app = Flask(__name__)


# Load the pre-trained model
model = joblib.load('docs/churn_prediction.pkl')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        age = int(request.form.get('age'))
        gender = request.form.get('gender')
        location = request.form.get('location')
        subscription_length = int(request.form.get('subscription_length'))
        monthly_bill = float(request.form.get('monthly_bill'))
        total_usage_gb = float(request.form.get('total_usage_gb'))

        # Preprocess input data
        input_data = pd.DataFrame({
            'Age': [age],
            'Gender': [gender],
            'Location': [location],
            'Subscription_Length_Months': [subscription_length],
            'Monthly_Bill': [monthly_bill],
            'Total_Usage_GB': [total_usage_gb]
        })

        # Handle missing values (for example, fill with the median for numeric columns)
        input_data['Age'].fillna(input_data['Age'].median(), inplace=True)

        # Encode categorical variables (Gender and Location) using one-hot encoding
        input_data = pd.get_dummies(input_data, columns=['Gender', 'Location'], drop_first=True)

        # Define expected feature names for model input
        expected_features = ['Age', 'Subscription_Length_Months', 'Monthly_Bill', 'Total_Usage_GB',
                             'Gender_Male', 'Location_Location_Name']

        # Ensure that input_data has the same columns as expected_features
        for feature in expected_features:
            if feature not in input_data.columns:
                input_data[feature] = 0  # Add missing feature with default value

        # Reorder columns to match the expected order
        input_data = input_data[expected_features]

        # Make a churn prediction
        prediction = model.predict(input_data)

        # Display prediction result
        prediction_text = 'Churn Prediction is correct' if prediction[0] == 1 else 'Churn Prediction is incorrect'

        return render_template('index.html', prediction_text=prediction_text)
    except Exception as e:
        return str(e)



import requests
import json

# Send a POST request to the Flask endpoint
if __name__ == "__main__":
    app.run(port=3000,debug = True) 


