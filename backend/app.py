
# Import necessary libraries
import numpy as np
import joblib # Loading the serialized model
import pandas as pd # for data manipulation
from flask import Flask, request, jsonify # Creating the Flask API

# Initialize the Flask application
superkart_model_predictor_api = Flask("SuperKart Model Predictor")

# Load the saved model pipeline from the file
saved_model = joblib.load('backend_files/superkart_model.joblib')

# Define a route for the home page (GET request)
@superkart_model_predictor_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Product Store Sales Prediction API!"

# Define an endpoint for single product sales prediction (POST request)
@superkart_model_predictor_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales total as a JSON response.
    """
    # Get the JSON data from the request body
    input_payload = request.get_json()

    # Extract relevant features matching the model's expected columns
    sample = {
        'Product_Weight': input_payload['Product_Weight'],
        'Product_Sugar_Content': input_payload['Product_Sugar_Content'],
        'Product_Allocated_Area': input_payload['Product_Allocated_Area'],
        'Product_MRP': input_payload['Product_MRP'],
        'Store_Size': input_payload['Store_Size'],
        'Store_Location_City_Type': input_payload['Store_Location_City_Type'],
        'Store_Type': input_payload['Store_Type']
    }

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction directly (targets are modeled in raw dollars, no log_price used)
    predicted_sales = saved_model.predict(input_data)[0]

    # Convert predicted sales value to float and round to 2 decimal places
    predicted_sales = round(float(predicted_sales), 2)

    # Return the actual predicted sales total
    return jsonify({'Predicted Product Store Sales': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_model_predictor_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing details for multiple products
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Make predictions for all records in the DataFrame
    predictions = saved_model.predict(input_data).tolist()

    # Round prediction float values to 2 decimal places
    predicted_sales_list = [round(float(val), 2) for val in predictions]

    # Create a dictionary of predictions mapping index to predicted sales total
    output_dict = {str(i): val for i, val in enumerate(predicted_sales_list)}

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_model_predictor_api.run(debug=True)
