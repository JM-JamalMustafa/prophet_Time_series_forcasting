
#  API for future prediction of samrt hr 
# ----------------------------------------------------------------------
from flask import Flask, jsonify, request
import joblib
import pandas as pd
from prophet import Prophet
from pymongo import MongoClient


app = Flask(__name__)

# Connect to MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['admin']  # Replace with your database name
#  Create two collections
data_collection = db['data'] 


# Load the trained model
model = joblib.load('D:/Smarthr project/smart hr/prophet_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = list(data_collection.find({}))  # Expecting a JSON payload

    # Convert the received data to a DataFrame
    df = pd.DataFrame(data)

    # Preprocess the data
    df["loginAt"] = pd.to_datetime(df["loginAt"])
    df["Date"] = df["loginAt"].dt.date

    # Group by Date and count unique employeeIds
    employee_count = df.groupby('Date')['employeeId'].nunique().reset_index()
    
    # Rename the DataFrame for Prophet
    ds = employee_count.rename(columns={'Date': 'ds', 'employeeId': 'y'})

    # Create a new DataFrame for the next month
    future_dates = pd.date_range(start=pd.to_datetime('today').normalize() + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1), periods=30, freq='D')
    future_df = pd.DataFrame({'ds': future_dates})

    # Fit the model on the existing data
      
    # Predict using the model
    forecast = model.predict(future_df)

    # Prepare the response with only employee counts
    response = {
        'predictions': forecast[['ds', 'yhat']].to_dict(orient='records')  # Only include yhat for employee counts
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
