from datetime import datetime
import requests
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load the saved models and scaler
with open('./models/max_temp.pkl', 'rb') as file:
    high_temp_model = pickle.load(file)

with open('./models/min_temp.pkl', 'rb') as file:
    low_temp_model = pickle.load(file)

with open('./models/precipitation.pkl', 'rb') as file:
    precipitation_model = pickle.load(file)

with open('./models/scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

def get_weather_data(lat, lon):
    api_key = 'API KEY'
    url = f"https://api.weatherbit.io/v2.0/forecast/daily?lat={lat}&lon={lon}&key={api_key}"
    response = requests.get(url)

    if response.status_code != 200:
        return None

    return response.json()

def extract_features_for_model(api_response, daily_data, target_variable):
    features = {
        'LATITUDE': api_response['lat'],
        'LONGITUDE': api_response['lon'],
        'PRCP': daily_data['precip'],
        'TAVG': daily_data['temp'],
        'TMAX': daily_data['max_temp'],
        'TMIN': daily_data['min_temp']
    }
    
    # Remove the target variable from the features
    if target_variable == 'PRCP':
        del features['PRCP']
    elif target_variable == 'TMAX':
        del features['TMAX']
    elif target_variable == 'TMIN':
        del features['TMIN']
    
    return list(features.values())

def scale_features(features):
    return scaler.transform([features])

@app.route('/forecast', methods=['GET'])
def forecast():
    # Assuming lat and lon are provided via request args
    lat = request.args.get('lat')
    lon = request.args.get('lon')

    if not lat or not lon:
        return jsonify({'error': 'Please provide latitude and longitude'}), 400

    # Fetch initial weather data from the Weatherbit API
    api_response = get_weather_data(lat, lon)
    if api_response is None:
        return jsonify({'error': 'Failed to fetch weather data from the API'}), 500
    
    # Extract the 7-day forecast data
    daily_data_list = api_response['data']

    # Initialize lists to store predictions
    prcp_predictions = []
    tmax_predictions = []
    tmin_predictions = []

    # Loop through each day's forecast and make predictions
    for daily_data in daily_data_list:
        # Prepare features for each model
        features_prcp = extract_features_for_model(api_response, daily_data, 'PRCP')
        features_tmax = extract_features_for_model(api_response, daily_data, 'TMAX')
        features_tmin = extract_features_for_model(api_response, daily_data, 'TMIN')

        # Scale the features using the pre-fitted scaler
        scaled_features_prcp = scale_features(features_prcp)
        scaled_features_tmax = scale_features(features_tmax)
        scaled_features_tmin = scale_features(features_tmin)

        # Make predictions
        prcp_prediction = precipitation_model.predict(scaled_features_prcp)[0]
        tmax_prediction = high_temp_model.predict(scaled_features_tmax)[0]
        tmin_prediction = low_temp_model.predict(scaled_features_tmin)[0]

        # Append the predictions to the lists
        prcp_predictions.append(prcp_prediction)
        tmax_predictions.append(tmax_prediction)
        tmin_predictions.append(tmin_prediction)

    # Combine the predictions into the forecast data
    forecast_data = {
        'daily_forecasts': [
            {
                'date': datetime.strptime(daily_data['datetime'], '%Y-%m-%d').strftime('%A'),  # Convert to day of the week,
                'prcp': prcp,
                'high_temp': tmax,
                'low_temp': tmin
            }
            for daily_data, prcp, tmax, tmin in zip(daily_data_list, prcp_predictions, tmax_predictions, tmin_predictions)
        ]
    }

    return jsonify(forecast_data)

if __name__ == '__main__':
    app.run(debug=True)

#api url
#http://127.0.0.1:5000/forecast?lat=27.70169&lon=85.3206
