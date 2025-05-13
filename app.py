from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and encoders
model = joblib.load('model.pkl')
le_location = joblib.load('location_encoder.pkl')
le_weather = joblib.load('weather_encoder.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        location = request.form['location']
        temperature = float(request.form['temperature'])
        humidity = float(request.form['humidity'])
        pressure = float(request.form['pressure'])
        wind_speed = float(request.form['wind_speed'])
        date = request.form['date']  # Get the selected date

        # Encode location
        location_encoded = le_location.transform([location])[0]

        # Prepare features for prediction
        features = np.array([[location_encoded, temperature, humidity, pressure, wind_speed]])

        # Make prediction
        pred_encoded = model.predict(features)[0]

        # Decode the prediction back to weather label
        prediction = le_weather.inverse_transform([pred_encoded])[0]

        # Return the result with prediction, temperature, and date
        return render_template('index.html', prediction=prediction, temperature=temperature, date=date)
    
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
