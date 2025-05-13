import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
data = pd.read_csv('weather_data.csv')

# Encode 'location' and 'weather'
le_location = LabelEncoder()
le_weather = LabelEncoder()

data['location_encoded'] = le_location.fit_transform(data['location'])
data['weather_encoded'] = le_weather.fit_transform(data['weather'])

# Features and target
X = data[['location_encoded', 'temperature', 'humidity', 'pressure', 'wind_speed']]
y = data['weather_encoded']

# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, 'model.pkl')
joblib.dump(le_location, 'location_encoder.pkl')
joblib.dump(le_weather, 'weather_encoder.pkl')
