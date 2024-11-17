from flask import Flask, render_template, request
import pickle
import pandas as pd
import math

# Load models
with open("clf_fan_model.pkl", "rb") as fan_file:
    clf_fan_loaded = pickle.load(fan_file)
with open("clf_lamp_model.pkl", "rb") as lamp_file:
    clf_lamp_loaded = pickle.load(lamp_file)

# Initialize Flask app
app = Flask(__name__)

# Define route for main page
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        barn_temperature=10,
        humidity=20,
        luminance=2,
        weather="clear_sky",
        hour=0,
        fan_prediction=None,
        lamp_prediction=None,
    )

# Define route to handle predictions
@app.route("/predict", methods=["POST"])
def predict():
    # Get input values from form
    barn_temperature = float(request.form["barn_temperature"])
    humidity = int(request.form["humidity"])
    luminance = int(request.form["luminance"])
    weather = request.form["weather"]
    hour = int(request.form["hour"])

    # Convert hour to hour_sin and hour_cos
    hour_rad = (hour / 24) * 2 * math.pi
    hour_sin = math.sin(hour_rad)
    hour_cos = math.cos(hour_rad)

    # Map weather to binary columns
    weather_data = {
        "weather_clear_sky": 1 if weather == "clear_sky" else 0,
        "weather_scattered_clouds": 1 if weather == "scattered_clouds" else 0,
        "weather_overcast": 1 if weather == "overcast" else 0,
        "weather_light_rain": 1 if weather == "light_rain" else 0
    }

    # Create DataFrame for prediction
    input_data = {
        "barn_temperature": [barn_temperature],
        "humidity": [humidity],
        "luminance": [luminance],
        "weather_clear_sky": [weather_data["weather_clear_sky"]],
        "weather_scattered_clouds": [weather_data["weather_scattered_clouds"]],
        "weather_overcast": [weather_data["weather_overcast"]],
        "weather_light_rain": [weather_data["weather_light_rain"]],
        "hour_sin": [hour_sin],
        "hour_cos": [hour_cos]
    }
    input_df = pd.DataFrame(input_data)

    # Predict fan and lamp status
    fan_prediction = clf_fan_loaded.predict(input_df)[0]
    lamp_prediction = clf_lamp_loaded.predict(input_df)[0]

    # Render template with prediction results and previous inputs
    return render_template(
        "index.html",
        barn_temperature=barn_temperature,
        humidity=humidity,
        luminance=luminance,
        weather=weather,
        hour=hour,
        fan_prediction=fan_prediction,
        lamp_prediction=lamp_prediction,
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
