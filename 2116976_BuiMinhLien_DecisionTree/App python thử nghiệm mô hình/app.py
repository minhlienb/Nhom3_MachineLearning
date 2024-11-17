from flask import Flask, render_template_string, request
import pickle
import pandas as pd
import math
import os  # Import os for environment variables

# Load models
model_path = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
with open(os.path.join(model_path, "clf_fan_model.pkl"), "rb") as fan_file:
    clf_fan_loaded = pickle.load(fan_file)
with open(os.path.join(model_path, "clf_lamp_model.pkl"), "rb") as lamp_file:
    clf_lamp_loaded = pickle.load(lamp_file)

# Initialize Flask app
app = Flask(__name__)

# HTML Template for the web form
html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fan and Lamp Prediction</title>
</head>
<body>
    <h1>Fan and Lamp Prediction Form</h1>
    <form id="predictionForm" method="post" action="/predict">
        <label for="barn_temperature">Barn Temperature (10-36°C):</label>
        <input type="number" id="barn_temperature" name="barn_temperature" min="10" max="45" step="0.1" value="{{ barn_temperature }}" required><br><br>

        <label for="humidity">Humidity (20-99%):</label>
        <input type="number" id="humidity" name="humidity" min="20" max="99" value="{{ humidity }}" required><br><br>

        <label for="luminance">Luminance (2-400 cd/m²):</label>
        <input type="number" id="luminance" name="luminance" min="2" max="400" value="{{ luminance }}" required><br><br>

        <fieldset>
            <legend>Weather</legend>
            <label><input type="radio" name="weather" value="clear_sky" {% if weather == 'clear_sky' %}checked{% endif %}> Clear Sky</label>
            <label><input type="radio" name="weather" value="scattered_clouds" {% if weather == 'scattered_clouds' %}checked{% endif %}> Scattered Clouds</label>
            <label><input type="radio" name="weather" value="overcast" {% if weather == 'overcast' %}checked{% endif %}> Overcast</label>
            <label><input type="radio" name="weather" value="light_rain" {% if weather == 'light_rain' %}checked{% endif %}> Light Rain</label>
        </fieldset>
        <br>

        <label for="hour">Hour of Day (0-23):</label>
        <input type="number" id="hour" name="hour" min="0" max="23" value="{{ hour }}" required><br><br>

        <button type="submit">Predict</button>
    </form>

    {% if fan_prediction is not none and lamp_prediction is not none %}
        <h2>Prediction Results:</h2>
        <p><strong>Ventilation Fan:</strong> {{ fan_prediction }}</p>
        <p><strong>Heating Lamp:</strong> {{ lamp_prediction }}</p>
    {% endif %}
</body>
</html>
"""

# Define route for main page
@app.route("/", methods=["GET"])
def home():
    return render_template_string(
        html_template,
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
    return render_template_string(
        html_template,
        barn_temperature=barn_temperature,
        humidity=humidity,
        luminance=luminance,
        weather=weather,
        hour=hour,
        fan_prediction=fan_prediction,
        lamp_prediction=lamp_prediction,
    )

# Entry point for Vercel
def handler(event, context):
    return app(event, context)

