from flask import Flask, render_template, request, jsonify
import pickle
import json
import numpy as np
import pandas as pd
import os
import requests
from datetime import datetime
 
app = Flask(__name__)
 
# Load model artifacts
MODEL_PATH = "models/crop_price_model.pkl"
META_PATH = "models/model_metadata.json"
 
if not os.path.exists(MODEL_PATH):
    print("❌ ERROR: Model not found! Please run 'python train_model.py' first.")
    artifacts = None
    metadata = {}
else:
    with open(MODEL_PATH, "rb") as f:
        artifacts = pickle.load(f)
    with open(META_PATH, "r") as f:
        metadata = json.load(f)
    print(f"✅ Model loaded. Accuracy: {metadata['accuracy']}%")
 
# Load dataset for reference
df = pd.read_csv("data/crop_prices.csv")
 
# District to lat/lon mapping for weather API
DISTRICT_COORDS = {
    "Chennai": (13.0827, 80.2707),
    "Coimbatore": (11.0168, 76.9558),
    "Madurai": (9.9252, 78.1198),
    "Salem": (11.6643, 78.1460),
    "Tiruchirappalli": (10.7905, 78.7047),
    "Tirunelveli": (8.7139, 77.7567),
    "Vellore": (12.9165, 79.1325),
    "Erode": (11.3410, 77.7172),
    "Thanjavur": (10.7870, 79.1378),
    "Dindigul": (10.3624, 77.9695),
    "Kanchipuram": (12.8185, 79.6947),
    "Villupuram": (11.9401, 79.4861),
    "Cuddalore": (11.7447, 79.7689),
    "Krishnagiri": (12.5186, 78.2137),
    "Dharmapuri": (12.1211, 78.1582),
    "Namakkal": (11.2189, 78.1677),
    "Tiruppur": (11.1085, 77.3411),
    "Thoothukudi": (8.7642, 78.1348),
    "Ramanathapuram": (9.3762, 78.8308),
    "Sivaganga": (9.8481, 78.4800),
    "Virudhunagar": (9.5851, 77.9624),
    "Pudukottai": (10.3797, 78.8202),
    "Thiruvarur": (10.7714, 79.6337),
    "Nagapattinam": (10.7672, 79.8449),
    "Tiruvannamalai": (12.2253, 79.0747),
}
 
# OpenWeatherMap API key — free tier (replace with your own key if needed)
WEATHER_API_KEY = "4d8fb5b93d4af21d66a2948a56317376"  # demo key
 
def get_weather(district):
    """Fetch live weather from OpenWeatherMap API."""
    if district not in DISTRICT_COORDS:
        return None
    lat, lon = DISTRICT_COORDS[district]
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={WEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        if data.get("cod") == 200:
            rain = data.get("rain", {}).get("1h", 0)
            # Classify season based on weather
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            if rain > 10:
                season = "Rainy"
            elif rain > 2:
                season = "Light Rain"
            elif temp > 35:
                season = "Summer"
            else:
                season = "Normal"
            return {
                "district": district,
                "temperature": round(data["main"]["temp"], 1),
                "feels_like": round(data["main"]["feels_like"], 1),
                "humidity": data["main"]["humidity"],
                "description": data["weather"][0]["description"].title(),
                "icon": data["weather"][0]["icon"],
                "wind_speed": round(data["wind"]["speed"] * 3.6, 1),  # m/s to km/h
                "rainfall": round(rain, 1),
                "pressure": data["main"]["pressure"],
                "visibility": data.get("visibility", 10000) // 1000,
                "season": season,
                "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
            }
    except Exception as e:
        print(f"Weather API error: {e}")
    # Fallback mock weather for demo
    import random
    season_map = {
        "Chennai": "Summer", "Coimbatore": "Normal", "Madurai": "Summer",
        "Salem": "Light Rain", "Thanjavur": "Rainy"
    }
    season = season_map.get(district, "Normal")
    return {
        "district": district,
        "temperature": round(32 + random.uniform(-5, 8), 1),
        "feels_like": round(35 + random.uniform(-4, 6), 1),
        "humidity": random.randint(55, 88),
        "description": "Partly Cloudy",
        "icon": "02d",
        "wind_speed": round(random.uniform(8, 25), 1),
        "rainfall": round(random.uniform(0, 5), 1),
        "pressure": random.randint(1005, 1015),
        "visibility": random.randint(8, 15),
        "season": season,
        "timestamp": datetime.now().strftime("%d %b %Y, %I:%M %p")
    }
 
 
def predict_price(district, crop, season, temperature, humidity, rainfall, demand, supply):
    """Run ML model prediction."""
    if artifacts is None:
        return None, "Model not loaded. Run train_model.py first."
    try:
        le_d = artifacts["le_district"]
        le_c = artifacts["le_crop"]
        le_s = artifacts["le_season"]
        model = artifacts["model"]
 
        if district not in le_d.classes_:
            return None, f"District '{district}' not in training data."
        if crop not in le_c.classes_:
            return None, f"Crop '{crop}' not in training data."
        if season not in le_s.classes_:
            return None, f"Season '{season}' not in training data."
 
        d_enc = le_d.transform([district])[0]
        c_enc = le_c.transform([crop])[0]
        s_enc = le_s.transform([season])[0]
 
        X = np.array([[d_enc, c_enc, s_enc, temperature, humidity, rainfall, demand, supply]])
        price = model.predict(X)[0]
        return round(price, 2), None
    except Exception as e:
        return None, str(e)
 
 
@app.route("/")
def index():
    districts = sorted(DISTRICT_COORDS.keys())
    crops = sorted(metadata.get("crops", df["crop"].unique().tolist()))
    seasons = ["Summer", "Rainy", "Light Rain", "Normal"]
    model_accuracy = metadata.get("accuracy", "N/A")
    return render_template("index.html",
                           districts=districts,
                           crops=crops,
                           seasons=seasons,
                           model_accuracy=model_accuracy)
 
 
@app.route("/api/weather/<district>")
def weather(district):
    data = get_weather(district)
    if data:
        return jsonify({"success": True, "weather": data})
    return jsonify({"success": False, "error": "Weather data unavailable"})
 
 
@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.json
    district = data.get("district")
    crop = data.get("crop")
    season = data.get("season")
    temperature = float(data.get("temperature", 30))
    humidity = float(data.get("humidity", 65))
    rainfall = float(data.get("rainfall", 20))
    demand = float(data.get("demand", 800))
    supply = float(data.get("supply", 750))
 
    price, error = predict_price(district, crop, season, temperature, humidity, rainfall, demand, supply)
    if error:
        return jsonify({"success": False, "error": error})
 
    # Get historical average for comparison (mean across all records for this combo)
    hist = df[(df["district"] == district) & (df["crop"] == crop) & (df["season"] == season)]
    hist_avg = round(float(hist["price_per_kg"].mean()), 2) if not hist.empty else price
    change = round(((price - hist_avg) / hist_avg) * 100, 1) if hist_avg else 0
 
    return jsonify({
        "success": True,
        "price": price,
        "historical_avg": hist_avg,
        "change_pct": change,
        "district": district,
        "crop": crop,
        "season": season,
        "accuracy": metadata.get("accuracy", "N/A")
    })
 
 
@app.route("/api/district_crops/<district>")
def district_crops(district):
    """Return crops available for a district with current season prices."""
    crops = df[df["district"] == district]["crop"].unique().tolist()
    return jsonify({"success": True, "crops": sorted(crops)})
 
 
@app.route("/api/all_prices/<district>")
def all_prices(district):
    """Return aggregated crop prices for a district for a given season (one row per crop)."""
    season = request.args.get("season", "Normal")
    filtered = df[(df["district"] == district) & (df["season"] == season)]
    prices = []
    if not filtered.empty:
        grouped = filtered.groupby("crop").agg(
            price=("price_per_kg", "mean"),
            demand=("demand", "mean"),
            supply=("supply", "mean")
        ).reset_index()
        for _, row in grouped.iterrows():
            prices.append({
                "crop": row["crop"],
                "price": round(row["price"], 2),
                "demand": int(row["demand"]),
                "supply": int(row["supply"])
            })
    return jsonify({"success": True, "prices": prices})
 
 
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    """Simple rule-based chatbot for crop price queries."""
    msg = request.json.get("message", "").lower().strip()
    response = chatbot_response(msg)
    return jsonify({"response": response})
 
 
def chatbot_response(msg):
    """Rule-based chatbot logic."""
    # Price queries
    for crop in df["crop"].str.lower().unique():
        if crop in msg:
            crop_title = crop.title()
            for district in df["district"].str.lower().unique():
                if district in msg:
                    district_title = district.title()
                    rows = df[(df["crop"].str.lower() == crop) &
                              (df["district"].str.lower() == district)]
                    if not rows.empty:
                        avg = rows["price_per_kg"].mean()
                        mn = rows["price_per_kg"].min()
                        mx = rows["price_per_kg"].max()
                        return (f"In {district_title}, {crop_title} prices range from "
                                f"₹{mn:.2f} to ₹{mx:.2f} per kg (average ₹{avg:.2f}/kg) "
                                f"across all seasons.")
            rows = df[df["crop"].str.lower() == crop]
            if not rows.empty:
                avg = rows["price_per_kg"].mean()
                mn = rows["price_per_kg"].min()
                mx = rows["price_per_kg"].max()
                return (f"{crop_title} prices across Tamil Nadu range from "
                        f"₹{mn:.2f} to ₹{mx:.2f} per kg, with an average of ₹{avg:.2f}/kg.")
 
    # Season queries
    season_map = {"summer": "Summer", "rainy": "Rainy", "light rain": "Light Rain", "normal": "Normal"}
    for s_key, s_val in season_map.items():
        if s_key in msg:
            rows = df[df["season"] == s_val]
            avg = rows["price_per_kg"].mean()
            return (f"During {s_val} season, average crop prices across Tamil Nadu "
                    f"are ₹{avg:.2f} per kg. Prices are generally "
                    f"{'higher due to low supply' if s_val == 'Summer' else 'lower due to high supply' if s_val == 'Rainy' else 'moderate'}.")
 
    # District queries
    for district in df["district"].str.lower().unique():
        if district in msg:
            district_title = district.title()
            crops_list = df[df["district"].str.lower() == district]["crop"].unique()
            return (f"{district_title} grows: {', '.join(sorted(crops_list))}. "
                    f"You can select {district_title} from the dashboard to see live weather and crop prices.")
 
    # Generic queries
    if any(w in msg for w in ["hello", "hi", "hey", "namaste"]):
        return "Hello! I'm your Tamil Nadu Crop Price Assistant. Ask me about crop prices by district, season, or crop name!"
 
    if any(w in msg for w in ["help", "what can you do", "how"]):
        return ("I can help you with:\n"
                "• Crop prices by district (e.g. 'tomato price in Chennai')\n"
                "• Prices by season (e.g. 'prices in summer')\n"
                "• Crops grown in a district (e.g. 'crops in Madurai')\n"
                "• Demand and supply info\n"
                "• Weather impact on prices")
 
    if any(w in msg for w in ["demand", "supply"]):
        return ("Demand and supply are key factors in crop pricing. "
                "High demand with low supply raises prices (common in Summer). "
                "High supply with moderate demand lowers prices (common in Rainy season). "
                "Use the Predict tab to enter custom demand/supply values.")
 
    if any(w in msg for w in ["model", "accuracy", "ml", "machine learning", "prediction"]):
        acc = metadata.get("accuracy", "N/A")
        return (f"The prediction model is a Random Forest Regressor trained on "
                f"{metadata.get('total_records', 0)} records. "
                f"Model accuracy: {acc}%. "
                f"It uses district, crop, season, temperature, humidity, rainfall, demand & supply as features.")
 
    if any(w in msg for w in ["best crop", "profitable", "highest price"]):
        top = df.groupby("crop")["price_per_kg"].mean().sort_values(ascending=False).head(3)
        result = ", ".join([f"{c} (₹{p:.2f}/kg)" for c, p in top.items()])
        return f"The highest-priced crops on average across Tamil Nadu are: {result}."
 
    if any(w in msg for w in ["cheapest", "lowest price", "affordable"]):
        bot = df.groupby("crop")["price_per_kg"].mean().sort_values().head(3)
        result = ", ".join([f"{c} (₹{p:.2f}/kg)" for c, p in bot.items()])
        return f"The most affordable crops on average are: {result}."
 
    if any(w in msg for w in ["districts", "how many", "list"]):
        districts = sorted(df["district"].unique())
        return f"This system covers {len(districts)} districts in Tamil Nadu: {', '.join(districts)}."
 
    return ("I didn't quite understand that. Try asking:\n"
            "• 'What is the price of rice in Thanjavur?'\n"
            "• 'Which crops are grown in Erode?'\n"
            "• 'What happens to prices in summer?'")
 
 
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" Tamil Nadu Crop Price Prediction Dashboard")
    print("=" * 60)
    if artifacts is None:
        print(" ⚠️  WARNING: Model not found!")
        print(" Run 'python train_model.py' first to train the model.")
    else:
        print(f" ✅ Model loaded — Accuracy: {metadata.get('accuracy')}%")
    print(" 🌐 Starting server at http://127.0.0.1:5000")
    print("=" * 60 + "\n")
    app.run(debug=True, port=5000)
 