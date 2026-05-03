# 🌾 Tamil Nadu Crop Price Intelligence Dashboard

AI-powered crop price prediction system for all districts of Tamil Nadu.
Built with Python, Flask, and Random Forest ML.

---

## 📁 Project Structure

```
tamil_crop_price/
├── data/
│   └── crop_prices.csv        ← Dataset (25 districts, 10+ crops, 4 seasons)
├── models/                    ← Auto-created after training
│   ├── crop_price_model.pkl
│   └── model_metadata.json
├── templates/
│   └── index.html             ← Full UI dashboard
├── app.py                     ← Flask web server
├── train_model.py             ← ML training script
├── requirements.txt
└── README.md
```

---

## 🚀 Setup Instructions (VS Code)

### Step 1 — Open the folder in VS Code
```
File → Open Folder → select tamil_crop_price
```

### Step 2 — Open the integrated terminal
```
Terminal → New Terminal   (or Ctrl + `)
```

### Step 3 — Create a virtual environment (recommended)
```bash
python -m venv venv
```
Activate it:
- **Windows**: `venv\Scripts\activate`
- **Mac/Linux**: `source venv/bin/activate`

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Train the ML model (REQUIRED before running the app)
```bash
python train_model.py
```
This will:
- Load `data/crop_prices.csv`
- Train a Random Forest Regressor
- Print accuracy (will be below 92%)
- Save model to `models/crop_price_model.pkl`

### Step 6 — Start the web app
```bash
python app.py
```

### Step 7 — Open in browser
```
http://127.0.0.1:5000
```

---

## 🌦️ Features

| Feature | Description |
|--------|-------------|
| 🌤️ Live Weather | Real-time weather for selected district |
| 🌾 Crop Prices | Price per KG for all crops by district & season |
| 🤖 ML Prediction | AI price prediction using Random Forest |
| 💬 Chatbot | Ask questions about prices, crops, seasons |
| 📊 Stats | Min, max, average prices dashboard |

---

## 📦 Dataset Info

- **File**: `data/crop_prices.csv`
- **Districts**: 25 Tamil Nadu districts
- **Crops**: Tomato, Rice, Onion, Potato, Banana, Mango, Coconut, Turmeric, Groundnut, Sugarcane, Cashew, Brinjal
- **Seasons**: Summer, Rainy, Light Rain, Normal
- **Features**: temperature, humidity, rainfall, demand, supply
- **Target**: price_per_kg (NOT per quintal)

---

## 🤖 ML Model

- **Algorithm**: Random Forest Regressor
- **Target Accuracy**: Below 92%
- **Features used**: district, crop, season, temperature, humidity, rainfall, demand, supply
- **Training split**: 80% train / 20% test

---

## ⚠️ Notes

- Make sure to run `train_model.py` before `app.py`
- The weather API uses free OpenWeatherMap data; internet connection needed for live weather
- If weather API fails, the system uses fallback demo data automatically
- All prices shown are per KG

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| Model not found error | Run `python train_model.py` first |
| Port already in use | Change port in `app.py`: `app.run(port=5001)` |
| Weather not loading | Check internet connection; fallback data will show |
