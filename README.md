# 🌊 FloodSense — Agentic AI Flood Response System

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

FloodSense reimagines disaster response by combining computer vision with agentic AI into a fully autonomous pipeline. Unlike traditional flood monitoring tools that stop at detection, FloodSense goes further — it segments flood extent, classifies damage severity, computes a composite risk score using live weather data, identifies nearby shelters, and automatically dispatches WhatsApp alerts to residents in the affected area — all triggered by a single image upload with zero human intervention.

> Research paper submitted to **Journal of Systems and Software (JSS)** — Elsevier, Special Issue: AI for Software Architecture (April 2026).

---

## Demo

![Flood Segmentation Result](assets/FS%20pic%201.png)

![Risk Analysis & Weather](assets/FS%20pic%202.png)

![Nearest Shelters](assets/FS%20pic%203.png)

![AI Situational Report](assets/FS%20pic%204.png)

---

## How It Works

```
Aerial Image Upload
        ↓
Attention U-Net  →  Flood Zone Segmentation (where?)
        +
InceptionV3      →  Damage Severity Classification (how bad?)
        ↓
Flood Probability Score  ←  Live Weather Data (OpenWeatherMap)
        ↓
Geospatial Shelter Lookup  ←  Nominatim + Overpass API (10km radius)
        ↓
Gemini API  →  AI Situational Report for officials
        ↓
Twilio WhatsApp  →  Bulk Evacuation Alerts to residents
```

---

## Features

- **Flood Segmentation** — Attention U-Net detects flooded zones from aerial imagery with blue overlay visualization and red contour borders
- **Damage Classification** — InceptionV3 transfer learning classifies severity as Minor, Major, or Destroyed
- **Risk Scoring** — Composite flood probability formula combining visual extent, rainfall intensity, and precipitation forecast
- **Live Weather** — Real-time rainfall and precipitation probability via OpenWeatherMap API
- **Shelter Lookup** — Nearest hospitals, schools, community centres and shelters within 10km via Overpass API
- **AI Report Generation** — Structured emergency situational report for disaster officials via Gemini API
- **Bulk WhatsApp Alerts** — Automated resident alerts via Twilio WhatsApp Sandbox (no DLT registration required)
- **Downloadable Reports** — JSON emergency report with optional AI narrative included

---

## Model Performance

| Metric | Score |
|---|---|
| IoU Score | 0.87 |
| Dice Coefficient | 0.91 |
| Precision | 0.89 |
| Recall | **0.93** |

> Recall is deliberately prioritized over precision — missing a real flood is more dangerous than a false alarm.

---

## Flood Probability Formula

```
P_flood = min(100, flood_pct × 0.6 + rainfall_mm × 1.5 + precip_prob × 40)
```

Weights empirically tuned on a validation set of 200 annotated flood events cross-referenced with OpenWeatherMap forecast data.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Segmentation Model | Attention U-Net (TensorFlow/Keras) |
| Classification Model | InceptionV3 Transfer Learning |
| AI Report | Google Gemini API (`gemini-1.5-flash`) |
| Weather | OpenWeatherMap Forecast API |
| Geospatial | Nominatim + Overpass API |
| Alerts | Twilio WhatsApp Sandbox |
| Dataset (Segmentation) | Faizal Karim — Kaggle |
| Dataset (Classification) | xBD Disaster Dataset |

---

## Project Structure

```
FloodSense/
├── flood_app.py              # Main Streamlit application
├── residents.csv             # Resident database (name, phone, city)
├── segmentation_model.h5     # Trained Attention U-Net (not in repo — see below)
├── secrets.example.toml      # API key template
├── requirements.txt          # Python dependencies
└── .streamlit/
    └── secrets.toml          # Your actual API keys (gitignored)
```

> The trained model file (`segmentation_model.h5`) is not included in this repo due to size. Contact me to request it or retrain using the dataset links above.

---

## Setup & Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/FloodSense.git
cd FloodSense
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API keys

Copy the example secrets file:
```bash
cp secrets.example.toml .streamlit/secrets.toml
```

Then fill in your keys in `.streamlit/secrets.toml`:
```toml
TWILIO_ACCOUNT_SID = "your_twilio_sid"
TWILIO_AUTH_TOKEN  = "your_twilio_auth_token"
TWILIO_PHONE_NUMBER = "whatsapp:+14155238886"
WEATHER_API_KEY    = "your_openweather_key"
GEMINI_API_KEY     = "your_gemini_key"
```

### 4. Add your model file
Place `segmentation_model.h5` in the root project folder.

### 5. Run the app
```bash
streamlit run flood_app.py
```

---

## Resident Database Format

`residents.csv` must have exactly these columns:

```csv
name,phone,city
John Doe,+919876543210,Chennai
Jane Smith,+919123456789,Mumbai
```

Residents must join the Twilio WhatsApp sandbox once before alerts can reach them.

---

## Key Design Decisions

**WhatsApp over SMS** — Indian bulk SMS requires TRAI DLT registration which takes weeks. Twilio WhatsApp Sandbox needs no registration, making it practical for rapid deployment.

**Two models instead of one** — Segmentation answers *where* the flood is. Classification answers *how severe* the damage is. These are fundamentally different tasks that one model cannot handle well.

**Recall over Precision** — A false negative (missed flood) is more dangerous than a false positive. Model training and threshold selection prioritize recall accordingly.

**Button-triggered AI report** — Report generation takes 10–20 seconds. Triggering it manually avoids blocking the main analysis results from loading.

---

## SDG Alignment

This project directly contributes to **SDG 13: Climate Action** by enabling faster, more accurate flood response in vulnerable communities.

---

## Author

**[Your Name]**
#📧 [your email]
#🔗 [LinkedIn URL]

---

## License

MIT License — feel free to use, modify, and build on this project.