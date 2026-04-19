# 🌿 MULTI WEED DETECTION

AI-Powered Detection System for Banana Weed, Paddy Weed, and Sugarcane Weed. Designed primarily as a final year college project to analyze and monitor agricultural environments.

## ✨ Features
- **Premium User Interface:** Uses custom CSS styling with a beautiful dark mode "Agritech / Honeycomb" design theme to elevate the Streamlit baseline.
- **Deep Learning Integrated:** Powered by Ultralytics YOLOv8 inference capability to automatically analyze ecological boundaries.
- **Dynamic Entity Mapping:** Specifically highlights **Banana Weed** 🍌, **Paddy Weed** 🌾 and **Sugarcane Weed** 🌱. 
- **Interactive UI:** Dynamic threshold adjustment, real-time image analysis display, and detailed inference analytics cards.

## 🚀 Deployment Instructions

### Local Development / Running the App

1. Ensure Python 3.8+ is installed on your environment.
2. Install the necessary dependencies into your virtual environment:

```bash
pip install -r requirements.txt
```

3. Ensure the core YOLO deep learning model `best.pt` is inside the root directory.

4. Run the Streamlit User Interface:

```bash
streamlit run app.py
```

### Streamlit Community Cloud Deployment

If you wish to deploy this on the web via Streamlit Community Cloud:

1. Push this entire repository (`app.py`, `requirements.txt`, and `best.pt`) to GitHub.
2. Visit [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub account and specify `app.py` as the main app path.
4. Streamlit will automatically read `requirements.txt` and build the container! Enjoy your deployed app!

## 📸 Screenshots & Usage
Simply upload a `.jpg`, `.jpeg`, or `.png` input capture from the application dashboard and press **SCAN ENVIRONMENT**. The app will handle bounding box plotting and output dynamic metric calculations.
