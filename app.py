import streamlit as st
import pandas as pd
import joblib
from google import genai
import os

API_KEY = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=API_KEY)

st.set_page_config(page_title="Agri-Advisor AI", page_icon="🌱", layout="wide")
st.title("🌱 Agricultural Advisor + Gemini AI Agent")

# Load ML Models and Encoders dynamically
@st.cache_resource
def load_assets():
    crop_m = joblib.load('crop_recommendation_model.pkl')
    macro_m = joblib.load('fertility_classifier_model.pkl')
    micro_m = joblib.load('soil_micro_fertility_model.pkl')

    # Load LabelEncoders if saved during training (fallback provided below)
    try:
        soil_enc = joblib.load('soil_encoder.pkl')
        photo_enc = joblib.load('photo_encoder.pkl')
    except Exception:
        soil_enc = None
        photo_enc = None

    return crop_m, macro_m, micro_m, soil_enc, photo_enc

crop_model, macro_model, micro_model, soil_encoder, photo_encoder = load_assets()

# Comprehensive Categorical Mappings
SOIL_TYPES = list(soil_encoder.classes_) if soil_encoder else ["Clay", "Sandy", "Loamy", "Black", "Red", "Alluvial", "Silty", "Peaty"]
PHOTOPERIODS = list(photo_encoder.classes_) if photo_encoder else ["Day Neutral", "Short Day", "Long Day"]
FERTILITY_LABELS = {0: 'Low', 1: 'Moderate', 2: 'High'}

# Form Inputs Layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Nutrients & pH")
    n = st.number_input("Nitrogen (N)", value=120.0)
    p = st.number_input("Phosphorus (P)", value=15.0)
    k = st.number_input("Potassium (K)", value=100.0)
    ph = st.number_input("pH Level", value=5.8)

with col2:
    st.subheader("Environment & Climate")
    temp = st.number_input("Temperature (°C)", value=27.0)
    humidity = st.number_input("Humidity (%)", value=75.0)
    rainfall = st.number_input("Rainfall (mm)", value=180.0)
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    photoperiod = st.selectbox("Photoperiod", PHOTOPERIODS)
    light_hours = st.number_input("Light Hours", value=12.0)
    light_intensity = st.number_input("Light Intensity", value=600.0)
    rh = st.number_input("RH (%)", value=75.0)

with col3:
    st.subheader("Micro/Biochemical")
    ec = st.number_input("EC (dS/m)", value=0.45)
    oc = st.number_input("Organic Carbon (OC %)", value=0.40)
    s = st.number_input("Sulphur (S)", value=6.0)
    zn = st.number_input("Zinc (Zn)", value=0.30)
    fe = st.number_input("Iron (Fe)", value=2.0)
    cu = st.number_input("Copper (Cu)", value=0.5)
    mn = st.number_input("Manganese (Mn)", value=5.0)
    b = st.number_input("Boron (B)", value=0.4)

if st.button("🚀 Analyze & Generate AI Plan", use_container_width=True):
    # Encode categorical variables for model inference
    if soil_encoder:
        encoded_soil = soil_encoder.transform([soil_type])[0]
    else:
        encoded_soil = SOIL_TYPES.index(soil_type)

    if photo_encoder:
        encoded_photo = photo_encoder.transform([photoperiod])[0]
    else:
        encoded_photo = PHOTOPERIODS.index(photoperiod)

    # 1. Crop Model Inputs
    # Pass raw user string selections directly into DataFrame
    crop_in = pd.DataFrame([{
        'N': n, 'P': p, 'K': k,
        'temperature': temp, 'humidity': humidity, 'ph': ph, 'rainfall': rainfall
    }])

    macro_in = pd.DataFrame([{
        'Nitrogen': n, 'Phosphorus': p, 'Potassium': k, 'pH': ph,
        'Temperature': temp, 'Rainfall': rainfall,
        'Soil_Type': soil_type,       # Pass raw string selection (e.g., "Loamy")
        'Photoperiod': photoperiod,   # Pass raw string selection (e.g., "Day Neutral")
        'Light_Hours': light_hours, 'Light_Intensity': light_intensity, 'Rh': rh
    }])

    micro_in = pd.DataFrame([{
        'N': n, 'P': p, 'K': k, 'pH': ph,
        'EC': ec, 'OC': oc, 'S': s, 'Zn': zn, 'Fe': fe, 'Cu': cu, 'Mn': mn, 'B': b
    }])

    try:
        # --- Top-3 crop recommendations with suitability % ---
        if hasattr(crop_model, "predict_proba"):
            probs = crop_model.predict_proba(crop_in)[0]
            classes = crop_model.classes_
            top_n = min(3, len(classes))
            top_idx = probs.argsort()[-top_n:][::-1]
            top_crops = [(str(classes[i]), round(float(probs[i]) * 100, 2)) for i in top_idx]
        else:
            # Model has no predict_proba (e.g. some SVM configs) — fall back to a single hard prediction
            top_crops = [(str(crop_model.predict(crop_in)[0]), None)]

        pred_crop = top_crops[0][0]  # top pick, used below for the Gemini plan

        pred_macro = macro_model.predict(macro_in)[0]
        pred_micro_raw = micro_model.predict(micro_in)[0]

        # Handle numeric vs string output mapping safely
        pred_micro = FERTILITY_LABELS.get(pred_micro_raw, str(pred_micro_raw))

        st.markdown("---")
        st.subheader("🌾 Top 3 Recommended Crops")
        crop_cols = st.columns(len(top_crops))
        for col, (crop_name, conf) in zip(crop_cols, top_crops):
            col.metric(crop_name, f"{conf}% suitable" if conf is not None else "—")

        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.metric("Macro Fertility", pred_macro)
        c2.metric("Micro Fertility", pred_micro)

        # Gemini Agent Integration
        top_crops_str = ", ".join(
            f"{crop} ({conf}% suitability)" if conf is not None else crop
            for crop, conf in top_crops
        )

        prompt = f"""
        You are an expert agricultural science advisor.

        Diagnostics Summary:
        - Top 3 Recommended Crops (by suitability): {top_crops_str}
        - Macro Fertility Status: {pred_macro}
        - Micro Fertility Status: {pred_micro}

        Measured Field Parameters:
        - N-P-K Ratio: {n}-{p}-{k}
        - pH Level: {ph} | Organic Carbon (OC): {oc}%
        - Micronutrients: Zn={zn} ppm, Fe={fe} ppm, B={b} ppm, S={s} ppm
        - Environmental: Temp={temp}°C, Rainfall={rainfall}mm, Soil Type={soil_type}

        Provide an actionable advisory report containing:
        1. A short comparison of the 3 candidate crops to help the farmer choose.
        2. Specific fertilizer dosage and organic soil amendment recommendations for
           the top-ranked crop, {pred_crop}.
        3. Remediation steps for low macro/micro nutrients or pH imbalances.
        4. Management guidance to maximize yield for {pred_crop}.
        """

        with st.spinner("Generating Agronomist Action Plan via Gemini AI..."):
            response = client.models.generate_content(
                model='gemini-3.6-flash',
                contents=prompt
            )
            st.markdown("### 📋 AI Agronomist Action Plan")
            st.markdown(response.text)

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")