import streamlit as st
import pandas as pd
import joblib
import time
from google import genai
import os

# Initialize Gemini Client safely
API_KEY = os.getenv("GEMINI_API_KEY", "Key")
client = genai.Client(api_key=API_KEY)

MODELS_TO_TRY = ("gemini-3.6-flash", "gemini-3.5-flash", "gemini-flash-latest")
TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "overloaded", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL")


def generate_with_retry_and_fallback(prompt, max_attempts=3):
    """Retry transient errors with backoff; move to the next model on repeated failure."""
    last_error = None
    for model_name in MODELS_TO_TRY:
        for attempt in range(max_attempts):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                text = (response.text or "").strip()
                if text:
                    return text, model_name
                last_error = RuntimeError(f"{model_name} returned an empty response.")
                break
            except Exception as exc:
                last_error = exc
                if any(marker in str(exc) for marker in TRANSIENT_MARKERS):
                    time.sleep(2 ** attempt)  # 1s, 2s, 4s
                    continue
                break  # permanent error for this model -> try the next one
    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")


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
            try:
                plan_text, used_model = generate_with_retry_and_fallback(prompt)
            except Exception as e:
                plan_text = None
                st.error(
                    f"The AI plan couldn't be generated right now (Gemini is likely overloaded): {e}\n\n"
                    "Your crop and fertility predictions above are still valid — you can retry by clicking "
                    "Analyze again, or check https://status.cloud.google.com/ for Gemini API status."
                )

        # Persist everything: Streamlit reruns the whole script on every widget
        # event (including a chat message), so results kept only in local
        # variables disappear the moment the chatbox below is used.
        st.session_state["analysis"] = {
            "top_crops": top_crops,
            "pred_macro": pred_macro,
            "pred_micro": pred_micro,
            "plan_text": plan_text,
            "soil_context": (
                f"Top 3 Recommended Crops (by suitability): {top_crops_str}\n"
                f"Macro Fertility Status: {pred_macro}\n"
                f"Micro Fertility Status: {pred_micro}\n"
                f"N-P-K: {n}-{p}-{k} | pH: {ph} | Organic Carbon: {oc}%\n"
                f"Micronutrients: Zn={zn} ppm, Fe={fe} ppm, Cu={cu} ppm, Mn={mn} ppm, B={b} ppm, S={s} ppm, EC={ec} dS/m\n"
                f"Environment: Temp={temp}°C, Humidity={humidity}%, Rainfall={rainfall}mm, "
                f"Soil Type={soil_type}, Photoperiod={photoperiod}, Light Hours={light_hours}, "
                f"Light Intensity={light_intensity}, Relative Humidity={rh}%"
            ),
        }

    except Exception as e:
        st.error(f"Inference Error: {str(e)}")

# --- Render results on every rerun, not just the click that produced them ---
analysis = st.session_state.get("analysis")
if analysis:
    st.markdown("---")
    st.subheader("🌾 Top 3 Recommended Crops")
    crop_cols = st.columns(len(analysis["top_crops"]))
    for col, (crop_name, conf) in zip(crop_cols, analysis["top_crops"]):
        col.metric(crop_name, f"{conf}% suitable" if conf is not None else "—")

    st.markdown("---")
    c1, c2 = st.columns(2)
    c1.metric("Macro Fertility", analysis["pred_macro"])
    c2.metric("Micro Fertility", analysis["pred_micro"])

    if analysis["plan_text"]:
        st.markdown("### 📋 AI Agronomist Action Plan")
        st.markdown(analysis["plan_text"])

# ---------------------------------------------------------
# Floating Chat Widget ("Ask me about your soil")
# Requires: pip install streamlit-float
# ---------------------------------------------------------
from streamlit_float import float_init

float_init()

if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Seed a greeting the first time the panel is opened
if st.session_state.chat_open and not st.session_state.chat_messages:
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": "Hi! 👋 How can I help you? Ask me about your soil results, fertilizer options, or alternative crops.",
    })

# Style the toggle button as a small circular icon
st.markdown("""
<style>
.st-key-chat_toggle_btn button {
    border-radius: 50% !important;
    width: 56px !important;
    height: 56px !important;
    font-size: 22px !important;
    background-color: #2e7d32 !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    padding: 0 !important;
}
.st-key-chat_panel {
    background: white;
    border-radius: 14px;
    box-shadow: 0 8px 28px rgba(0,0,0,0.28);
    padding: 14px 16px 8px 16px;
    border: 1px solid #e0e0e0;
}
.st-key-chat_panel_header {
    background-color: #2e7d32;
    color: white;
    padding: 10px 14px;
    border-radius: 10px;
    margin: -14px -16px 10px -16px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --- Floating toggle icon (bottom-right, always visible) ---
toggle_box = st.container(key="chat_toggle_box")
with toggle_box:
    icon_label = "✕" if st.session_state.chat_open else "💬"
    if st.button(icon_label, key="chat_toggle_btn"):
        st.session_state.chat_open = not st.session_state.chat_open
        st.rerun()
toggle_box.float("bottom: 24px; right: 24px; z-index: 9999;")

# --- Expandable chat panel ---
if st.session_state.chat_open:
    panel = st.container(key="chat_panel")
    with panel:
        st.markdown('<div class="st-key-chat_panel_header">🌱 Ask me about your soil</div>', unsafe_allow_html=True)

        history_box = st.container(height=300)
        with history_box:
            for msg in st.session_state.chat_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        user_query = st.chat_input("Ask about your soil...", key="floating_chat_input")

    panel.float(
        "bottom: 90px; right: 24px; width: 340px; max-height: 480px; z-index: 9998;"
    )

    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})

        soil_ctx = (analysis or {}).get("soil_context", "No field analysis has been run yet in this session.")
        plan_ctx = (analysis or {}).get("plan_text") or "No AI plan generated yet."

        # Include recent turns so follow-ups like "what about the second one?" resolve correctly
        history = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}"
            for m in st.session_state.chat_messages[-9:-1]
        )

        chat_prompt = f"""
        You are an agricultural advisor assistant helping a farmer act on a soil/crop analysis
        that has already been run. Answer the farmer's question directly and practically.

        Field Analysis Context:
        {soil_ctx}

        Previously Generated Advisory Plan:
        {plan_ctx}

        Conversation so far:
        {history}

        Guidance:
        - If asked about fertilizer brands, give general product *categories* and active-ingredient
          guidance (e.g., "a urea-based N fertilizer (46-0-0)" or "a balanced NPK 19-19-19 blend")
          rather than naming specific commercial brands, since availability varies heavily by region
          and endorsing a brand is not something you can verify. Suggest the farmer check with a
          local agricultural supply store or extension office for specific brands available to them.
        - If asked about alternative crops beyond the top 3, reason using the same field parameters
          (N, P, K, pH, soil type, rainfall, temperature, fertility levels) to suggest other
          agronomically reasonable options, and briefly explain why each could or couldn't work here.
        - Keep answers concise and actionable. Use the field context above rather than generic advice.

        Farmer's Question: {user_query}
        """

        try:
            bot_text, _ = generate_with_retry_and_fallback(chat_prompt)
            st.session_state.chat_messages.append({"role": "assistant", "content": bot_text})
        except Exception as e:
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": f"Sorry, I couldn't get a response right now ({e}). Please try again in a moment.",
            })
        st.rerun()
