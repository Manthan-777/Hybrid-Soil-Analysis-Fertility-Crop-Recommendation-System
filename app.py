import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import time
from google import genai
import os
import base64
from pathlib import Path

# Initialize Gemini Client safely
API_KEY = os.getenv("GEMINI_API_KEY", "key")
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


st.set_page_config(page_title="AgriSense | Soil Intelligence", page_icon="🌱", layout="wide", initial_sidebar_state="collapsed")

# -------------------- UI THEME + LANDING SCREEN --------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --forest:#174c35; --green:#287a50; --mint:#e9f5ec; --ink:#17352a; --muted:#60756a; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background: linear-gradient(180deg,#f7faf6 0%,#f1f7f1 100%); color:var(--ink); }
[data-testid="stMainBlockContainer"] { animation:dashboardIn .45s ease-out both; }
[data-testid="stHeader"] { background:rgba(247,250,246,.88); }
.block-container { padding-top:1.5rem; padding-bottom:4rem; max-width:1240px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:var(--ink); letter-spacing:-.035em; }
.stButton > button { border-radius:14px; min-height:3rem; font-weight:700; border:1px solid #d4e5d8; transition:all .2s ease; }
.stButton > button:hover { transform:translateY(-2px); border-color:#287a50; box-shadow:0 8px 20px rgba(23,76,53,.12); }
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#236b46,#174c35); color:white; border:0; }
[data-testid="stMetric"] { background:white; border:1px solid #e1ebe2; padding:18px 20px; border-radius:18px; box-shadow:0 6px 22px rgba(25,68,44,.045); }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }
[data-testid="stNumberInput"] input, [data-testid="stSelectbox"] > div { border-radius:11px; }
[data-testid="stMarkdownContainer"] p { line-height:1.65; }
.hero-wrap { position:relative; overflow:hidden; padding:clamp(30px,5.2vw,70px); border-radius:30px; background:radial-gradient(circle at 88% 12%,rgba(174,222,176,.35),transparent 28%),linear-gradient(135deg,#123d2b 0%,#1e6241 55%,#2d8054 100%); color:#fff; box-shadow:0 24px 60px rgba(20,70,44,.18); animation:heroIn .7s ease-out both; }
.hero-wrap:after { content:' '; position:absolute; width:320px;height:320px;right:-95px;bottom:-170px;border:1px solid rgba(255,255,255,.2);border-radius:50%;box-shadow:0 0 0 35px rgba(255,255,255,.035),0 0 0 70px rgba(255,255,255,.025); pointer-events:none; animation:orbitGlow 8s ease-in-out infinite alternate; }
.hero-wrap:before { content:'🌱'; position:absolute; right:7%; top:17%; font-size:clamp(50px,7vw,92px); opacity:.16; transform:rotate(-12deg); animation:floatLeaf 4s ease-in-out infinite; pointer-events:none; }
.st-key-open_dashboard button { position:relative; overflow:hidden; box-shadow:0 10px 24px rgba(23,76,53,.2); }
.st-key-open_dashboard button:after { content:' '; position:absolute; top:0; left:-80%; width:45%; height:100%; background:linear-gradient(100deg,transparent,rgba(255,255,255,.28),transparent); transform:skewX(-20deg); animation:buttonShine 3.8s ease-in-out infinite; }
.feature-card { position:relative; overflow:hidden; transition:transform .22s ease, box-shadow .22s ease, border-color .22s ease; }
.feature-card:before { content:''; position:absolute; inset:0 auto 0 0; width:4px; background:linear-gradient(180deg,#8ccf91,#287a50); opacity:.7; }
.st-key-feature_crop button, .st-key-feature_fertility button, .st-key-feature_advice button { margin-top:9px; background:rgba(232,244,233,.72); color:#20563a; border:1px solid #d1e5d4; }
.st-key-feature_crop button:hover, .st-key-feature_fertility button:hover, .st-key-feature_advice button:hover { background:#dff0e2; }
.st-key-feature_start button { box-shadow:0 8px 20px rgba(35,107,70,.18); }
.feature-card:hover { transform:translateY(-5px); box-shadow:0 15px 35px rgba(30,70,44,.11); border-color:#b9d8c0; }
.interaction-hint { color:#668174; font-size:12px; text-align:center; margin:10px 0 2px; }
.dashboard-background-note { display:none; }
.dashboard-progress { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:0 0 22px; }
.progress-step { display:flex; align-items:center; gap:10px; padding:12px 14px; border:1px solid rgba(209,228,212,.9); border-radius:14px; background:rgba(255,255,255,.76); color:#456554; font-size:13px; font-weight:700; backdrop-filter:blur(10px); }
.progress-num { display:flex; align-items:center; justify-content:center; flex:0 0 30px; height:30px; border-radius:10px; background:#e4f2e6; color:#236b46; font-weight:800; }
.input-panel-title { display:flex; align-items:center; gap:9px; margin-bottom:12px; }
.input-panel-subtitle { color:#6a8072; font-size:12px; margin-top:-5px; margin-bottom:14px; }
.results-banner { padding:18px 20px; border-radius:18px; border:1px solid rgba(197,224,202,.9); background:linear-gradient(110deg,rgba(255,255,255,.94),rgba(231,245,233,.90)); margin:12px 0 18px; }
@media(max-width:760px) { .dashboard-progress { grid-template-columns:1fr; gap:8px; } .hero-copy { font-size:15px; } .hero-wrap { border-radius:22px; } }
@keyframes floatLeaf { 0%,100% { transform:translateY(0) rotate(-12deg); } 50% { transform:translateY(-12px) rotate(-3deg); } }
@keyframes orbitGlow { from { opacity:.65; transform:scale(.98); } to { opacity:1; transform:scale(1.04); } }
@keyframes buttonShine { 0%,60% { left:-80%; } 85%,100% { left:140%; } }
.eyebrow { display:inline-block; padding:7px 12px; border-radius:999px; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2); color:#e3f4e6; font-size:12px; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
.hero-title { margin:20px 0 12px; max-width:720px; color:#fff; font:800 clamp(34px,5vw,58px)/1.08 'Manrope',sans-serif; letter-spacing:-.055em; }
.hero-copy { max-width:620px; color:#e0efe4; font-size:17px; line-height:1.75; }
.hero-stats { display:flex; flex-wrap:wrap; gap:10px; margin-top:28px; }
.hero-chip { padding:10px 13px; border-radius:12px; background:rgba(255,255,255,.09); border:1px solid rgba(255,255,255,.16); color:#f3fff5; font-size:13px; }
.landing-note { text-align:center; color:#718177; font-size:13px; margin-top:20px; }
.feature-card { height:100%; padding:20px; background:rgba(255,255,255,.85); border:1px solid #e0ebe1; border-radius:18px; box-shadow:0 8px 25px rgba(30,70,44,.035); }
.feature-icon { width:42px;height:42px;display:flex;align-items:center;justify-content:center;border-radius:13px;background:#e8f4e9;font-size:21px;margin-bottom:13px; }
.feature-title { font:700 16px 'Manrope',sans-serif; color:#1b4933; margin-bottom:5px; }
.feature-copy { color:#68796e; font-size:13px; line-height:1.55; }
.dashboard-head { padding:22px 24px; margin-bottom:22px; border:1px solid #dce9dd; border-radius:20px; background:linear-gradient(110deg,#fff,#f0f8f0); animation:dashboardIn .55s ease-out both; }
.section-kicker { color:#39805a; font-size:11px; font-weight:800; text-transform:uppercase; letter-spacing:.14em; }
@keyframes heroIn { from {opacity:0;transform:translateY(14px) scale(.99)} to {opacity:1;transform:translateY(0) scale(1)} }
@keyframes dashboardIn { from {opacity:0;transform:translateY(16px)} to {opacity:1;transform:translateY(0)} }
@media (prefers-reduced-motion: reduce) { *,*:before,*:after { animation:none !important; transition:none !important; } }
</style>
""", unsafe_allow_html=True)

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

if not st.session_state.show_dashboard:
    st.markdown("""
    <div class="hero-wrap">
      <div class="eyebrow">🌿 AGRISENSE &nbsp;·&nbsp; SMART AGRICULTURE</div>
      <div class="hero-title">Know your soil.<br><span style="color:#bde9c3">Grow with confidence.</span></div>
      <div class="hero-copy">Turn soil nutrients and field conditions into crop predictions, fertility insights, and practical next steps — all in one place.</div>
      <div class="hero-stats">
        <div class="hero-chip">🌱 Crop recommendations</div>
        <div class="hero-chip">🧪 Soil fertility analysis</div>
        <div class="hero-chip">✨ AI agronomist report</div><div class="hero-chip">↗ Guided analysis workflow</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    left, center, right = st.columns([1, 1.3, 1])
    with center:
        if st.button("🌱  Start my soil analysis   →", type="primary", use_container_width=True, key="open_dashboard"):
            st.session_state.show_dashboard = True
            st.rerun()
    st.markdown('<div class="landing-note">Your guided soil workspace is one step away · No changes to your analysis workflow</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown('<div class="interaction-hint">YOUR TOOLKIT  ·  Choose a feature to preview what you can do</div>', unsafe_allow_html=True)
    f1, f2, f3 = st.columns(3, gap="medium")
    with f1:
        st.markdown('<div class="feature-card"><div class="feature-icon">🌾</div><div class="feature-title">Crop insights</div><div class="feature-copy">See the top crop predictions and their model confidence scores.</div></div>', unsafe_allow_html=True)
        crop_detail = st.button("Explore crop insights  ↗", use_container_width=True, key="feature_crop")
    with f2:
        st.markdown('<div class="feature-card"><div class="feature-icon">🧪</div><div class="feature-title">Fertility snapshot</div><div class="feature-copy">Review macro and micro fertility classifications from your inputs.</div></div>', unsafe_allow_html=True)
        fertility_detail = st.button("Explore fertility  ↗", use_container_width=True, key="feature_fertility")
    with f3:
        st.markdown('<div class="feature-card"><div class="feature-icon">✨</div><div class="feature-title">Actionable guidance</div><div class="feature-copy">Get an AI-generated advisory report and ask follow-up questions.</div></div>', unsafe_allow_html=True)
        advice_detail = st.button("Explore AI guidance  ↗", use_container_width=True, key="feature_advice")

    if crop_detail:
        st.session_state["landing_feature"] = "crop"
    elif fertility_detail:
        st.session_state["landing_feature"] = "fertility"
    elif advice_detail:
        st.session_state["landing_feature"] = "advice"

    feature_info = {
        "crop": ("🌾 Crop predictions", "Your Random Forest model ranks the three most likely crop classes for the entered N, P, K, pH, temperature, humidity and rainfall. Percentages represent model confidence, not guaranteed suitability."),
        "fertility": ("🧪 Soil fertility", "The macro and micro fertility models provide separate classifications to help you understand possible nutrient-related concerns."),
        "advice": ("✨ AI agronomist guidance", "Gemini turns the model outputs and field measurements into practical guidance. Confirm fertilizer rates with a soil test and local agricultural expertise."),
    }
    selected_feature = st.session_state.get("landing_feature")
    if selected_feature in feature_info:
        title, detail = feature_info[selected_feature]
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(detail)
            if st.button("Start soil analysis →", type="primary", key="feature_start"):
                st.session_state.show_dashboard = True
                st.rerun()

    # Decorative Soil Buddy: cursor-reactive only (no greeting, no click action).
    # The iframe is positioned over the page; its script listens for pointer activity
    # in the parent document when the browser permits same-origin access.
    components.html(
        """
        <div id="soil-buddy-root" aria-hidden="true">
          <div class="soil-buddy">
            <svg class="buddy-svg" viewBox="0 0 180 205" role="presentation">
              <defs>
                <linearGradient id="buddyLeaf" x1="0" y1="0" x2="1" y2="1">
                  <stop offset="0%" stop-color="#b7ef72"/>
                  <stop offset="55%" stop-color="#65c84c"/>
                  <stop offset="100%" stop-color="#238044"/>
                </linearGradient>
                <linearGradient id="buddyPot" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stop-color="#956344"/>
                  <stop offset="100%" stop-color="#543623"/>
                </linearGradient>
                <radialGradient id="buddyFace" cx="35%" cy="25%" r="80%">
                  <stop offset="0%" stop-color="#efffb1"/>
                  <stop offset="100%" stop-color="#a7e65b"/>
                </radialGradient>
              </defs>
              <!-- Leaves -->
              <path d="M89 47 C49 4, 19 19, 24 43 C40 65, 66 61, 89 47Z" fill="url(#buddyLeaf)" stroke="#398b3d" stroke-width="2"/>
              <path d="M91 47 C119 4, 155 10, 157 33 C148 57, 118 62, 91 47Z" fill="url(#buddyLeaf)" stroke="#398b3d" stroke-width="2"/>
              <path d="M89 47 C79 12, 95 1, 105 8 C115 20, 105 39, 89 47Z" fill="#7fd44f" stroke="#398b3d" stroke-width="2"/>
              <path d="M89 47 L45 30 M93 46 L139 28 M91 44 L99 13" stroke="#c7f68a" stroke-width="2" fill="none" opacity=".7"/>
              <!-- Head -->
              <ellipse cx="91" cy="91" rx="59" ry="51" fill="url(#buddyFace)" stroke="#5aab42" stroke-width="2.5"/>
              <!-- Cheeks -->
              <ellipse cx="52" cy="105" rx="10" ry="5" fill="#f5a7a1" opacity=".65"/>
              <ellipse cx="130" cy="105" rx="10" ry="5" fill="#f5a7a1" opacity=".65"/>
              <!-- Eyes and pupils -->
              <g class="buddy-eye left-eye">
                <ellipse cx="70" cy="83" rx="12" ry="15" fill="#fff"/>
                <ellipse class="buddy-pupil" cx="72" cy="84" rx="7" ry="10" fill="#164a36"/>
                <circle cx="74" cy="80" r="2.5" fill="#fff"/>
              </g>
              <g class="buddy-eye right-eye">
                <ellipse cx="111" cy="83" rx="12" ry="15" fill="#fff"/>
                <ellipse class="buddy-pupil" cx="113" cy="84" rx="7" ry="10" fill="#164a36"/>
                <circle cx="115" cy="80" r="2.5" fill="#fff"/>
              </g>
              <!-- Brows -->
              <path d="M59 64 Q69 57 78 64 M100 64 Q111 57 121 64" stroke="#356d35" stroke-width="3" stroke-linecap="round" fill="none"/>
              <!-- Mouth: expressions are driven by classes -->
              <path class="buddy-mouth smile-mouth" d="M78 105 Q91 117 104 105" stroke="#285b35" stroke-width="3.5" stroke-linecap="round" fill="none"/>
              <path class="buddy-mouth open-mouth" d="M82 104 Q91 101 100 104 Q99 117 91 117 Q83 117 82 104Z" fill="#8c3d36" opacity="0"/>
              <!-- Pot -->
              <ellipse cx="91" cy="142" rx="43" ry="9" fill="#422c20"/>
              <path d="M49 140 L58 183 Q91 197 124 183 L133 140Z" fill="url(#buddyPot)" stroke="#4b3324" stroke-width="2"/>
              <ellipse cx="91" cy="140" rx="43" ry="10" fill="#b88654" stroke="#4b3324" stroke-width="2"/>
              <ellipse cx="91" cy="139" rx="35" ry="6" fill="#3d2b1d"/>
              <path d="M91 157 C77 146, 69 160, 82 167 C88 169, 91 164, 91 157Z" fill="#66c84b"/>
              <path d="M91 157 C105 145, 113 159, 100 167 C94 169, 91 164, 91 157Z" fill="#3c9f3d"/>
              <path d="M91 157 L91 173" stroke="#287b37" stroke-width="2"/>
            </svg>
          </div>
        </div>
        <style>
          html, body { margin:0; padding:0; background:transparent; overflow:visible; }
          #soil-buddy-root {
            position: fixed; right: 18px; bottom: 14px;
            width: 142px; height: 166px; z-index: 2147483000;
            pointer-events: none; overflow: visible;
          }
          .soil-buddy {
            width:100%; height:100%;
            transform-origin: 50% 100%;
            animation: buddy-float 3.4s ease-in-out infinite;
            filter: drop-shadow(0 8px 12px rgba(24,74,43,.18));
          }
          .buddy-svg { width:100%; height:100%; overflow:visible; }
          .buddy-pupil { transition: transform 140ms cubic-bezier(.2,.75,.25,1); }
          .buddy-eye { transform-box: fill-box; transform-origin: center; }
          .buddy-mouth { transition: opacity 120ms ease, transform 120ms ease; }
          .open-mouth { transform-origin: 91px 109px; }
          #soil-buddy-root.buddy-happy .smile-mouth { opacity:0; }
          #soil-buddy-root.buddy-happy .open-mouth { opacity:1; }
          #soil-buddy-root.buddy-surprised .buddy-eye { transform: scale(1.12); }
          #soil-buddy-root.buddy-surprised .smile-mouth { opacity:0; }
          #soil-buddy-root.buddy-surprised .open-mouth { opacity:1; transform:scale(1.18); }
          #soil-buddy-root.buddy-wink .right-eye { transform: scaleY(.12); }
          @keyframes buddy-float {
            0%,100% { transform:translateY(0) rotate(-1deg); }
            50% { transform:translateY(-6px) rotate(1deg); }
          }
          @keyframes buddy-blink {
            0%, 92%, 100% { transform:scaleY(1); }
            95% { transform:scaleY(.08); }
          }
          .buddy-eye { animation: buddy-blink 5.2s ease-in-out infinite; }
          .right-eye { animation-delay:.12s; }
          @media (max-width: 700px) {
            #soil-buddy-root { width:94px; height:112px; right:8px; bottom:8px; }
          }
          @media (prefers-reduced-motion: reduce) {
            .soil-buddy, .buddy-eye { animation:none !important; }
            .buddy-pupil, .buddy-mouth { transition:none !important; }
          }
        </style>
        <script>
          (() => {
            const root = document.getElementById('soil-buddy-root');
            if (!root) return;
            const pupils = root.querySelectorAll('.buddy-pupil');
            let lastReaction = 0;
            let parentDoc;
            try {
              parentDoc = window.parent.document;
              // Make this component iframe a fixed transparent overlay above the app.
              const frame = window.frameElement;
              if (frame) {
                frame.style.position = 'fixed';
                frame.style.right = '0';
                frame.style.bottom = '0';
                frame.style.width = '170px';
                frame.style.height = '190px';
                frame.style.zIndex = '2147482999';
                frame.style.border = '0';
                frame.style.background = 'transparent';
                frame.style.pointerEvents = 'none';
                frame.style.overflow = 'visible';
              }
              const moveEyes = (event) => {
                const x = (event.clientX / Math.max(parentDoc.documentElement.clientWidth, 1) - 0.5) * 2;
                const y = (event.clientY / Math.max(parentDoc.documentElement.clientHeight, 1) - 0.5) * 2;
                const dx = Math.max(-4, Math.min(4, x * 4));
                const dy = Math.max(-3, Math.min(3, y * 3));
                pupils.forEach((pupil) => {
                  pupil.style.transform = `translate(${dx}px, ${dy}px)`;
                });
              };
              parentDoc.addEventListener('mousemove', moveEyes, {passive:true});
              parentDoc.addEventListener('click', () => {
                const now = Date.now();
                if (now - lastReaction < 500) return;
                lastReaction = now;
                root.classList.remove('buddy-happy', 'buddy-surprised', 'buddy-wink');
                const reactions = ['buddy-surprised', 'buddy-happy', 'buddy-wink'];
                const reaction = reactions[Math.floor(Math.random() * reactions.length)];
                root.classList.add(reaction);
                window.setTimeout(() => root.classList.remove(reaction), 650);
              }, {passive:true});
            } catch (error) {
              // If the browser isolates the component iframe, the mascot remains animated.
            }
          })();
        </script>
        """,
        height=190,
        width=170,
        scrolling=False,
    )

    st.stop()

# Apply the supplied landscape image only on the dashboard screen. Keep a pale overlay
# so form labels, input fields and results remain easy to read.
BASE_DIR = Path(__file__).resolve().parent
background_path = BASE_DIR / "soil_background.png"

if background_path.is_file():
    background_b64 = base64.b64encode(
        background_path.read_bytes()
    ).decode("ascii")

    st.markdown(f"""
    <style>
    .stApp {{
        background-image:
            linear-gradient(
                rgba(246,250,246,.83),
                rgba(242,248,242,.88)
            ),
            url("data:image/png;base64,{background_b64}");

        background-size: cover;
        background-position: center center;
        background-attachment: fixed;
        background-repeat: no-repeat;
    }}

    [data-testid="stMainBlockContainer"] {{
        background: rgba(255,255,255,.08);
    }}

    [data-testid="stHorizontalBlock"] > [data-testid="column"] {{
        background: rgba(255,255,255,.66);
        border: 1px solid rgba(220,235,222,.72);
        border-radius: 20px;
        padding: 16px 14px 12px;
        box-shadow: 0 8px 26px rgba(23,76,53,.045);
        backdrop-filter: blur(10px);
    }}

    [data-testid="stHorizontalBlock"] > [data-testid="column"]
    [data-testid="stNumberInput"] input {{
        background: rgba(255,255,255,.92);
    }}

    .dashboard-head {{
        background: rgba(250,253,250,.88) !important;
        backdrop-filter: blur(8px);
    }}

    [data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,.88);
        backdrop-filter: blur(8px);
    }}

    [data-testid="stMetric"] {{
        background: rgba(255,255,255,.91);
    }}
    </style>
    """, unsafe_allow_html=True)

else:
    st.warning(
        f"Background image not found: {background_path.name}. "
        "Check that it exists beside app.py in your repository."
    )


# Dashboard entrance: this container fades in after the landing-screen action.
head_l, head_r = st.columns([5, 1])
with head_l:
    st.markdown('<div class="dashboard-head"><div class="section-kicker">AgriSense / Soil intelligence</div><h1 style="margin:6px 0 4px">Your soil analysis workspace</h1><div style="color:#64786b">Enter your field parameters below to generate crop predictions and an AI advisory.</div></div>', unsafe_allow_html=True)
with head_r:
    st.write("")
    if st.button("← Home", use_container_width=True, key="back_home"):
        st.session_state.show_dashboard = False
        st.rerun()

st.markdown("""
<div class="dashboard-progress">
  <div class="progress-step"><span class="progress-num">01</span><span>Enter field measurements</span></div>
  <div class="progress-step"><span class="progress-num">02</span><span>Review model predictions</span></div>
  <div class="progress-step"><span class="progress-num">03</span><span>Get AI guidance</span></div>
</div>
""", unsafe_allow_html=True)

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
st.markdown("### 🧭 Field measurements")
st.caption("Use your soil-test and local weather values where available. The model output is a prediction, not a substitute for a laboratory soil test.")
col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown("#### 🧪 Nutrients & pH")
    st.markdown('<div class="input-panel-subtitle">Core macronutrients and soil acidity</div>', unsafe_allow_html=True)
    n = st.number_input("Nitrogen (N)", value=120.0)
    p = st.number_input("Phosphorus (P)", value=15.0)
    k = st.number_input("Potassium (K)", value=100.0)
    ph = st.number_input("pH Level", value=5.8)

with col2:
    st.markdown("#### 🌤️ Environment & climate")
    st.markdown('<div class="input-panel-subtitle">Weather and growing conditions</div>', unsafe_allow_html=True)
    temp = st.number_input("Temperature (°C)", value=27.0)
    humidity = st.number_input("Humidity (%)", value=75.0)
    rainfall = st.number_input("Rainfall (mm)", value=180.0)
    soil_type = st.selectbox("Soil Type", SOIL_TYPES)
    photoperiod = st.selectbox("Photoperiod", PHOTOPERIODS)
    light_hours = st.number_input("Light Hours", value=12.0)
    light_intensity = st.number_input("Light Intensity", value=600.0)
    rh = st.number_input("RH (%)", value=75.0)

with col3:
    st.markdown("#### 🔬 Micro / biochemical")
    st.markdown('<div class="input-panel-subtitle">Secondary nutrients and trace elements</div>', unsafe_allow_html=True)
    ec = st.number_input("EC (dS/m)", value=0.45)
    oc = st.number_input("Organic Carbon (OC %)", value=0.40)
    s = st.number_input("Sulphur (S)", value=6.0)
    zn = st.number_input("Zinc (Zn)", value=0.30)
    fe = st.number_input("Iron (Fe)", value=2.0)
    cu = st.number_input("Copper (Cu)", value=0.5)
    mn = st.number_input("Manganese (Mn)", value=5.0)
    b = st.number_input("Boron (B)", value=0.4)

if st.button("✨ Analyze soil & generate AI plan", type="primary", use_container_width=True):
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
            f"{crop} ({conf}% model confidence)" if conf is not None else crop
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
                f"Top 3 Model Predictions: {top_crops_str}\n"
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
    st.markdown('<div class="results-banner"><div class="section-kicker">Analysis results</div><h3 style="margin:5px 0 3px">Your soil intelligence snapshot</h3><div style="color:#61796a">Review the model outputs first, then use the AI advisory for practical next steps.</div></div>', unsafe_allow_html=True)
    st.subheader("🌾 Top 3 model predictions")
    st.caption("Percentages show model confidence; they are not independently validated agronomic suitability scores.")
    crop_cols = st.columns(len(analysis["top_crops"]))
    for col, (crop_name, conf) in zip(crop_cols, analysis["top_crops"]):
        col.metric(crop_name, f"{conf}% model confidence" if conf is not None else "—")

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
