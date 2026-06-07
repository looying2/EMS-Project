import requests
import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
import joblib
import json
import os
import threading
import plotly.graph_objects as go
from datetime import datetime, timedelta
from collections import deque
import io
# import tensorflow as tf # Uncomment this when you have your actual model ready
import pyrebase
import easyocr
import re
from PIL import Image, ImageEnhance, ImageFilter

# ==========================================
# FIREBASE CONFIGURATION
# ==========================================
firebase_config = {
    "apiKey": "AIzaSyAOAXbZN8Q8P88QhWDYcW-qx4H0420-syA",
    "authDomain": "ems-project-7ea46.firebaseapp.com",
    "databaseURL": "https://ems-project-7ea46-default-rtdb.asia-southeast1.firebasedatabase.app",
    "projectId": "ems-project-7ea46",
    "storageBucket": "ems-project-7ea46.firebasestorage.app",
    "messagingSenderId": "793067450606",
    "appId": "1:793067450606:web:8b70fd1311aa6bf27003ff",
    "measurementId": "G-L2ZGPLSRGM"
}

firebase = pyrebase.initialize_app(firebase_config)
db = firebase.database()

FIREBASE_URL    = "https://ems-project-7ea46-default-rtdb.asia-southeast1.firebasedatabase.app"
FIREBASE_SECRET = "8nUsdEhjNt7hQkhnPcNPAPHKXVY9SRNOIDcnXITW"

def fb_write(path, data):
    url = f"{FIREBASE_URL}/{path}.json?auth={FIREBASE_SECRET}"
    try:
        r = requests.put(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"Firebase write error: {e}")
        return False

def fb_read(path):
    url = f"{FIREBASE_URL}/{path}.json?auth={FIREBASE_SECRET}"
    try:
        r = requests.get(url, timeout=5)
        return r.json()
    except Exception as e:
        print(f"Firebase read error: {e}")
        return None

def fb_push(path, data):
    url = f"{FIREBASE_URL}/{path}.json?auth={FIREBASE_SECRET}"
    try:
        r = requests.post(url, json=data, timeout=5)
        return r.status_code == 200
    except Exception as e:
        print(f"Firebase push error: {e}")
        return False

# ==========================================
# 1. PAGE CONFIGURATION & AESTHETICS
# ==========================================
st.set_page_config(
    page_title="AI-EMS Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #F0F2F6 0%, #E3F2FD 100%);
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: #31333F;
    }
    h1, h2, h3 {
        color: #264653;
        font-weight: 600;
    }
    .dashboard-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }
    div[data-testid="stMetric"] {
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        background-color: #FFFFFF;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #EEEEEE;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #78909C;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #2A9D8F;
        font-weight: 700;
    }
    button[data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 400;
        color: #607D8B;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        font-size: 18px !important;
        font-weight: 700 !important;
        background-color: #E3F2FD !important;
        color: #264653 !important;
        border-radius: 8px 8px 0 0;
    }
    div[data-testid="column"]:nth-of-type(5) div.stButton > button {
        background-color: #2E8B57; 
        color: white; 
        border: none;
    }
    div[data-testid="column"]:nth-of-type(5) div.stButton > button:hover {
        background-color: #3CB371;
        transform: scale(1.05);
    }
    div[data-testid="column"]:nth-of-type(6) div.stButton > button {
        background-color: #FF8C00; 
        color: white; 
        border: none;
    }
    div[data-testid="column"]:nth-of-type(6) div.stButton > button:hover {
        background-color: #FFA500;
        transform: scale(1.05);
    }
    div[data-testid="column"]:nth-of-type(7) div.stButton > button {
        background-color: #D32F2F; 
        color: white; 
        border: none;
    }
    div[data-testid="column"]:nth-of-type(7) div.stButton > button:hover {
        background-color: #EF5350;
        transform: scale(1.05);
    }
    div[data-testid="column"]:nth-of-type(4) div.stButton > button {
        font-weight: 900 !important;
        font-size: 1.1em !important;
        text-transform: uppercase;
        box-shadow: 0 4px 6px rgba(239, 83, 80, 0.3);
        background-color: #EF5350 !important;
        color: white !important;
        border: none;
    }
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div,
    div[data-baseweb="base-input"],
    div[data-testid="stMultiSelect"] > div > div {
        background-color: #F0F9FF !important;
        border: 1px solid #BAE6FD !important;
        border-radius: 8px !important;
    }
    input, select, div[data-baseweb="select"] span {
        color: #264653 !important;
        -webkit-text-fill-color: #264653 !important;
        font-weight: 600 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    .alert-box {
        padding: 15px; border-radius: 8px; margin-bottom: 10px; border-left: 5px solid;
    }
    .alert-safe { background-color: #FFFFFF; border-color: #4CAF50; color: #1B5E20; }
    .alert-risk { background-color: #FFEBEE; border-color: #EF5350; color: #B71C1C; }
    .alert-info { background-color: #E3F2FD; border-color: #2196F3; color: #0D47A1; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATABASE & ML SETUP
# ==========================================
DB_PATH = "session_audit.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT,
            patient_id TEXT,
            event TEXT,
            details TEXT
        )
    """)
    conn.commit()
    conn.close()

def log_event(patient_id, event, details=""):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO audit_log (ts, patient_id, event, details) VALUES (?, ?, ?, ?)",
        (datetime.now().isoformat(timespec="seconds"), patient_id, event, details)
    )
    conn.commit()
    conn.close()

def read_logs(patient_id, limit=200):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts, event, details FROM audit_log WHERE patient_id=? ORDER BY id DESC LIMIT ?",
        conn,
        params=(patient_id, limit)
    )
    conn.close()
    return df

init_db()

ML_API_URL = "https://escargot-coastline-tingling.ngrok-free.dev/api/start-monitoring"

def call_ml_api():
    """
    Remote ML backend reads latest Firebase EMG data automatically.
    Returns:
        prediction: str
        confidence_fraction: float, 0 to 1
        summary: dict
        latest: dict
        probabilities: list
        session: dict
    """

    try:
        response = requests.post(
            ML_API_URL,
            json={},
            timeout=10,
            verify=False
        )

        # Debugging
        print("ML status:", response.status_code)
        print("ML raw response:", response.text[:500])

        if response.status_code != 200:
            return f"API Error {response.status_code}", 0.0, {}, {}, [], {}

        try:
            data = response.json()
        except Exception:
            return "Invalid API Response", 0.0, {}, {}, [], {}

        prediction = data.get("prediction", "Unknown")

        confidence = float(data.get("confidence", 0.0))
        if confidence > 1:
            confidence = confidence / 100.0

        latest = data.get("latest", {})
        probabilities = data.get("probabilities", [])
        session = data.get("session", {})
        summary = data.get("summary", {})

        return prediction, confidence, summary, latest, probabilities, session

    except requests.exceptions.Timeout:
        return "API Timeout", 0.0, {}, {}, [], {}

    except requests.exceptions.ConnectionError:
        return "API Offline", 0.0, {}, {}, [], {}

    except Exception as e:
        return f"API Error: {str(e)}", 0.0, {}, {}, [], {}

# ==========================================
# 3. SESSION STATE
# ==========================================
def ss_init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

ss_init("elapsed_time", 0.0) 
ss_init("system_status", "IDLE")
ss_init("connected", True)
ss_init("session_start_time", None)
ss_init("intensity", 15)
ss_init("frequency", 40)
ss_init("pulse_width", 300)
ss_init("duty_on", 10)
ss_init("duty_off", 20)
ss_init("telemetry", pd.DataFrame(columns=["t", "emg", "hr", "imp"]))
ss_init("ml_window", None)
ss_init("ml_prediction", "WAITING")
ss_init("ml_probability", 0.0)
ss_init("live_pain", 2)
ss_init("live_fatigue", 4)
ss_init("esp_state", None)   # stores the latest state from ESP32
ss_init("esp_mode", None)
ss_init("esp_channel", None)
ss_init("ml_summary", {})
ss_init("session_summary_generated", False)
ss_init("session_summary_text", "")
ss_init("last_ml_call_time", 0)
ML_CALL_INTERVAL = 5  # seconds between ML API calls
SESSION_DURATION_MINUTES = 20  # fixed EMS protocol duration in minutes
ss_init("ml_latest", {})
ss_init("ml_summary", {})
ss_init("current_session_id", None)
ss_init("last_ml_summary_snapshot", {})

# Frozen snapshots — captured at STOP before the main loop clears ML state
ss_init("frozen_ml_prediction",    "WAITING")
ss_init("frozen_ml_probability",   0.0)
ss_init("frozen_ml_latest",        {})
ss_init("frozen_ml_session",       {})
ss_init("frozen_ml_probabilities", [])
ss_init("frozen_ml_summary",       {})
ss_init("frozen_intensity",        15)
ss_init("frozen_frequency",        40)
ss_init("frozen_pulse_width",      300)
ss_init("frozen_duty_on",          10)
ss_init("frozen_duty_off",         20)
ss_init("frozen_pain",             2)
ss_init("frozen_fatigue",          4)
ss_init("ml_probabilities", [])
ss_init("ml_session", {})
ss_init("ml_thread", None)       # background thread handle
ss_init("ml_pending", False)     # True while a thread is running

# Process-level shared state for background ML thread ↔ main loop communication.
# st.cache_resource survives all reruns and is guaranteed to return the same object.
@st.cache_resource
def _get_ml_buf():
    return {"result": {}, "thread": [None]}   # result dict + thread ref list

_ML_SHARED = _get_ml_buf()

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en'])

reader = load_easyocr()

def generate_session_summary():
    tele = st.session_state.telemetry
    if tele.empty:
        st.session_state.session_summary_text = {"error": "No telemetry data collected during this session."}
        return

    # ── Derived EMG stats ─────────────────────────────────────────────────
    avg_emg  = tele["emg"].mean()
    max_emg  = tele["emg"].max()
    min_emg  = tele["emg"].min()
    std_emg  = tele["emg"].std()
    q        = max(1, len(tele) // 4)
    emg_early = tele["emg"].iloc[:q].mean()
    emg_late  = tele["emg"].iloc[-q:].mean()
    emg_trend = emg_late - emg_early
    trend_desc = (
        f"increased by {emg_trend:.1f} µV (progressive recruitment)"
        if emg_trend > 10 else
        f"decreased by {abs(emg_trend):.1f} µV (possible fatigue or accommodation)"
        if emg_trend < -10 else
        f"remained stable (Δ{emg_trend:+.1f} µV)"
    )
    n_total   = len(tele)
    n_relaxed = (tele["emg"] <= 300).sum()
    n_mod     = ((tele["emg"] > 300) & (tele["emg"] <= 500)).sum()
    n_fat     = ((tele["emg"] > 500) & (tele["emg"] <= 700)).sum()
    n_over    = (tele["emg"] > 700).sum()
    pct_act   = (n_mod + n_fat + n_over) / n_total * 100

    # ── Read from frozen snapshot (set at STOP before values are wiped) ───
    intensity   = st.session_state.frozen_intensity
    frequency   = st.session_state.frozen_frequency
    pulse_width = st.session_state.frozen_pulse_width
    duty_on     = st.session_state.frozen_duty_on
    duty_off    = st.session_state.frozen_duty_off
    pain        = st.session_state.frozen_pain
    fatigue     = st.session_state.frozen_fatigue
    gait        = (st.session_state.ml_prediction
                   if st.session_state.ml_prediction not in ("WAITING", "")
                   else st.session_state.frozen_ml_prediction)
    ml_conf     = (st.session_state.ml_probability
                   if st.session_state.ml_probability > 0
                   else st.session_state.frozen_ml_probability)
    ml_latest   = st.session_state.ml_latest or st.session_state.frozen_ml_latest
    ml_session  = st.session_state.ml_session or st.session_state.frozen_ml_session
    ml_probs    = st.session_state.ml_probabilities or st.session_state.frozen_ml_probabilities
    ml_summary_snap = st.session_state.ml_summary or st.session_state.frozen_ml_summary

    conf_pct    = ml_conf * 100 if ml_conf <= 1.0 else ml_conf
    rms_val     = ml_latest.get("rms_recto_femoral", "N/A")
    spread_val  = ml_latest.get("rms_signal_spread", "N/A")
    std_val     = ml_latest.get("rms_signal_std", "N/A")
    ml_avg      = ml_session.get("avg", "N/A")
    ml_min      = ml_session.get("min", "N/A")
    ml_max      = ml_session.get("max", "N/A")

    st.session_state.last_ml_summary_snapshot = dict(ml_summary_snap)

    conditions = ", ".join(condition_tags) if condition_tags else "general rehabilitation"
    intensity_desc = (f"{intensity} mA" if intensity > 0
                      else "device-controlled current (intensity set by ESP32)")
    intensity_action = (
        f"set by the ESP32 controller — verify the actual delivered current level from the device log"
        if intensity == 0 else f"{intensity} mA"
    )

    # ── Parameter adjustment suggestions ─────────────────────────────────
    param_suggestions = []
    if gait not in ("Normal", "NORMAL", "WAITING", ""):
        param_suggestions.append(
            f"Gait classified as {gait} — consider reviewing electrode placement on the recto femoris "
            f"and adjusting frequency (current: {frequency} Hz; try 35–50 Hz range for better motor unit recruitment)."
        )
    else:
        param_suggestions.append(
            f"Gait classified as Normal — maintain current frequency ({frequency} Hz) and pulse width ({pulse_width} µs) for the next session."
        )
    if pct_act < 50:
        param_suggestions.append(
            f"Active recruitment was only {pct_act:.0f}% of session time (target >50%) — "
            f"consider increasing intensity from {intensity_action} by 5–10 mA if patient tolerance allows."
        )
    else:
        param_suggestions.append(
            f"Active recruitment was {pct_act:.0f}% of session time (target met) — "
            f"intensity ({intensity_action}) is appropriate; maintain for next session."
        )
    if fatigue >= 6:
        param_suggestions.append(
            f"Patient reported fatigue {fatigue}/10 (high) — reduce duty cycle ON-time "
            f"from {duty_on}s to {max(6, duty_on-2)}s, or increase OFF-time from {duty_off}s to {duty_off+5}s."
        )
    else:
        param_suggestions.append(
            f"Patient fatigue was {fatigue}/10 (acceptable) — maintain duty cycle ({duty_on}s ON / {duty_off}s OFF)."
        )
    if pain >= 5:
        param_suggestions.append(
            f"Patient reported pain {pain}/10 (exceeds comfort threshold of 5/10) — "
            f"reduce intensity by 10–20% from {intensity_action} at next session start, "
            f"and verify electrode placement is not over bony prominences."
        )
    else:
        param_suggestions.append(
            f"Patient pain was {pain}/10 (within acceptable range) — no intensity reduction required."
        )

    # ── Fallback dict ─────────────────────────────────────────────────────
    fallback = {
        "title": f"{gait} recto femoris EMG pattern — {SESSION_DURATION_MINUTES}-min EMS session ({conditions})",
        "summary": (
            f"A {SESSION_DURATION_MINUTES}-minute EMS session was completed for patient {patient_id} "
            f"(age group {age_group}, {conditions}) at {frequency} Hz, "
            f"{pulse_width} µs pulse width, {duty_on}s ON / {duty_off}s OFF duty cycle"
            f"{f', {intensity} mA stimulation intensity' if intensity > 0 else ''}. "
            f"The recto femoris EMG averaged {avg_emg:.1f} µV (max {max_emg:.1f}, min {min_emg:.1f}, "
            f"STD {std_emg:.1f} µV) across {n_total} samples. "
            f"Signal amplitude {trend_desc}. "
            f"Active muscle recruitment occurred in {pct_act:.0f}% of the session "
            f"({n_mod} moderate, {n_fat} fatigue-range, {n_over} overexertion samples). "
            f"The ML gait classifier returned {gait} at {conf_pct:.1f}% confidence, "
            f"with final patient-reported pain {pain}/10 and fatigue {fatigue}/10."
        ),
        "interpretation": [
            f"Recto Femoris RMS = {rms_val} µV (session avg/min/max: {ml_avg}/{ml_min}/{ml_max}): "
            f"signal level reflects muscle response to {intensity_desc}; "
            f"compare to patient baseline for age group {age_group} with {conditions}.",
            f"Signal Spread = {spread_val}, STD = {std_val}: "
            f"{'stable motor unit firing observed' if str(std_val) not in ('N/A', '') and float(str(std_val).replace('N/A','0') or 0) < 5 else 'notable signal variability — possible artefact or irregular recruitment'}. "
            f"Session EMG STD was {std_emg:.1f} µV; trend {trend_desc}.",
            f"Gait classification: {gait} ({conf_pct:.1f}% confidence). "
            f"Muscle state distribution — relaxed: {n_relaxed}, moderate: {n_mod}, "
            f"fatigue-range: {n_fat}, overexertion: {n_over} samples. "
            f"Active recruitment: {pct_act:.0f}% of session time."
        ],
        "actions": [
            param_suggestions[0],
            param_suggestions[1],
            param_suggestions[2],
            param_suggestions[3],
        ]
    }

    # ── Build prompt ──────────────────────────────────────────────────────
    probability_lines = [
        f"- {p.get('label', '?')}: {float(p.get('probability', 0)):.1f}%"
        for p in ml_probs
    ]
    feature_lines = [
        f"- Recto Femoris RMS: {rms_val} µV",
        f"- Signal Spread: {spread_val}",
        f"- Signal STD: {std_val}",
        f"- Session EMG avg: {avg_emg:.1f} µV, max: {max_emg:.1f} µV, min: {min_emg:.1f} µV",
        f"- Active recruitment: {pct_act:.0f}% of session",
        f"- Stimulation: {intensity} mA / {frequency} Hz / {pulse_width} µs / {duty_on}s ON {duty_off}s OFF",
        f"- Pain: {pain}/10, Fatigue: {fatigue}/10",
        f"- Patient: {patient_id}, Age group: {age_group}, Conditions: {conditions}",
    ]

    prompt = f"""You are a rehabilitation engineering assistant explaining an EMG monitoring result.
Use the draft explanation below as the factual base. Improve wording only if needed, but do not remove the actual values or make the response generic.
Draft explanation:
Title: {fallback["title"]}
Summary: {fallback["summary"]}
Interpretation:
- {fallback["interpretation"][0]}
- {fallback["interpretation"][1]}
- {fallback["interpretation"][2]}
Actions:
- {fallback["actions"][0]}
- {fallback["actions"][1]}
- {fallback["actions"][2]}
- {fallback["actions"][3]}
Model result:
Prediction: {gait}
Confidence: {conf_pct:.1f}%
Class probabilities:
{chr(10).join(probability_lines) if probability_lines else "Not available"}
Input measurements:
{chr(10).join(feature_lines) if feature_lines else "Not available"}
Rules:
- Do not diagnose.
- Do not claim certainty.
- Do not use the words "diagnosis", "diagnostic", or "disease" except in the disclaimer.
- The title must describe the EMG signal pattern.
- The summary must be 4 to 6 complete sentences.
- The interpretation list must contain exactly 3 points.
- The actions list must contain exactly 4 practical actions.
- Each interpretation point must mention the actual measurement value.
- Each action must state whether to increase, decrease, or maintain a specific parameter, and by how much.
- Do not use vague advice such as "monitor closely" unless you explain what to monitor and what threshold triggers action.
- Return only valid JSON.
- Do not use markdown.
JSON format:
{{
  "title": "specific EMG signal pattern title",
  "summary": "4 to 6 sentences explaining this specific result",
  "interpretation": ["value-specific point 1", "value-specific point 2", "value-specific point 3"],
  "actions": ["specific action 1", "specific action 2", "specific action 3", "specific action 4"]
}}"""

    answer, _ = call_rag_api(prompt)

    parsed = None
    if answer and "Error" not in answer:
        try:
            clean = answer.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except Exception:
            parsed = None

    st.session_state.session_summary_text = parsed if parsed else fallback

    # Save session to Firebase
    session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.current_session_id = session_id
    fb_write(f"patients/{patient_id}/sessions/{session_id}", {
        "date":              datetime.now().strftime("%Y-%m-%d"),
        "duration_min":      SESSION_DURATION_MINUTES,
        "avg_emg":           float(avg_emg),
        "max_emg":           float(max_emg),
        "pain_score":        pain,
        "fatigue_score":     fatigue,
        "ml_classification": gait,
        "ml_confidence":     float(conf_pct),
        "rag_summary":       json.dumps(parsed if parsed else fallback),
        "generated_at":      datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "approval_status":   "pending"
    })

def run_easyocr(uploaded_file):
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    results = reader.readtext(image_np)
    extracted_text = [res[1] for res in results]
    return extracted_text

def predict_muscle_state(emg_value):
    if emg_value > 700:
        return ("Overexertion", "🔴", "High muscle stress detected")
    elif emg_value > 500:
        return ("Muscle Fatigue", "🟠", "Sustained muscle activation observed")
    elif emg_value > 300:
        return ("Moderate Activity", "🟡", "Normal rehabilitation activity")
    else:
        return ("Relaxed", "🟢", "Low muscle activity")

smooth_buffer = deque(maxlen=10)
def smooth_emg(value):
    smooth_buffer.append(value)
    return np.mean(smooth_buffer)

def read_latest_emg_data():
    """
    Reads the latest EMG entry from Firebase.
    Returns a tuple: (emg_value, state, mode, channel) or (0.0, None, None, None)
    """
    try:
        url = "https://ems-project-7ea46-default-rtdb.asia-southeast1.firebasedatabase.app/emg_data.json?orderBy=\"$key\"&limitToLast=1"
        response = requests.get(url, timeout=5)
        data = response.json()
        if data:
            latest_key = list(data.keys())[-1]
            entry = data[latest_key]
            # EMG value: prefer "avg", then "emg", then last sample
            if "avg" in entry:
                emg = float(entry["avg"])
            elif "emg" in entry:
                emg = float(entry["emg"])
            elif "samples" in entry and len(entry["samples"]) > 0:
                emg = float(entry["samples"][-1])
            else:
                emg = 0.0
            state = entry.get("state", None)
            mode = entry.get("mode", None)
            channel = entry.get("channel", None)
            return emg, state, mode, channel
        return 0.0, None, None, None
    except Exception as e:
        print(f"Firebase read error: {e}")
        return 0.0, None, None, None

# Keep old read_emg for compatibility (but we'll update update_telemetry_stream)
def read_emg():
    emg, _ = read_latest_emg_data()
    return emg

RAG_API_URL = "https://escargot-coastline-tingling.ngrok-free.dev/api/ask"

def call_rag_api(question: str) -> tuple:
    if not question.strip():
        return "Please enter a question.", []
    payload = {"query": question}
    try:
        response = requests.post(RAG_API_URL, json=payload, timeout=30)
        response.raise_for_status()
        data = response.json()
        return data.get("answer", "No answer returned."), data.get("references", [])
    except requests.exceptions.Timeout:
        return "The AI service is taking too long. Please try again.", []
    except requests.exceptions.ConnectionError:
        return "Cannot reach the AI service. The ngrok tunnel may be offline.", []
    except Exception as e:
        return f"Error: {str(e)}", []

def update_telemetry_stream():
    df = st.session_state.telemetry.copy()
    now = datetime.now().strftime("%H:%M:%S")
    if st.session_state.system_status == "ACTIVE":
        raw_emg, esp_state, esp_mode, esp_channel = read_latest_emg_data()
        emg = smooth_emg(raw_emg)
        hr = int(np.clip(np.random.normal(74, 3), 60, 110))
        imp = float(np.clip(np.random.normal(1.2, 0.1), 0.7, 2.5))
        st.session_state.esp_state = esp_state
        st.session_state.esp_mode = esp_mode
        st.session_state.esp_channel = esp_channel
    else:
        emg = np.random.normal(0, 2)
        hr = int(np.random.normal(72, 2))
        imp = float(np.random.normal(1.2, 0.1))
        st.session_state.esp_state = None
        st.session_state.esp_mode = None
        st.session_state.esp_channel = None
    new_row = pd.DataFrame([{"t": now, "emg": emg, "hr": hr, "imp": imp}])
    df = pd.concat([df, new_row], ignore_index=True)
    st.session_state.telemetry = df.tail(200)

def generate_report(pid, mass_df, pain_df, fatigue_df):
    report_buffer = io.StringIO()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_buffer.write(f"RehaTech Clinical Progress Report\n")
    report_buffer.write(f"=================================\n")
    report_buffer.write(f"Patient ID: {pid}\n")
    report_buffer.write(f"Date Generated: {timestamp}\n\n")
    report_buffer.write(f"1. SESSION METRICS (Current)\n")
    report_buffer.write(f"----------------------------\n")
    report_buffer.write(f"Start Pain Score: {pain_df.iloc[0]['Pain Score']}\n")
    report_buffer.write(f"End Pain Score:   {pain_df.iloc[-1]['Pain Score']} (Improvement: {pain_df.iloc[0]['Pain Score'] - pain_df.iloc[-1]['Pain Score']})\n")
    report_buffer.write(f"End Fatigue Lvl:  {fatigue_df.iloc[-1]['Fatigue Level']}\n\n")
    report_buffer.write(f"2. MUSCLE MASS TREND (Last 10 Sessions)\n")
    report_buffer.write(f"-------------------------------------\n")
    recent_mass = mass_df.tail(5)
    for _, row in recent_mass.iterrows():
        date_str = row['Date'].strftime("%Y-%m-%d")
        report_buffer.write(f"{date_str}: {row['Muscle Mass (kg)']:.2f} kg\n")
    start_mass = mass_df.iloc[0]['Muscle Mass (kg)']
    current_mass = mass_df.iloc[-1]['Muscle Mass (kg)']
    change = current_mass - start_mass
    report_buffer.write(f"\nTotal Mass Gain: {change:+.2f} kg\n")
    report_buffer.write(f"\n=================================\n")
    report_buffer.write(f"End of Report\n")
    return report_buffer.getvalue()

def extract_metrics_from_text(text):
    cleaned = text.replace("_", ".").replace(",", ".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    metrics = {
        "Weight (kg)": None,
        "Skeletal Muscle Mass (kg)": None,
        "Body Fat Mass (kg)": None,
        "Body Fat Percentage": None,
        "BMI": None,
        "Mineral (kg)": None,
        "Visceral Fat Level": None,
        "SMI": None,
        "InBody Score": None,
        "Target Weight (kg)": None,
        "Weight Control (kg)": None,
        "Fat Control (kg)": None,
        "Muscle Control (kg)": None,
        "Patient ID": None,
        "Test Date": None
    }

    def get_float(patterns):
        for pattern in patterns:
            m = re.search(pattern, cleaned, re.IGNORECASE)
            if m:
                try:
                    value = m.group(1).replace(" ", "").replace(",", ".")
                    return float(value)
                except:
                    pass
        return None

    def get_int(patterns):
        val = get_float(patterns)
        return int(val) if val is not None else None

    # InBody Score
    metrics["InBody Score"] = get_int([
        r"(\d+)\s*n\s*100\s*1?\s*Pouts",
        r"(\d+)\s*/\s*100\s*Points",
        r"(\d+)\s*n?100\s*P",
        r"InBody Score.*?(\d+)\s*n\s*100",
    ])

    # Patient ID (11+ digits)
    id_match = re.search(r'\b(\d{11,})\b', cleaned)
    if id_match:
        metrics["Patient ID"] = id_match.group(1)

    # Test date: "07 . 01.,2026 12:49"
    date_match = re.search(r'(\d{2})\s*\.\s*(\d{2})\s*[\.,]?\s*,?\s*(\d{4})\s+(\d{2})\s*:\s*(\d{2})', text)
    if date_match:
        day = date_match.group(1)
        month = date_match.group(2)
        year = date_match.group(3)
        hour = date_match.group(4)
        minute = date_match.group(5)
        metrics["Test Date"] = f"{day}/{month}/{year} {hour}:{minute}"
    else:
        metrics["Test Date"] = None

    # Visceral Fat Level
    vfl_match = re.search(r'Visceral Fat Level.*?Letel\s*(\d+)', cleaned, re.IGNORECASE)
    if vfl_match:
        metrics["Visceral Fat Level"] = int(vfl_match.group(1))
    else:
        metrics["Visceral Fat Level"] = 5   # fallback based on user info

    # Control values (note the typos in OCR)
    metrics["Target Weight (kg)"] = get_float([r"Target Weight\s*(\d+\.?\d*)\s*kg"])
    metrics["Weight Control (kg)"] = get_float([r"Weight Contol\s*([+-]?\d+\.?\d*)\s*kg"])
    metrics["Fat Control (kg)"] = get_float([r"Fat\s*Cont(?:rol|ol)\s*([+-]?\d+\.?\d*)\s*kg"])
    metrics["Muscle Control (kg)"] = get_float([r"Muscle Contol\s*([+-]?\d+\.?\d*)\s*kg"])

    # Other metrics
    metrics["Mineral (kg)"] = get_float([r"Mineral\s*(?:\(kg\))?\s*(\d+\s*\.\s*\d+)"])
    metrics["Body Fat Mass (kg)"] = get_float([
        r"Body Fat Mass\s*(?:\(kg\))?\s*(\d+\s*\.\s*\d+)",
        r"Eody FatMass\s*(?:\(kg\))?\s*(\d+\s*\.\s*\d+)",
    ])
    metrics["Weight (kg)"] = get_float([
        r"Sum of the above\s*Weight\s*(?:\(kg\))?\s*(\d+\s*\.\s*\d+)",
        r"Muscle-Fat Analysis.*?Weight\s*(?:\(kg\))?\s*(\d+\s*\.\s*\d+)",
        r"Woight\s*(\d+\s*\.\s*\d+)",
    ])
    metrics["Skeletal Muscle Mass (kg)"] = get_float([r"(?:SMM|SmM|Skeletal Muscle Mass).*?(\d+\s*\.\s*\d+)"])
    metrics["BMI"] = get_float([
        r"Idn\s*\(kgm\s*\}\s*(\d+\s*\.\s*\d+)",
        r"Idn\s*\(kgm\s*\)\s*(\d+\s*\.\s*\d+)",
        r"BMI.*?Research Parameters.*?Idn\s*\(kgm\s*\}\s*(\d+\s*\.\s*\d+)",
    ])
    metrics["Body Fat Percentage"] = get_float([
        r"PBF.*?SMI\s*\d+\s*\.\s*\d+.*?(\d+\s*\.\s*\d+)",
        r"PBF.*?DJJen\s*(\d+\s*\.\s*\d+)",
        r"PBF.*?(\d+\s*\.\s*\d+)\s*Reomended",
    ])
    metrics["SMI"] = get_float([r"SMI\s*(\d+\s*\.\s*\d+)"])

    return metrics   

def generate_inbody_ai_insight(metrics):
    overall_insights = []
    risk_level = "LOW"
    smm = metrics.get("Skeletal Muscle Mass (kg)")
    pbf = metrics.get("Body Fat Percentage")
    bmi = metrics.get("BMI")
    vfl = metrics.get("Visceral Fat Level")
    score = metrics.get("InBody Score")
    smi = metrics.get("SMI")
    if smm is not None:
        if smm < 28:
            overall_insights.append(f"⚠️ Low skeletal muscle mass ({smm:.1f} kg) detected. Possible sarcopenia risk – consider resistance training and protein intake.")
            risk_level = "HIGH"
        elif smm < 32:
            overall_insights.append(f"🟡 Skeletal muscle mass ({smm:.1f} kg) is slightly below optimal range. Maintain regular strength exercise.")
            risk_level = "MEDIUM" if risk_level != "HIGH" else risk_level
        else:
            overall_insights.append(f"✅ Skeletal muscle mass ({smm:.1f} kg) is within healthy range. Continue current physical activity.")
    if pbf is not None:
        if pbf > 30:
            overall_insights.append(f"⚠️ Elevated body fat percentage ({pbf:.1f}%) detected. This may increase cardiovascular risk – consider dietary and exercise adjustments.")
            risk_level = "HIGH"
        elif pbf > 20:
            overall_insights.append(f"🟡 Moderate body fat percentage ({pbf:.1f}%). Aim to reduce to ≤20% for optimal metabolic health.")
            risk_level = "MEDIUM" if risk_level != "HIGH" else risk_level
        else:
            overall_insights.append(f"✅ Body fat percentage ({pbf:.1f}%) is within acceptable range. Well done.")
    if bmi is not None:
        if bmi < 18.5:
            overall_insights.append(f"⚠️ BMI = {bmi:.1f} indicates underweight. Consider nutritional assessment and weight gain strategies.")
        elif bmi > 25:
            overall_insights.append(f"⚠️ BMI = {bmi:.1f} indicates overweight/obesity. Weight management may reduce health risks.")
        else:
            overall_insights.append(f"✅ BMI = {bmi:.1f} is in the normal range. Maintain healthy lifestyle.")
    if vfl is not None:
        if vfl >= 10:
            overall_insights.append(f"⚠️ Visceral fat level = {vfl} (≥10) indicates increased metabolic risk. Reducing abdominal fat is recommended.")
            risk_level = "HIGH"
        else:
            overall_insights.append(f"✅ Visceral fat level = {vfl} is within healthy range. Good metabolic health indicator.")
    if score is not None:
        if score < 70:
            overall_insights.append(f"🟡 InBody Score = {score}/100 – below ideal. Focus on improving muscle mass and reducing body fat.")
        else:
            overall_insights.append(f"✅ InBody Score = {score}/100 – good body composition balance.")
    if smi is not None:
        if smi < 7:
            overall_insights.append(f"⚠️ SMI = {smi:.1f} kg/m² – low muscle index. Strength training and adequate protein are key.")
            risk_level = "MEDIUM" if risk_level != "HIGH" else risk_level
        else:
            overall_insights.append(f"✅ SMI = {smi:.1f} kg/m² – normal muscle index. Keep up the good work.")
    return overall_insights, risk_level

# ==========================================
# 5. SNAPSHOT HELPER
# ==========================================
def freeze_session_state():
    """Capture all ML and session values before STOP clears them."""
    st.session_state.frozen_ml_prediction    = st.session_state.ml_prediction
    st.session_state.frozen_ml_probability   = st.session_state.ml_probability
    st.session_state.frozen_ml_latest        = dict(st.session_state.get("ml_latest", {}))
    st.session_state.frozen_ml_session       = dict(st.session_state.get("ml_session", {}))
    st.session_state.frozen_ml_probabilities = list(st.session_state.get("ml_probabilities", []))
    st.session_state.frozen_ml_summary       = dict(st.session_state.get("ml_summary", {}))
    st.session_state.frozen_intensity        = st.session_state.intensity
    st.session_state.frozen_frequency        = st.session_state.frequency
    st.session_state.frozen_pulse_width      = st.session_state.pulse_width
    st.session_state.frozen_duty_on          = st.session_state.duty_on
    st.session_state.frozen_duty_off         = st.session_state.duty_off
    st.session_state.frozen_pain             = st.session_state.live_pain
    st.session_state.frozen_fatigue          = st.session_state.live_fatigue

# ==========================================
# 6. DIALOGS
# ==========================================
@st.dialog("Start Session Confirmation")
def show_start_confirmation(pid, proto):
    st.write("### Safety Check")
    st.info("Please confirm that electrode placement and skin conditions have been verified manually.")
    st.warning("Ensure patient is ready for stimulation.")
    col_d1, col_d2 = st.columns(2)
    if col_d1.button("Yes (Start)", type="primary"):
        if st.session_state.system_status != "PAUSED":
            st.session_state.elapsed_time = 0.0
            st.session_state.intensity = 15  # reset to default on fresh start
        st.session_state.system_status = "ACTIVE"
        st.session_state.session_start_time = time.time()
        try:
            log_event(pid, "SESSION_START", f"Protocol={proto}")
        except Exception:
            pass
        st.rerun()
    if col_d2.button("No (Cancel)"):
        st.rerun()

@st.dialog("Confirm Intensity Adjustment")
def show_intensity_confirmation(pid, new_val):
    st.write(f"### Adjust Intensity?")
    st.write(f"You are changing the intensity to **{new_val} mA**.")
    st.warning("Please verify this level is safe for the patient.")
    col_i1, col_i2 = st.columns(2)
    if col_i1.button("Accept", type="primary"):
        st.session_state.intensity = new_val
        try:
            log_event(pid, "PARAM_CHANGE", f"Intensity set to {new_val}")
            st.toast(f"Intensity updated to {new_val} mA", icon="⚡") 
        except Exception:
            pass
        st.rerun()
    if col_i2.button("Deny"):
        st.rerun()

# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("AI-Enhanced EMS System")
    st.caption(f"ML Engine: Remote API")
    st.divider()
    st.subheader("Patient Profile")
    patient_id = st.text_input("Patient ID", value="PT-2024-89")
    age_group = st.selectbox("Age Group", ["60-69", "70-79", "80+"])
    condition_tags = st.multiselect("Conditions", ["Sarcopenia", "Post-Stroke", "Osteoarthritis"], default=["Sarcopenia"])
    st.info(f"Height: 170 cm | Weight: 70 kg")

    # Save patient profile to Firebase
    fb_write(f"patients/{patient_id}/profile", {
        "patient_id":  patient_id,
        "age_group":   age_group,
        "conditions":  condition_tags,
        "height_cm":   170,
        "weight_kg":   70,
        "last_seen":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    user_role = st.selectbox("User Role", ["Doctor", "Caregiver"])
    st.divider()
    st.subheader("Simulation")
    sim_mode = st.radio("Patient State:", ["Normal", "Risk (Abnormal)"])
    st.session_state.connected = st.toggle("Device Connected", value=True)
    st.divider()
    st.subheader("Session Control")
    protocol = st.selectbox("Protocol", ["Muscle Stimulation"])
    

# Additional button styling (kept)
st.markdown("""
<style>
    div[data-testid="column"]:nth-of-type(1) div.stButton > button {
        background-color: #2E8B57; color: white; border: none;
    }
    div[data-testid="column"]:nth-of-type(1) div.stButton > button:hover {
        background-color: #3CB371;
    }
    div[data-testid="column"]:nth-of-type(2) div.stButton > button {
        background-color: #FF8C00; color: white; border: none;
    }
    div[data-testid="column"]:nth-of-type(2) div.stButton > button:hover {
        background-color: #FFA500;
    }
    div[data-testid="column"]:nth-of-type(3) div.stButton > button {
        background-color: #D32F2F; color: white; border: none;
    }
    div[data-testid="column"]:nth-of-type(3) div.stButton > button:hover {
        background-color: #EF5350;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 7. HEADER & STATUS
# ==========================================
st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
header_cols = st.columns([3, 1, 1, 1])
with header_cols[0]:
    st.title("Clinical Dashboard")
    st.caption(f"Patient: **{patient_id}** | Protocol: **{protocol}**")
with header_cols[1]:
    color_map = {"ACTIVE": "#2A9D8F", "IDLE": "#78909C", "PAUSED": "#FFB74D", "STOPPED": "#EF5350"}
    status_color = color_map.get(st.session_state.system_status, "#78909C")
    st.markdown(f"<div style='text-align:center; color:{status_color}; font-weight:bold; font-size:1.2em; margin-top:10px;'>● {st.session_state.system_status}</div>", unsafe_allow_html=True)
with header_cols[2]:
    total_seconds = st.session_state.elapsed_time
    if st.session_state.system_status == "ACTIVE" and st.session_state.session_start_time:
        total_seconds += (time.time() - st.session_state.session_start_time)
    elapsed = int(total_seconds)
    timer = f"{elapsed//60:02d}:{elapsed%60:02d}"
    st.metric("Session Time", timer)
with header_cols[3]:
    if st.button("Emergency STOP", type="primary", use_container_width=True):
        if st.session_state.session_start_time:
            st.session_state.elapsed_time += time.time() - st.session_state.session_start_time
        freeze_session_state()
        st.session_state.session_summary_generated = False
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        st.session_state.session_start_time = None
        try:
            log_event(patient_id, "EMERGENCY_STOP", "Immediate Trigger - No Confirmation")
        except Exception:
            pass
        st.rerun()

# ==========================================
# 8. SESSION CONTROLS (Main Body)
# ==========================================
st.markdown("### Session Control")
col_start, col_pause, col_stop = st.columns(3)
with col_start:
    if st.session_state.system_status == "ACTIVE":
        st.button("✅ Session Running...", disabled=True, use_container_width=True)
    else:
        if st.button("▶ START", use_container_width=True):
            show_start_confirmation(patient_id, protocol)
with col_pause:
    is_disabled = st.session_state.system_status != "ACTIVE"
    if st.button("⏸ PAUSE", disabled=is_disabled, use_container_width=True):
        if st.session_state.session_start_time:
            segment_duration = time.time() - st.session_state.session_start_time
            st.session_state.elapsed_time += segment_duration
        st.session_state.system_status = "PAUSED"
        st.session_state.session_start_time = None
        log_event(patient_id, "SESSION_PAUSE")
        st.rerun()
with col_stop:
    is_disabled = st.session_state.system_status not in ["ACTIVE", "PAUSED"]
    if st.button("⏹ STOP SESSION", disabled=is_disabled, use_container_width=True):
        if st.session_state.session_start_time:
            st.session_state.elapsed_time += time.time() - st.session_state.session_start_time
        freeze_session_state()
        st.session_state.session_summary_generated = False
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        st.session_state.session_start_time = None
        log_event(patient_id, "SESSION_STOP")
        st.rerun()

# ==========================================
# 9. MAIN LOGIC LOOP (Telemetry & ML update)
# ==========================================
if st.session_state.connected:
    # 1. Update telemetry from Firebase
    update_telemetry_stream()
    print("Latest EMG:", st.session_state.telemetry['emg'].iloc[-1] if not st.session_state.telemetry.empty else "No data")

    # 2. Fire ML API call in a background thread (non-blocking)
    if st.session_state.system_status == "ACTIVE":

        # Drain result buffer written by the last completed thread
        if _ML_SHARED["result"]:
            st.session_state.ml_prediction    = _ML_SHARED["result"].get("prediction",    "Unknown")
            st.session_state.ml_probability   = _ML_SHARED["result"].get("confidence",    0.0)
            st.session_state.ml_summary       = _ML_SHARED["result"].get("summary",       {})
            st.session_state.ml_latest        = _ML_SHARED["result"].get("latest",        {})
            st.session_state.ml_probabilities = _ML_SHARED["result"].get("probabilities",  [])
            st.session_state.ml_session       = _ML_SHARED["result"].get("session",       {})
            st.session_state.ml_pending       = False
            _ML_SHARED["result"].clear()

        # Start a new worker if previous one is done and interval has elapsed
        now_ts = time.time()
        current_thread = _ML_SHARED["thread"][0]
        thread_dead = current_thread is None or not current_thread.is_alive()
        if thread_dead and (now_ts - st.session_state.last_ml_call_time >= ML_CALL_INTERVAL):
            def _ml_worker():
                prediction, confidence, summary, latest, probabilities, session = call_ml_api()
                _ML_SHARED["result"]["prediction"]    = prediction
                _ML_SHARED["result"]["confidence"]    = confidence
                _ML_SHARED["result"]["summary"]       = summary
                _ML_SHARED["result"]["latest"]        = latest
                _ML_SHARED["result"]["probabilities"] = probabilities
                _ML_SHARED["result"]["session"]       = session

            t = threading.Thread(target=_ml_worker, daemon=True)
            _ML_SHARED["thread"][0]            = t
            st.session_state.ml_pending        = True
            st.session_state.last_ml_call_time = now_ts
            t.start()
    else:
        # Only wipe ML state on IDLE/PAUSED — not on STOPPED,
        # because generate_session_summary() still needs the last-known values.
        if st.session_state.system_status != "STOPPED":
            st.session_state.ml_prediction    = "WAITING"
            st.session_state.ml_probability   = 0.0
            st.session_state.ml_summary       = {}
            st.session_state.ml_latest        = {}
            st.session_state.ml_probabilities = []
            st.session_state.ml_session       = {}
            st.session_state.ml_pending       = False
            _ML_SHARED["result"].clear()

    # 3. Detect session end and generate AI summary (once), then reset timer
    if st.session_state.system_status == "STOPPED" and not st.session_state.session_summary_generated:
        generate_session_summary()
        st.session_state.session_summary_generated = True
        st.session_state.elapsed_time = 0.0  # reset only after summary captured duration

# ==========================================
# 10. INTERFACE – DOCTOR vs CAREGIVER
# ==========================================

if user_role == "Doctor":
    # ---------- DOCTOR VIEW: full clinical dashboard with tabs ----------
    tab_live_ai, tab_body, tab_device, tab_records, tab_chat = st.tabs(
        ["🩺 Live & AI", "🧬 Body Composition", "⚙️ Device Control", "📋 Records & Reports", "💬 Clinical AI Chat"]
    )

        # ---------- TAB 1: LIVE & AI ----------
    with tab_live_ai:
        tele = st.session_state.telemetry
        latest_emg = tele['emg'].iloc[-1] if not tele.empty else 0
        last_hr = tele['hr'].iloc[-1] if not tele.empty else 0
        last_imp = tele['imp'].iloc[-1] if not tele.empty else 0
        recent_peak = tele['emg'].tail(50).max() if not tele.empty else 0

        st.markdown("### Session Status & EMG")
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            mode_value = st.session_state.esp_mode if st.session_state.esp_mode else "—"
            st.metric("🎮 Mode", mode_value, delta=None)
        with col_m2:
            channel_value = st.session_state.esp_channel if st.session_state.esp_channel is not None else "—"
            st.metric("🔢 Channel", channel_value, delta=None)
        with col_m3:
            st.metric("📈 Current EMG", f"{latest_emg:.1f} µV", delta=f"{latest_emg - tele['emg'].iloc[-2] if len(tele)>1 else 0:.1f}")
        with col_m4:
            st.metric("🔔 Peak (Session)", f"{recent_peak:.1f} µV")

        st.markdown("### Real‑Time EMG Telemetry")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tele["t"], y=tele["emg"], mode="lines", fill='tozeroy', line=dict(color='#2A9D8F', width=3)))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis=dict(title="Amplitude (µV)", gridcolor='#E2E8F0'), xaxis=dict(title="Time", showgrid=False), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.esp_state:
            state = st.session_state.esp_state
            if state == "ACTIVE":
                st.info("🔵 **ESP32 Status:** Active – muscle contraction detected")
            elif state == "RELAX":
                st.success("🟢 **ESP32 Status:** Relaxed – low muscle activity")
            elif state == "MEDIUM":
                st.warning("🟡 **ESP32 Status:** Medium activity – monitor closely")
            else:
                st.write(f"ESP32 State: {state}")
    
        st.markdown("---")
        
        # Two columns: left = Safety + Feedback, right = ML Engine
        col_left, col_ml = st.columns(2)

        with col_left:
            # ── Safety & Optimization Rules ────────────────────────────────
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Safety & Optimization (Rules)")
            if st.session_state.system_status == "ACTIVE":
                pain_val = st.session_state.get("live_pain", 0)
                fatigue_val = st.session_state.get("live_fatigue", 0)
                if pain_val >= 6:
                    st.markdown("""<div class="alert-box alert-risk"><strong>High Pain Detected</strong><br>Observation: Pain Score > 6<br>Action: Reducing intensity by 20% (Rule PAIN-01)</div>""", unsafe_allow_html=True)
                elif fatigue_val >= 7:
                    st.markdown("""<div class="alert-box alert-info"><strong>High Fatigue</strong><br>Observation: Patient reported fatigue > 7<br>Action: Increasing OFF time (Rule ONOFF-04)</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="alert-box alert-safe"><strong>System Nominal</strong><br>All parameters within safety limits.<br>Action: Maintain current protocol (Rule MAIN-01)</div>""", unsafe_allow_html=True)
            else:
                st.caption("System Inactive - Start session to monitor safety rules.")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Patient Feedback (now directly below safety rules) ─────────
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("💬 Patient Feedback")
            pain = st.slider("Pain Score (0‑10)", 0, 10, value=st.session_state.get("live_pain", 2), key="live_pain")
            fatigue = st.slider("Fatigue Level (0‑10)", 0, 10, value=st.session_state.get("live_fatigue", 4), key="live_fatigue")
            if pain > 7:
                st.error("🚨 High pain – consider reducing intensity.")
            elif pain > 4:
                st.warning("⚠️ Moderate pain – monitor closely.")
            else:
                st.success("✅ Pain acceptable.")
            if fatigue > 7:
                st.info("💤 High fatigue – suggest longer rest periods.")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col_ml:
            # ── ML Engine (unchanged) ───────────────────────────────────────
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Gait Pathology (ML Engine)")
            if st.session_state.get("ml_pending", False):
                st.caption("🔄 Fetching latest prediction…")

            res = st.session_state.ml_prediction
            prob = st.session_state.ml_probability

            if st.session_state.system_status == "ACTIVE":
                ERROR_PREFIXES = ("API Timeout", "API Offline", "API Error", "Invalid API Response")
                is_error = any(res.startswith(p) for p in ERROR_PREFIXES) if isinstance(res, str) else False

                if res in ("WAITING", ""):
                    st.markdown("""
                    <div style="text-align:center; padding:40px 0; color:#94A3B8;">
                        <div style="font-size:2rem; margin-bottom:8px;">⏳</div>
                        <div style="font-size:1rem; font-weight:600;">Waiting for first prediction…</div>
                        <div style="font-size:0.8rem; margin-top:6px;">The ML engine is reading the latest EMG data.</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif is_error:
                    icon = "⏱️" if "Timeout" in res else "📡" if "Offline" in res else "⚠️"
                    tip = ("The ngrok tunnel may be down or the Flask backend is not running." if "Offline" in res else
                           "The backend took too long to respond. It will retry automatically." if "Timeout" in res else
                           "Check the backend logs for details.")
                    st.markdown(f"""
                    <div style="background:#FFF8E1; border:1px solid #FFD54F; border-radius:12px;
                                padding:20px; text-align:center; color:#7B4F00;">
                        <div style="font-size:2rem; margin-bottom:6px;">{icon}</div>
                        <div style="font-size:1rem; font-weight:700; margin-bottom:4px;">{res}</div>
                        <div style="font-size:0.8rem; color:#8D6E63;">{tip}</div>
                        <div style="font-size:0.75rem; margin-top:10px; color:#BDBDBD;">
                            Retrying every {ML_CALL_INTERVAL}s…
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    latest = st.session_state.get("ml_latest", {})
                    session = st.session_state.get("ml_session", {})
                    probabilities = st.session_state.get("ml_probabilities", [])
                    summary = st.session_state.get("ml_summary", {})

                    is_abnormal = res in ("ABNORMAL", "Abnormal")
                    pred_color = "#B71C1C" if is_abnormal else "#1B5E20"
                    pred_bg    = "#FFEBEE" if is_abnormal else "#E8F5E9"
                    pred_label = res if res else "—"
                    conf_pct   = prob * 100 if prob <= 1.0 else prob
                    conf_fill  = max(0.0, min(conf_pct, 100.0))

                    conf_note = ("High confidence. The model result is relatively stable for this reading." if conf_pct >= 75 else
                                 "Moderate confidence. Consider monitoring additional readings." if conf_pct >= 50 else
                                 "Low confidence. Result may be unreliable — check sensor placement.")

                    st.markdown(f"""
                    <div style="margin-bottom:16px;">
                        <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                            <span style="background:#E8EAF6; color:#3949AB; font-weight:700;
                                         font-size:0.8rem; padding:4px 12px; border-radius:20px;
                                         letter-spacing:0.05em;">PREDICTED CLASS</span>
                            <span style="font-size:1.25rem; font-weight:700; color:{pred_color};
                                         background:{pred_bg}; padding:3px 14px; border-radius:16px;">
                                {pred_label}
                            </span>
                        </div>
                        <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                            <span style="background:#E8EAF6; color:#3949AB; font-weight:700;
                                         font-size:0.8rem; padding:4px 12px; border-radius:20px;
                                         letter-spacing:0.05em;">CONFIDENCE</span>
                            <span style="font-size:1.1rem; font-weight:700; color:#1E293B;">
                                {conf_pct:.2f}%
                            </span>
                        </div>
                        <div style="background:#E2E8F0; border-radius:8px; height:14px; width:100%; margin-bottom:4px;">
                            <div style="background:#3949AB; width:{conf_fill}%; height:14px;
                                        border-radius:8px; transition:width 0.4s ease;"></div>
                        </div>
                        <div style="display:flex; justify-content:space-between;
                                    font-size:0.72rem; color:#78909C; margin-bottom:4px;">
                            <span>0%</span><span>50%</span><span>100%</span>
                        </div>
                        <p style="font-size:0.8rem; color:#546E7A; margin:0;">{conf_note}</p>
                    </div>
                    """, unsafe_allow_html=True)

                    if latest:
                        rms_val    = latest.get('rms_recto_femoral', 0)
                        spread_val = latest.get('rms_signal_spread', 0)
                        std_val    = latest.get('rms_signal_std', 0)
                        st.markdown(f"""
                        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:14px;">
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 10px;">
                                <div style="font-size:0.68rem; font-weight:700; color:#78909C; letter-spacing:0.07em; margin-bottom:6px;">
                                    RECTO FEMORIS RMS
                                </div>
                                <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:6px;">{rms_val:.2f}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; line-height:1.4;">
                                    Average EMG level from the recto femoris sensor in the latest cleaned 1-second window.
                                </div>
                            </div>
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 10px;">
                                <div style="font-size:0.68rem; font-weight:700; color:#78909C; letter-spacing:0.07em; margin-bottom:6px;">
                                    SIGNAL SPREAD
                                </div>
                                <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:6px;">{spread_val:.0f}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; line-height:1.4;">
                                    Formula: max − min after cleaning. Shows the signal range within the 1-second window.
                                </div>
                            </div>
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0; border-radius:12px; padding:14px 10px;">
                                <div style="font-size:0.68rem; font-weight:700; color:#78909C; letter-spacing:0.07em; margin-bottom:6px;">
                                    SIGNAL STD
                                </div>
                                <div style="font-size:1.6rem; font-weight:800; color:#0F172A; margin-bottom:6px;">{std_val:.4f}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; line-height:1.4;">
                                    Standard deviation of cleaned EMG samples. Shows how much the signal fluctuates around the average.
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if probabilities:
                        st.markdown("**Prediction Probability**")
                        for p in probabilities:
                            label_p = p.get("label", "Unknown")
                            prob_p  = float(p.get("probability", 0))
                            st.progress(prob_p / 100, text=f"{label_p}: {prob_p:.2f}%")

                    if session:
                        st.markdown(f"""
                        <div style="background:#F1F5F9; border:1px solid #E2E8F0; border-radius:10px;
                                    padding:12px 16px; margin:10px 0;">
                            <div style="font-weight:700; font-size:0.9rem; color:#1E293B; margin-bottom:4px;">Session Summary</div>
                            <div style="font-size:0.85rem; color:#475569;">
                                Readings: <strong>{session.get('count', '—')}</strong> &nbsp;|&nbsp;
                                Average: <strong>{session.get('avg', '—')}</strong> &nbsp;|&nbsp;
                                Min: <strong>{session.get('min', '—')}</strong> &nbsp;|&nbsp;
                                Max: <strong>{session.get('max', '—')}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    if summary:
                        with st.expander("📋 AI Summary & Recommendations"):
                            st.markdown(f"**{summary.get('title', '')}**")
                            st.markdown(summary.get('summary', ''))
                            if summary.get('interpretation'):
                                st.markdown("**Interpretation**")
                                for item in summary['interpretation']:
                                    st.markdown(f"- {item}")
                            if summary.get('actions'):
                                st.markdown("**Recommended Actions**")
                                for item in summary['actions']:
                                    st.markdown(f"- {item}")
                            disclaimer = summary.get('disclaimer', '')
                            if disclaimer:
                                st.caption(f"*Note: {disclaimer}*")
            else:
                st.info("Start session to enable ML analysis.")

            st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TAB 2: BODY COMPOSITION ----------
    with tab_body:
        st.markdown("## InBody Scan Analysis")
        st.info("Upload an InBody report image. EasyOCR will extract and organize the body composition results.")
        col_ocr_left, col_ocr_right = st.columns([1.1, 1.4], gap="large")
        with col_ocr_left:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Upload InBody Report")
            uploaded_scan = st.file_uploader("Choose report image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
            if uploaded_scan is not None:
                st.image(uploaded_scan, caption="Uploaded InBody Report", use_container_width=True)
                run_btn = st.button("Run OCR Analysis", type="primary", use_container_width=True)
                st.info("📸 **Tip:** Ensure the report is well‑lit, flat, and the text is clearly visible. Avoid shadows and blurry images.")
            else:
                st.markdown("""<div style="height:300px; display:flex; align-items:center; justify-content:center; border:2px dashed #CBD5E1; border-radius:12px; color:#64748B; background:#F8FAFC;">Waiting for report upload...</div>""", unsafe_allow_html=True)
                run_btn = False
            st.markdown('</div>', unsafe_allow_html=True)

        with col_ocr_right:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Body Composition Results")
            def show_metric_card(label, value, unit, status):
                if value is None:
                    return None
                if status == "normal":
                    bg_color = "#FFFFFF"
                    status_color = "#2E7D32"
                    border_color = "#E0E0E0"
                elif status == "under":
                    bg_color = "#FFF8E1"
                    status_color = "#F57C00"
                    border_color = "#FFE0B2"
                elif status == "over":
                    bg_color = "#FFF8E1"
                    status_color = "#F57C00"
                    border_color = "#FFE0B2"
                elif status == "high":
                    bg_color = "#FFEBEE"
                    status_color = "#C62828"
                    border_color = "#FFCDD2"
                else:
                    bg_color = "#F5F5F5"
                    status_color = "#757575"
                    border_color = "#E0E0E0"
                return f"""
                <div style="background-color:{bg_color}; border-radius:12px; padding:10px; margin-bottom:8px; border:1px solid {border_color}; height:100%;">
                    <div style="font-size:0.75rem; color:#546E7A;">{label}</div>
                    <div style="font-size:1.5rem; font-weight:700; color:#1E293B;">{value} {unit}</div>
                    <div style="font-size:0.65rem; color:{status_color}; margin-top:4px;">{status.upper()}</div>
                </div>
                """
            if uploaded_scan is not None and run_btn:
                with st.spinner("Extracting and analyzing report..."):
                    try:
                        text_results = run_easyocr(uploaded_scan)
                        full_text = " ".join(text_results)
                        metrics = extract_metrics_from_text(full_text)
                        overall_insights, risk_level = generate_inbody_ai_insight(metrics)
                        st.success("OCR Analysis Completed")
                        pid = metrics.get("Patient ID", "Not found")
                        tdate = metrics.get("Test Date", "Not found")
                        st.markdown(f"**Patient ID:** `{pid}` &nbsp;&nbsp;|&nbsp;&nbsp; **Test Date:** `{tdate}`")
                        smm = metrics.get("Skeletal Muscle Mass (kg)")
                        smi_val = metrics.get("SMI")
                        col1, col2 = st.columns(2)
                        with col1:
                            if smm is not None:
                                if smm < 28:
                                    smm_status = "under"
                                elif smm > 35:
                                    smm_status = "over"
                                else:
                                    smm_status = "normal"
                                st.markdown(show_metric_card("Skeletal Muscle Mass", smm, "kg", smm_status), unsafe_allow_html=True)
                        with col2:
                            if smi_val is not None:
                                if smi_val < 7:
                                    smi_status = "under"
                                else:
                                    smi_status = "normal"
                                st.markdown(show_metric_card("SMI", smi_val, "kg/m²", smi_status), unsafe_allow_html=True)
                        pbf = metrics.get("Body Fat Percentage")
                        vfl = metrics.get("Visceral Fat Level")
                        col3, col4 = st.columns(2)
                        with col3:
                            if pbf is not None:
                                if pbf < 10:
                                    pbf_status = "under"
                                elif 10 <= pbf <= 20:
                                    pbf_status = "normal"
                                else:
                                    pbf_status = "over"
                                st.markdown(show_metric_card("Body Fat %", pbf, "%", pbf_status), unsafe_allow_html=True)
                        with col4:
                            if vfl is not None:
                                vfl_status = "high" if vfl >= 10 else "normal"
                                st.markdown(show_metric_card("Visceral Fat Level", vfl, "", vfl_status), unsafe_allow_html=True)
                        bmi = metrics.get("BMI")
                        score = metrics.get("InBody Score")
                        col5, col6 = st.columns(2)
                        with col5:
                            if bmi is not None:
                                if bmi < 18.5:
                                    bmi_status = "under"
                                elif 18.5 <= bmi <= 25:
                                    bmi_status = "normal"
                                else:
                                    bmi_status = "over"
                                st.markdown(show_metric_card("BMI", bmi, "kg/m²", bmi_status), unsafe_allow_html=True)
                        with col6:
                            if score is not None:
                                score_status = "under" if score < 70 else "normal"
                                st.markdown(show_metric_card("InBody Score", score, "/100", score_status), unsafe_allow_html=True)
                        target = metrics.get("Target Weight (kg)")
                        w_control = metrics.get("Weight Control (kg)")
                        fat_control = metrics.get("Fat Control (kg)")
                        m_control = metrics.get("Muscle Control (kg)")
                        controls = []
                        if target is not None:
                            controls.append(("Target Weight", target, "kg"))
                        if w_control is not None:
                            controls.append(("Weight Control", w_control, "kg"))
                        if fat_control is not None:
                            controls.append(("Fat Control", fat_control, "kg"))
                        if m_control is not None:
                            controls.append(("Muscle Control", m_control, "kg"))
                        if controls:
                            st.markdown("#### Weight Control Recommendations")
                            cols = st.columns(len(controls))
                            for i, (label, val, unit) in enumerate(controls):
                                with cols[i]:
                                    if label in ["Weight Control", "Fat Control"]:
                                        color = "#DC2626" if val < 0 else "#16A34A"
                                    elif label == "Muscle Control":
                                        color = "#16A34A" if val > 0 else "#DC2626"
                                    else:
                                        color = "#0F172A"
                                    sign = "+" if val > 0 else ""
                                    st.markdown(f"""
                                    <div style="background-color:#FFFFFF; border-radius:12px; padding:14px; border:1px solid #E0E0E0; text-align:center;">
                                        <div style="font-size:0.8rem; color:#64748B;">{label}</div>
                                        <div style="font-size:1.7rem; font-weight:700; color:{color};">{sign}{val:.1f} {unit}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        if risk_level == "HIGH":
                            st.error(f"**Overall Risk Level: {risk_level}** – Requires clinical attention.")
                        elif risk_level == "MEDIUM":
                            st.warning(f"**Overall Risk Level: {risk_level}** – Monitor closely.")
                        else:
                            st.success(f"**Overall Risk Level: {risk_level}** – Within acceptable limits.")
                        with st.expander("📋 Full Clinical Insights (AI‑generated)"):
                            for insight in overall_insights:
                                st.write(insight)
                        with st.expander("🦵 Segmental Analysis Interpretation"):
                            st.markdown("""
                            **Segmental Lean Analysis** – Evaluates muscle distribution compared to current weight.  
                            **Segmental Fat Analysis** – Evaluates fat distribution compared to ideal.  
                            **Visceral Fat Level** – Fat surrounding internal organs; keep under 10.  
                            **SMI** – Appendicular lean mass / height²; <7 kg/m² indicates low muscle index.
                            """)
                        with st.expander("🔍 View Raw OCR Extracted Text"):
                            st.text(full_text)
                        if st.button("Save to Patient Record", use_container_width=True):
                            details = json.dumps(metrics)
                            log_event(patient_id, "INBODY_OCR_UPLOAD", details)
                            _report_date = metrics.get("Test Date", datetime.now().strftime("%Y-%m-%d"))
                            _safe_date = str(_report_date).replace("/", "-").replace(" ", "_").replace(":", "")
                            _bia_data = {
                                "date":                    _report_date,
                                "skeletal_muscle_mass_kg": metrics.get("Skeletal Muscle Mass (kg)"),
                                "body_fat_pct":            metrics.get("Body Fat Percentage"),
                                "bmi":                     metrics.get("BMI"),
                                "visceral_fat_level":      metrics.get("Visceral Fat Level"),
                                "inbody_score":            metrics.get("InBody Score"),
                                "smi":                     metrics.get("SMI"),
                                "weight_kg":               metrics.get("Weight (kg)"),
                                "uploaded_at":             datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "source":                  "EasyOCR_InBody"
                            }
                            if fb_write(f"patients/{patient_id}/bia_reports/{_safe_date}", _bia_data):
                                st.toast("InBody OCR data saved to Firebase!", icon="✅")
                            else:
                                st.toast("Saved locally. Firebase sync failed.", icon="⚠️")
                    except Exception as e:
                        st.error(f"OCR failed: {e}")
            else:
                st.markdown("""
                <div style="height:450px; display:flex; align-items:center; justify-content:center; border:2px dashed #CBD5E1; border-radius:12px; color:#64748B; background:#F8FAFC; text-align:center; padding:20px;">
                    Upload a report and run OCR analysis to view<br>
                    body composition metrics.
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    # ---------- TAB 3: DEVICE CONTROL ----------
    with tab_device:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Stimulation Parameters")
        c1, c2, c3 = st.columns(3)
        c1.metric("Intensity", f"{st.session_state.intensity} mA")
        c2.metric("Frequency", f"{st.session_state.frequency} Hz")
        c3.metric("Pulse Width", f"{st.session_state.pulse_width} us")
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(0.5, text=f"Duty Cycle: {st.session_state.duty_on}s ON / {st.session_state.duty_off}s OFF")
        st.divider()
        if user_role == "Doctor":
            col_adj, col_btn = st.columns([3, 1])
            with col_adj:
                new_int = st.slider("Adjust Intensity (mA)", 0, 100, st.session_state.intensity)
            with col_btn:
                st.markdown("<br>", unsafe_allow_html=True)
                if new_int != st.session_state.intensity:
                    if st.button("Apply Changes", type="primary"):
                        show_intensity_confirmation(patient_id, new_int)
        else:
            st.warning("Intensity adjustments are locked for Caregiver role.")
        st.markdown('</div>', unsafe_allow_html=True)
        
   # ---------- TAB 4: RECORDS & REPORTS ----------
    with tab_records:
        if st.session_state.session_summary_text:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("📝 AI Session Summary")
    
            summary = st.session_state.session_summary_text
            if isinstance(summary, dict):
        # Title
            st.markdown(f"## {summary.get('title', 'Clinical Session Summary')}")
        # Summary paragraph
            st.markdown(summary.get('summary', ''))
        # Signal Interpretation
            st.markdown("### Signal Interpretation")
            for point in summary.get('interpretation', []):
                st.markdown(f"- {point}")
        # Recommended Actions
            st.markdown("### Recommended Actions")
            for action in summary.get('actions', []):
                st.markdown(f"- {action}")
         else:
            st.markdown(summary)  # fallback
    
         if st.button("Regenerate Summary", key="regenerate_summary"):
            st.session_state.session_summary_generated = False
            st.rerun()
         st.markdown('</div>', unsafe_allow_html=True)
    
            st.divider()
            st.subheader("🔐 Clinician Approval")
            st.info("Review the AI-generated summary above before approving parameter changes for the next session.")

            _col_approve, _col_decline = st.columns(2)
            _session_id = st.session_state.get("current_session_id", None)

            with _col_approve:
                if st.button("✅ Approve Recommendation", type="primary",
                             use_container_width=True, key="btn_approve"):
                    if _session_id:
                        fb_write(
                            f"patients/{patient_id}/sessions/{_session_id}/approval",
                            {
                                "status":    "approved",
                                "clinician": user_role,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        )
                    log_event(patient_id, "APPROVAL_GRANTED", f"Session={_session_id}")
                    st.success("✅ Recommendation approved and saved to Firebase.")

            with _col_decline:
                if st.button("❌ Decline — Keep Current Parameters",
                             use_container_width=True, key="btn_decline"):
                    if _session_id:
                        fb_write(
                            f"patients/{patient_id}/sessions/{_session_id}/approval",
                            {
                                "status":    "declined",
                                "clinician": user_role,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            }
                        )
                    log_event(patient_id, "APPROVAL_DECLINED", f"Session={_session_id}")
                    st.warning("❌ Declined. Current parameters will be retained.")

            # Live approval status from Firebase
            if _session_id:
                try:
                    _ap = fb_read(f"patients/{patient_id}/sessions/{_session_id}/approval")
                    if _ap and isinstance(_ap, dict):
                        _ts = _ap.get("timestamp", "—")
                        _by = _ap.get("clinician", "—")
                        if _ap.get("status") == "approved":
                            st.markdown(f"""
                            <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;
                                        padding:12px 16px;margin-top:10px;display:flex;align-items:center;gap:10px;">
                                <span style="font-size:1.2rem;">✅</span>
                                <div>
                                    <div style="font-size:0.82rem;font-weight:700;color:#15803D;">Last Decision: Approved</div>
                                    <div style="font-size:0.75rem;color:#166534;">{_ts} &nbsp;·&nbsp; {_by}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                        elif _ap.get("status") == "declined":
                            st.markdown(f"""
                            <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;
                                        padding:12px 16px;margin-top:10px;display:flex;align-items:center;gap:10px;">
                                <span style="font-size:1.2rem;">❌</span>
                                <div>
                                    <div style="font-size:0.82rem;font-weight:700;color:#92400E;">Last Decision: Declined</div>
                                    <div style="font-size:0.75rem;color:#78350F;">{_ts} &nbsp;·&nbsp; {_by}</div>
                                </div>
                            </div>""", unsafe_allow_html=True)
                except Exception:
                    pass

            st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # 2. SESSION PAIN & FATIGUE TRENDS (improved layout)
        # =====================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📊 Session Pain & Fatigue Trends")

        _pr_times  = [0, 5, 10, 15, 20]
        _pr_pain   = [5, 4, 3, 2, 1]
        _pr_fatigue = [6, 5, 4, 3, 2]

        # Helper for shaded zones
        def _make_zone(fig, y0, y1, color, label):
            fig.add_hrect(y0=y0, y1=y1, fillcolor=color, opacity=0.06,
                          line_width=0, annotation_text=label,
                          annotation_position="right", annotation_font_size=9,
                          annotation_font_color="#94A3B8")

        col_p1, col_p2 = st.columns(2, gap="large")

        with col_p1:
            st.markdown("#### Pain Score (0–10)")
            _fig_pain = go.Figure()
            _fig_pain.add_trace(go.Scatter(
                x=_pr_times, y=_pr_pain,
                mode="lines+markers+text",
                name="Pain",
                line=dict(color="#EF5350", width=3, shape="spline"),
                marker=dict(size=12, color="#EF5350", line=dict(color="white", width=2)),
                fill="tozeroy", fillcolor="rgba(239,83,80,0.1)",
                text=[str(v) for v in _pr_pain],
                textposition="top center",
                textfont=dict(size=12, color="#B71C1C", family="Arial Black"),
                hovertemplate="<b>%{x} min</b><br>Pain: %{y}/10<extra></extra>"
            ))
            _fig_pain.add_hline(y=7, line_dash="dash", line_color="#EF5350", line_width=1.5,
                                annotation_text="High (≥7)", annotation_position="right",
                                annotation_font_size=10, annotation_font_color="#EF5350")
            _fig_pain.add_hline(y=4, line_dash="dot", line_color="#F4A261", line_width=1.5,
                                annotation_text="Moderate (≥4)", annotation_position="right",
                                annotation_font_size=10, annotation_font_color="#F4A261")
            _make_zone(_fig_pain, 0, 4,   "#16A34A", "")
            _make_zone(_fig_pain, 4, 7,   "#F4A261", "")
            _make_zone(_fig_pain, 7, 10,  "#EF5350", "")
            _fig_pain.update_layout(
                height=320,
                margin=dict(l=10, r=60, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title="Session Time (min)", tickvals=_pr_times,
                    ticktext=["0 min","5 min","10 min","15 min","20 min"],
                    gridcolor="#E2E8F0", showline=True,
                    linecolor="#CBD5E1", tickfont=dict(size=11)
                ),
                yaxis=dict(
                    title="Score", range=[0, 10], dtick=2,
                    gridcolor="#E2E8F0", showline=True,
                    linecolor="#CBD5E1", tickfont=dict(size=11)
                ),
                showlegend=False
            )
            st.plotly_chart(_fig_pain, use_container_width=True)

            # Stats strip
            _p_start, _p_end = _pr_pain[0], _pr_pain[-1]
            _p_delta = _p_end - _p_start
            _p_col = "#16A34A" if _p_delta < 0 else "#EF5350" if _p_delta > 0 else "#64748B"
            _p_arrow = "↓" if _p_delta < 0 else "↑" if _p_delta > 0 else "→"
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-top:8px;">
                <div style="flex:1;background:#FEF2F2;border-radius:12px;padding:10px 8px;text-align:center;">
                    <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;">START</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#EF5350;">{_p_start}/10</div>
                </div>
                <div style="flex:1;background:#F0FDF4;border-radius:12px;padding:10px 8px;text-align:center;">
                    <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;">END</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#16A34A;">{_p_end}/10</div>
                </div>
                <div style="flex:1;background:#F8FAFC;border-radius:12px;padding:10px 8px;text-align:center;">
                    <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;">CHANGE</div>
                    <div style="font-size:1.4rem;font-weight:800;color:{_p_col};">{_p_arrow} {abs(_p_delta)}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        with col_p2:
            st.markdown("#### Fatigue Level (0–10)")
            _fig_fat = go.Figure()
            _fig_fat.add_trace(go.Scatter(
                x=_pr_times, y=_pr_fatigue,
                mode="lines+markers+text",
                name="Fatigue",
                line=dict(color="#3B82F6", width=3, shape="spline"),
                marker=dict(size=12, color="#3B82F6", line=dict(color="white", width=2)),
                fill="tozeroy", fillcolor="rgba(59,130,246,0.1)",
                text=[str(v) for v in _pr_fatigue],
                textposition="top center",
                textfont=dict(size=12, color="#1D4ED8", family="Arial Black"),
                hovertemplate="<b>%{x} min</b><br>Fatigue: %{y}/10<extra></extra>"
            ))
            _fig_fat.add_hline(y=7, line_dash="dash", line_color="#3B82F6", line_width=1.5,
                               annotation_text="High (≥7)", annotation_position="right",
                               annotation_font_size=10, annotation_font_color="#3B82F6")
            _fig_fat.add_hline(y=4, line_dash="dot", line_color="#93C5FD", line_width=1.5,
                               annotation_text="Moderate (≥4)", annotation_position="right",
                               annotation_font_size=10, annotation_font_color="#93C5FD")
            _make_zone(_fig_fat, 0, 4,  "#16A34A", "")
            _make_zone(_fig_fat, 4, 7,  "#3B82F6", "")
            _make_zone(_fig_fat, 7, 10, "#1D4ED8", "")
            _fig_fat.update_layout(
                height=320,
                margin=dict(l=10, r=60, t=30, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(
                    title="Session Time (min)", tickvals=_pr_times,
                    ticktext=["0 min","5 min","10 min","15 min","20 min"],
                    gridcolor="#E2E8F0", showline=True,
                    linecolor="#CBD5E1", tickfont=dict(size=11)
                ),
                yaxis=dict(
                    title="Level", range=[0, 10], dtick=2,
                    gridcolor="#E2E8F0", showline=True,
                    linecolor="#CBD5E1", tickfont=dict(size=11)
                ),
                showlegend=False
            )
            st.plotly_chart(_fig_fat, use_container_width=True)

            _f_start, _f_end = _pr_fatigue[0], _pr_fatigue[-1]
            _f_delta = _f_end - _f_start
            _f_col = "#16A34A" if _f_delta < 0 else "#EF5350" if _f_delta > 0 else "#64748B"
            _f_arrow = "↓" if _f_delta < 0 else "↑" if _f_delta > 0 else "→"
            st.markdown(f"""
            <div style="display:flex;gap:12px;margin-top:8px;">
                <div style="flex:1;background:#EFF6FF;border-radius:12px;padding:10px 8px;text-align:center;">
                    <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;">START</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#3B82F6;">{_f_start}/10</div>
                </div>
                <div style="flex:1;background:#F0FDF4;border-radius:12px;padding:10px 8px;text-align:center;">
                    <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;">END</div>
                    <div style="font-size:1.4rem;font-weight:800;color:#16A34A;">{_f_end}/10</div>
                </div>
                <div style="flex:1;background:#F8FAFC;border-radius:12px;padding:10px 8px;text-align:center;">
                    <div style="font-size:0.75rem;font-weight:600;color:#94A3B8;">CHANGE</div>
                    <div style="font-size:1.4rem;font-weight:800;color:{_f_col};">{_f_arrow} {abs(_f_delta)}</div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Keep these dataframes for export report
        pain_score_progress = pd.DataFrame({'Time': ['0 min','5 min','10 min','15 min','20 min'], 'Pain Score': _pr_pain})
        fatigue_progress    = pd.DataFrame({'Time': ['0 min','5 min','10 min','15 min','20 min'], 'Fatigue Level': _pr_fatigue})

        st.divider()

        # =====================================================
        # 3. MUSCLE ACTIVATION (EMG) – 20‑min session overview
        # =====================================================
        st.subheader("⚡ Muscle Activation (EMG) — Last 20‑Min Session")
        st.markdown("_Real‑time EMG amplitude during the session (smoothed with 20‑point moving average). Colour indicates muscle state: 🔴 Active, 🟡 Medium, 🔵 Relax._")

        # Embedded EMG data (the same as in the original professional dashboard)
        _emg_mins   = [0.0, 0.03, 0.07, 0.1, 0.13, 0.17, 0.2, 0.23, 0.27, 0.3, 0.33, 0.37, 0.4, 0.43, 0.47, 0.5, 0.53, 0.57, 0.6, 0.63, 0.67, 0.7, 0.73, 0.77, 0.8, 0.83, 0.87, 0.9, 0.93, 0.97, 1.0, 1.03, 1.07, 1.1, 1.13, 1.17, 1.2, 1.23, 1.27, 1.3, 1.33, 1.37, 1.4, 1.43, 1.47, 1.5, 1.53, 1.57, 1.6, 1.63, 1.67, 1.7, 1.73, 1.77, 1.8, 1.83, 1.87, 1.9, 1.93, 1.97, 2.0, 2.03, 2.07, 2.1, 2.13, 2.17, 2.2, 2.23, 2.27, 2.3, 2.33, 2.37, 2.4, 2.43, 2.47, 2.5, 2.53, 2.57, 2.6, 2.63, 2.67, 2.7, 2.73, 2.77, 2.8, 2.83, 2.87, 2.9, 2.93, 2.97, 3.0, 3.03, 3.07, 3.1, 3.13, 3.17, 3.2, 3.23, 3.27, 3.3, 3.33, 3.37, 3.4, 3.43, 3.47, 3.5, 3.53, 3.57, 3.6, 3.63, 3.67, 3.7, 3.73, 3.77, 3.8, 3.83, 3.87, 3.9, 3.93, 3.97, 4.0, 4.03, 4.07, 4.1, 4.13, 4.17, 4.2, 4.23, 4.27, 4.3, 4.33, 4.37, 4.4, 4.43, 4.47, 4.5, 4.53, 4.57, 4.6, 4.63, 4.67, 4.7, 4.73, 4.77, 4.8, 4.83, 4.87, 4.9, 4.93, 4.97, 5.0, 5.03, 5.07, 5.1, 5.13, 5.17, 5.2, 5.23, 5.27, 5.3, 5.33, 5.37, 5.4, 5.43, 5.47, 5.5, 5.53, 5.57, 5.6, 5.63, 5.67, 5.7, 5.73, 5.77, 5.8, 5.83, 5.87, 5.9, 5.93, 5.97, 6.02, 6.05, 6.08, 6.12, 6.15, 6.18, 6.22, 6.25, 6.28, 6.32, 6.35, 6.38, 6.42, 6.45, 6.48, 6.52, 6.55, 6.58, 6.62, 6.65, 6.68, 6.72, 6.75, 6.78, 6.82, 6.85, 6.88, 6.92, 6.95, 6.98, 7.02, 7.05, 7.08, 7.12, 7.15, 7.18, 7.22, 7.25, 7.28, 7.32, 7.35, 7.38, 7.42, 7.45, 7.48, 7.52, 7.55, 7.58, 7.62, 7.65, 7.68, 7.72, 7.75, 7.78, 7.82, 7.85, 7.88, 7.92, 7.95, 7.98, 8.02, 8.05, 8.08, 8.12, 8.15, 8.18, 8.22, 8.25, 8.28, 8.32, 8.35, 8.38, 8.42, 8.45, 8.48, 8.52, 8.55, 8.58, 8.62, 8.65, 8.68, 8.72, 8.75, 8.78, 8.82, 8.85, 8.88, 8.92, 8.95, 8.98, 9.02, 9.05, 9.08, 9.12, 9.15, 9.18, 9.22, 9.25, 9.28, 9.32, 9.35, 9.38, 9.42, 9.45, 9.48, 9.52, 9.55, 9.58, 9.62, 9.65, 9.68, 9.72, 9.75, 9.78, 9.82, 9.85, 9.88, 9.92, 9.95, 9.98, 10.02, 10.05, 10.08, 10.12, 10.15, 10.18, 10.22, 10.25, 10.28, 10.32, 10.35, 10.38, 10.42, 10.45, 10.48, 10.52, 10.55, 10.58, 10.62, 10.65, 10.68, 10.72, 10.75, 10.78, 10.82, 10.85, 10.88, 10.92, 10.95, 10.98, 11.02, 11.05, 11.08, 11.12, 11.15, 11.18, 11.22, 11.25, 11.28, 11.32, 11.35, 11.38, 11.42, 11.45, 11.48, 11.52, 11.55, 11.58, 11.62, 11.65, 11.68, 11.72, 11.75, 11.78, 11.82, 11.85, 11.88, 11.92, 11.95, 11.98, 12.02, 12.05, 12.08, 12.12, 12.15, 12.18, 12.22, 12.25, 12.28, 12.32, 12.35, 12.38, 12.42, 12.45, 12.48, 12.52, 12.55, 12.58, 12.62, 12.65, 12.68, 12.72, 12.75, 12.78, 12.82, 12.85, 12.88, 12.92, 12.95, 12.98, 13.02, 13.05, 13.08, 13.12, 13.15, 13.18, 13.22, 13.25, 13.28, 13.32, 13.35, 13.38, 13.42, 13.45, 13.48, 13.52, 13.55, 13.58, 13.62, 13.65, 13.68, 13.72, 13.75, 13.78, 13.82, 13.85, 13.88, 13.92, 13.95, 13.98, 14.02, 14.05, 14.08, 14.12, 14.15, 14.18, 14.22, 14.25, 14.28, 14.32, 14.35, 14.38, 14.42, 14.45, 14.48, 14.52, 14.55, 14.58, 14.62, 14.65, 14.68, 14.72, 14.75, 14.78, 14.82, 14.85, 14.88, 14.92, 14.95, 14.98, 15.02, 15.05, 15.08, 15.12, 15.15, 15.18, 15.22, 15.25, 15.28, 15.32, 15.35, 15.38, 15.42, 15.45, 15.48, 15.52, 15.55, 15.58, 15.62, 15.65, 15.68, 15.72, 15.75, 15.78, 15.82, 15.85, 15.88, 15.92, 15.95, 15.98, 16.02, 16.05, 16.08, 16.12, 16.15, 16.18, 16.22, 16.25, 16.28, 16.32, 16.35, 16.38, 16.42, 16.45, 16.48, 16.52, 16.55, 16.58, 16.62, 16.65, 16.68, 16.72, 16.75, 16.78, 16.82, 16.85, 16.88, 16.92, 16.95, 16.98, 17.02, 17.05, 17.08, 17.12, 17.15, 17.18, 17.22, 17.25, 17.28, 17.32, 17.35, 17.38, 17.42, 17.45, 17.48, 17.52, 17.55, 17.58, 17.62, 17.65, 17.68, 17.72, 17.75, 17.78, 17.82, 17.85, 17.88, 17.92, 17.95, 17.98, 18.02, 18.05, 18.08, 18.12, 18.15, 18.18, 18.22, 18.25, 18.28, 18.32, 18.35, 18.38, 18.42, 18.45, 18.48, 18.52, 18.55, 18.58, 18.62, 18.65, 18.68, 18.72, 18.75, 18.78, 18.82, 18.85, 18.88, 18.92, 18.95, 18.98, 19.02, 19.05, 19.08, 19.12, 19.15, 19.18, 19.22, 19.25, 19.28, 19.32, 19.35, 19.38, 19.42, 19.45, 19.48, 19.52, 19.55, 19.58, 19.62, 19.65, 19.68, 19.72, 19.75, 19.78, 19.82, 19.85, 19.88, 19.92, 19.95, 19.98]
        _emg_vals   = [454.0, 446.0, 446.0, 460.0, 495.0, 381.0, 466.0, 502.0, 443.0, 418.0, 438.0, 455.0, 448.0, 467.0, 431.0, 414.0, 500.0, 401.0, 364.0, 338.0, 533.0, 346.0, 429.0, 499.0, 509.0, 504.0, 386.0, 458.0, 538.0, 586.0, 523.0, 545.0, 455.0, 455.0, 569.0, 431.0, 462.0, 350.0, 443.0, 455.0, 394.0, 323.0, 298.0, 427.0, 454.0, 526.0, 452.0, 467.0, 458.0, 451.0, 510.0, 433.0, 454.0, 455.0, 460.0, 469.0, 447.0, 436.0, 449.0, 455.0, 458.0, 516.0, 491.0, 480.0, 454.0, 458.0, 463.0, 466.0, 471.0, 457.0, 457.0, 457.0, 441.0, 510.0, 494.0, 466.0, 447.0, 449.0, 470.0, 362.0, 461.0, 459.0, 454.0, 460.0, 375.0, 467.0, 456.0, 456.0, 466.0, 601.0, 531.0, 413.0, 430.0, 461.0, 434.0, 439.0, 576.0, 403.0, 433.0, 432.0, 473.0, 450.0, 454.0, 443.0, 341.0, 510.0, 524.0, 567.0, 465.0, 384.0, 438.0, 347.0, 412.0, 474.0, 359.0, 365.0, 435.0, 529.0, 504.0, 522.0, 434.0, 440.0, 373.0, 170.0, 337.0, 226.0, 501.0, 415.0, 652.0, 583.0, 429.0, 595.0, 549.0, 359.0, 483.0, 547.0, 447.0, 595.0, 666.0, 377.0, 521.0, 446.0, 377.0, 461.0, 372.0, 432.0, 559.0, 423.0, 411.0, 604.0, 432.0, 428.0, 498.0, 549.0, 579.0, 441.0, 529.0, 519.0, 558.0, 549.0, 643.0, 530.0, 537.0, 435.0, 454.0, 543.0, 718.0, 583.0, 749.0, 512.0, 510.0, 658.0, 373.0, 567.0, 375.0, 669.0, 473.0, 394.0, 450.0, 734.0, 390.0, 689.0, 420.0, 547.0, 565.0, 550.0, 586.0, 558.0, 549.0, 368.0, 424.0, 428.0, 373.0, 376.0, 543.0, 409.0, 115.0, 284.0, 372.0, 372.0, 373.0, 416.0, 680.0, 358.0, 540.0, 569.0, 428.0, 609.0, 663.0, 505.0, 549.0, 433.0, 415.0, 372.0, 562.0, 358.0, 783.0, 385.0, 378.0, 372.0, 510.0, 327.0, 572.0, 417.0, 373.0, 508.0, 591.0, 549.0, 421.0, 488.0, 377.0, 339.0, 514.0, 548.0, 626.0, 372.0, 524.0, 381.0, 561.0, 416.0, 549.0, 373.0, 646.0, 415.0, 135.0, 297.0, 300.0, 518.0, 247.0, 432.0, 420.0, 532.0, 397.0, 511.0, 538.0, 408.0, 372.0, 305.0, 549.0, 356.0, 396.0, 288.0, 372.0, 559.0, 558.0, 277.0, 502.0, 344.0, 486.0, 238.0, 549.0, 410.0, 373.0, 372.0, 400.0, 223.0, 549.0, 196.0, 372.0, 278.0, 549.0, 415.0, 305.0, 349.0, 353.0, 314.0, 373.0, 253.0, 413.0, 362.0, 266.0, 519.0, 336.0, 273.0, 502.0, 462.0, 302.0, 558.0, 287.0, 372.0, 215.0, 532.0, 331.0, 391.0, 272.0, 559.0, 321.0, 372.0, 310.0, 513.0, 173.0, 373.0, 191.0, 549.0, 243.0, 503.0, 238.0, 523.0, 253.0, 502.0, 300.0, 502.0, 222.0, 477.0, 268.0, 509.0, 231.0, 373.0, 205.0, 494.0, 445.0, 414.0, 208.0, 503.0, 230.0, 550.0, 360.0, 553.0, 190.0, 549.0, 203.0, 559.0, 79.0, 428.0, 142.0, 316.0, 131.0, 549.0, 197.0, 558.0, 207.0, 440.0, 201.0, 372.0, 180.0, 502.0, 222.0, 322.0, 288.0, 361.0, 281.0, 373.0, 326.0, 375.0, 371.0, 343.0, 321.0, 373.0, 336.0, 427.0, 494.0, 165.0, 553.0, 389.0, 254.0, 291.0, 361.0, 217.0, 549.0, 292.0, 305.0, 315.0, 352.0, 160.0, 540.0, 387.0, 547.0, 289.0, 373.0, 170.0, 550.0, 307.0, 372.0, 405.0, 432.0, 196.0, 550.0, 203.0, 558.0, 352.0, 391.0, 236.0, 550.0, 253.0, 549.0, 400.0, 559.0, 272.0, 360.0, 409.0, 503.0, 207.0, 502.0, 322.0, 334.0, 317.0, 549.0, 257.0, 549.0, 296.0, 423.0, 432.0, 479.0, 439.0, 433.0, 419.0, 413.0, 504.0, 416.0, 409.0, 375.0, 469.0, 442.0, 408.0, 361.0, 422.0, 458.0, 396.0, 427.0, 366.0, 470.0, 489.0, 451.0, 380.0, 364.0, 497.0, 502.0, 456.0, 368.0, 414.0, 456.0, 476.0, 428.0, 407.0, 448.0, 503.0, 463.0, 428.0, 420.0, 433.0, 482.0, 452.0, 412.0, 405.0, 464.0, 506.0, 453.0, 377.0, 413.0, 494.0, 472.0, 428.0, 389.0, 406.0, 490.0, 489.0, 445.0, 412.0, 384.0, 466.0, 458.0, 446.0, 357.0, 409.0, 463.0, 458.0, 433.0, 374.0, 418.0, 470.0, 443.0, 384.0, 355.0, 424.0, 464.0, 432.0, 409.0, 363.0, 457.0, 466.0, 426.0, 385.0, 352.0, 482.0, 455.0, 422.0, 376.0, 393.0, 482.0, 439.0, 392.0, 371.0, 369.0, 472.0, 412.0, 438.0, 327.0, 407.0, 467.0, 449.0, 413.0, 379.0, 367.0, 467.0, 447.0, 405.0, 397.0, 445.0, 494.0, 440.0, 397.0, 376.0, 382.0, 487.0, 435.0, 414.0, 397.0, 431.0, 494.0, 458.0, 450.0, 466.0, 519.0, 536.0, 488.0, 506.0, 490.0, 502.0, 484.0, 513.0, 481.0, 451.0, 446.0, 514.0, 523.0, 461.0, 480.0, 445.0, 477.0, 486.0, 473.0, 472.0, 498.0, 481.0, 510.0, 465.0, 505.0, 504.0, 446.0, 520.0, 515.0, 503.0, 488.0, 485.0, 415.0, 519.0, 510.0, 490.0, 502.0, 464.0, 492.0, 535.0, 490.0, 503.0, 496.0, 467.0, 548.0, 477.0, 507.0, 463.0, 479.0, 471.0, 474.0, 510.0, 480.0, 472.0, 460.0, 502.0, 445.0, 455.0]
        _emg_states = ["MEDIUM", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "MEDIUM", "ACTIVE", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "MEDIUM", "ACTIVE", "ACTIVE", "RELAX", "MEDIUM", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "MEDIUM", "MEDIUM", "ACTIVE", "RELAX", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "ACTIVE", "RELAX", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "RELAX", "ACTIVE", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "RELAX", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "RELAX", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "ACTIVE", "ACTIVE", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "MEDIUM", "RELAX", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "RELAX", "MEDIUM", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "MEDIUM", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "MEDIUM", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "MEDIUM", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "MEDIUM", "RELAX", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "MEDIUM", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "ACTIVE", "MEDIUM", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "MEDIUM", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "ACTIVE", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "ACTIVE", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "ACTIVE", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "ACTIVE", "MEDIUM", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "RELAX", "RELAX", "RELAX", "RELAX", "MEDIUM", "MEDIUM", "RELAX", "MEDIUM", "ACTIVE", "ACTIVE", "MEDIUM", "ACTIVE", "MEDIUM", "ACTIVE", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "RELAX", "ACTIVE", "ACTIVE", "MEDIUM", "MEDIUM", "RELAX", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "ACTIVE", "MEDIUM", "ACTIVE", "ACTIVE", "RELAX", "ACTIVE", "ACTIVE", "ACTIVE", "MEDIUM", "MEDIUM", "RELAX", "ACTIVE", "ACTIVE", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "ACTIVE", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "ACTIVE", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "MEDIUM", "MEDIUM", "ACTIVE", "MEDIUM", "MEDIUM", "MEDIUM", "ACTIVE", "RELAX", "MEDIUM"]
        _avg_emg    = 433.6
        _max_emg    = 783
        _n_emg      = 600
        _dom_state  = "RELAX"

        _sc_map = {"ACTIVE": "#EF5350",   # red
                   "MEDIUM": "#F4A261",   # orange
                   "RELAX":  "#3B82F6"}   # blue

        _fig_emg = go.Figure()
        # Add scatter points by state
        for _st, _sc in _sc_map.items():
            _xi = [_emg_mins[i] for i in range(len(_emg_vals)) if _emg_states[i] == _st]
            _yi = [_emg_vals[i] for i in range(len(_emg_vals)) if _emg_states[i] == _st]
            if _xi:
                _fig_emg.add_trace(go.Scatter(
                    x=_xi, y=_yi, mode="markers", name=_st.title(),
                    marker=dict(size=5, color=_sc, opacity=0.8),
                    hovertemplate="<b>%{y:.0f} µV</b> @ %{x:.1f} min<extra>" + _st + "</extra>"
                ))
        # Add moving average line
        _sm_w = 20
        _smoothed = [sum(_emg_vals[max(0,i-_sm_w):i+1]) / len(_emg_vals[max(0,i-_sm_w):i+1]) for i in range(len(_emg_vals))]
        _fig_emg.add_trace(go.Scatter(
            x=_emg_mins, y=_smoothed, mode="lines", name="Trend (20-pt avg)",
            line=dict(color="#264653", width=3),
            hovertemplate="<b>Trend: %{y:.0f} µV</b><extra></extra>"
        ))
        # Reference thresholds
        _fig_emg.add_hline(y=700, line_dash="dot", line_color="#EF5350", line_width=1.5,
                           annotation_text="Overexertion 700 µV",
                           annotation_position="top right", annotation_font_size=11)
        _fig_emg.add_hline(y=500, line_dash="dot", line_color="#F4A261", line_width=1.5,
                           annotation_text="Fatigue 500 µV",
                           annotation_position="top right", annotation_font_size=11)
        _fig_emg.add_hline(y=300, line_dash="dot", line_color="#81B29A", line_width=1.5,
                           annotation_text="Moderate 300 µV",
                           annotation_position="top right", annotation_font_size=11)
        _fig_emg.update_layout(
            height=400,
            margin=dict(l=15, r=15, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Session Time (minutes)", gridcolor="#E2E8F0",
                       ticksuffix=" min", range=[0, 20], tickfont=dict(size=11)),
            yaxis=dict(title="EMG Amplitude (µV)", gridcolor="#E2E8F0",
                       range=[0, 850], tickfont=dict(size=11)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=10)),
            hovermode="x unified"
        )
        st.plotly_chart(_fig_emg, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # 4. SEGMENTAL LEAN MASS ANALYSIS (pentagon radar chart)
        # =====================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("💪 Segmental Lean Mass Analysis")
        st.caption("Soft Lean Mass — percentage of ideal muscle mass per body segment (≥90% = Normal, <90% = Under)")

        _seg_names  = ["Left Arm", "Trunk", "Right Arm", "Left Leg", "Right Leg"]
        _pcts       = [65.3, 84.8, 69.8, 93.4, 93.9]
        _statuses   = ["Under", "Under", "Under", "Normal", "Normal"]
        _masses     = [1.13, 13.3, 1.21, 5.11, 5.13]
        _changes    = ["0.00", "-0.1", "0.00", "+0.04", "+0.09"]
        _icons      = ["💪", "🎯", "💪", "🦵", "🦵"]

        def _seg_color(p):
            return "#EF5350" if p < 90 else "#2A9D8F"

        def _change_meta(c):
            if c.startswith("-"):    return "#DC2626", "↓"
            elif c in ("0.00","0"):  return "#94A3B8", "→"
            else:                    return "#16A34A", "↑"

        # Pentagon radar — clockwise from top: Trunk, Right Arm, Right Leg, Left Leg, Left Arm
        _radar_order = ["Trunk", "Right Arm", "Right Leg", "Left Leg", "Left Arm"]
        _radar_pcts  = [_pcts[_seg_names.index(s)] for s in _radar_order]
        _radar_pcts_closed = _radar_pcts + [_radar_pcts[0]]
        _radar_theta = _radar_order + [_radar_order[0]]

        _fig_radar = go.Figure()
        _fig_radar.add_trace(go.Scatterpolar(
            r=_radar_pcts_closed,
            theta=_radar_theta,
            fill="toself",
            fillcolor="rgba(42,157,143,0.18)",
            line=dict(color="#2A9D8F", width=2),
            marker=dict(size=7, color=[_seg_color(p) for p in _radar_pcts + [_radar_pcts[0]]]),
            name="% of ideal",
            hovertemplate="%{theta}: %{r:.1f}%<extra></extra>"
        ))
        _fig_radar.add_trace(go.Scatterpolar(
            r=[90]*6,
            theta=_radar_theta,
            fill="none",
            line=dict(color="rgba(46,125,50,0.35)", width=1.5, dash="dash"),
            hoverinfo="skip",
            showlegend=False
        ))
        _fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(240,247,255,0.6)",
                radialaxis=dict(
                    visible=True, range=[0,100],
                    tickvals=[25,50,75,100],
                    tickfont=dict(size=8, color="#94A3B8"),
                    gridcolor="#E2E8F0", linecolor="#E2E8F0"
                ),
                angularaxis=dict(
                    tickmode="array",
                    tickvals=_radar_order,
                    ticktext=[f"<b>{s}</b>" for s in _radar_order],
                    tickfont=dict(size=11),
                    rotation=90, direction="clockwise",
                    gridcolor="#E2E8F0", linecolor="#E2E8F0"
                )
            ),
            showlegend=False,
            height=360,
            margin=dict(l=50, r=50, t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
        )

        _col_fig, _col_cards = st.columns([1, 1.05], gap="large")

        with _col_fig:
            st.plotly_chart(_fig_radar, use_container_width=True)
            st.caption("Dashed ring = 90% ideal threshold")

        with _col_cards:
            st.markdown("#### Soft Lean Mass (kg)")
            for _seg, _mass, _chg, _pct, _stat, _icon in zip(_seg_names, _masses, _changes, _pcts, _statuses, _icons):
                _cc, _arr = _change_meta(_chg)
                _sc = _seg_color(_pct)
                _disp = _chg if _chg.startswith("-") else (f"+{_chg}" if _chg != "0.00" else "0.0")
                _cbg = "#FEE2E2" if _cc == "#DC2626" else "#DCFCE7" if _cc == "#16A34A" else "#F1F5F9"
                st.markdown(
                    f'<div style="background:#fff;border:1px solid #E2E8F0;border-radius:12px;'
                    f'padding:10px 14px;margin-bottom:8px;display:flex;align-items:center;justify-content:space-between;">'
                    f'<div style="display:flex;align-items:center;gap:9px;">'
                    f'<span style="font-size:1.2rem;">{_icon}</span>'
                    f'<div>'
                    f'<div style="font-size:11px;font-weight:600;color:#94A3B8;letter-spacing:0.05em;text-transform:uppercase;">{_seg}</div>'
                    f'<div style="font-size:20px;font-weight:700;color:#0F172A;line-height:1.1;">{_mass}<span style="font-size:12px;font-weight:400;color:#64748B;margin-left:2px;">kg</span></div>'
                    f'<div style="font-size:12px;font-weight:600;color:{_sc};">{_stat} · {_pct:.1f}%</div>'
                    f'</div></div>'
                    f'<div style="background:{_cbg};border-radius:8px;padding:5px 10px;text-align:center;min-width:46px;">'
                    f'<div style="font-size:15px;color:{_cc};">{_arr}</div>'
                    f'<div style="font-size:11px;font-weight:600;color:{_cc};">{_disp}</div>'
                    f'</div></div>',
                    unsafe_allow_html=True
                )
            st.caption("Change from previous measurement")

        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # 5. LONG-TERM MUSCULOSKELETAL HEALTH
        # =====================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📈 Long-Term Musculoskeletal Health")
        st.markdown("**Skeletal Muscle Mass Trend (Last 30 Days)**")
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        base_mass = np.linspace(68, 69.8, 30)
        noise = np.random.normal(0, 0.1, 30)
        mass_values = base_mass + noise
        muscle_mass_df = pd.DataFrame({'Date': dates, 'Muscle Mass (kg)': mass_values})
        fig_mass = go.Figure()
        fig_mass.add_trace(go.Scatter(x=muscle_mass_df['Date'], y=muscle_mass_df['Muscle Mass (kg)'], mode='lines+markers', line=dict(color='#2A9D8F', width=3), marker=dict(size=6, color='#2A9D8F'), name='Muscle Mass'))
        fig_mass.update_layout(xaxis_title="Date", yaxis_title="Muscle Mass (kg)", height=400, margin=dict(l=20, r=20, t=20, b=20), plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#264653'))
        st.plotly_chart(fig_mass, use_container_width=True)
        st.caption("Showing estimated skeletal muscle mass trajectory based on bio‑impedance analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # 6. SESSION AUDIT TRAIL
        # =====================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📋 Session Audit Trail")

        df_logs = read_logs(patient_id)

        if df_logs.empty:
            st.markdown("""
            <div style="text-align:center;padding:40px 0;color:#94A3B8;">
                <div style="font-size:2rem;margin-bottom:8px;">📂</div>
                <div style="font-size:0.95rem;font-weight:600;color:#64748B;">No audit events recorded for this patient.</div>
                <div style="font-size:0.8rem;margin-top:4px;">Events will appear here once a session is started.</div>
            </div>""", unsafe_allow_html=True)
        else:
            # Summary stat tiles
            _au_starts  = df_logs[df_logs['event'] == 'SESSION_START'].shape[0]
            _au_stops   = df_logs[df_logs['event'] == 'SESSION_STOP'].shape[0]
            _au_emrg    = df_logs[df_logs['event'] == 'EMERGENCY_STOP'].shape[0]
            _au_params  = df_logs[df_logs['event'] == 'PARAM_CHANGE'].shape[0]
            _au_last    = str(df_logs['ts'].iloc[0])[:16].replace("T", " ")

            st.markdown(f"""
            <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:10px;margin-bottom:18px;">
                <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:700;color:#15803D;">{_au_starts}</div>
                    <div style="font-size:0.72rem;font-weight:600;color:#166534;">SESSIONS STARTED</div>
                </div>
                <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:700;color:#B45309;">{_au_stops}</div>
                    <div style="font-size:0.72rem;font-weight:600;color:#92400E;">SESSIONS STOPPED</div>
                </div>
                <div style="background:{"#FEF2F2" if _au_emrg > 0 else "#F8FAFC"};border:1px solid {"#FECACA" if _au_emrg > 0 else "#E2E8F0"};border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:700;color:{"#DC2626" if _au_emrg > 0 else "#94A3B8"};">{_au_emrg}</div>
                    <div style="font-size:0.72rem;font-weight:600;color:{"#991B1B" if _au_emrg > 0 else "#64748B"};">EMERGENCY STOPS</div>
                </div>
                <div style="background:#F5F3FF;border:1px solid #DDD6FE;border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:1.6rem;font-weight:700;color:#6D28D9;">{_au_params}</div>
                    <div style="font-size:0.72rem;font-weight:600;color:#5B21B6;">PARAM ADJUSTMENTS</div>
                </div>
                <div style="background:#F0F9FF;border:1px solid #BAE6FD;border-radius:10px;padding:12px;text-align:center;">
                    <div style="font-size:0.82rem;font-weight:700;color:#0369A1;">{_au_last}</div>
                    <div style="font-size:0.72rem;font-weight:600;color:#075985;">LAST EVENT</div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Filter bar
            _au_fc1, _au_fc2, _au_fc3 = st.columns([1.2, 2, 0.8])
            _au_all_evts = ["All Events"] + sorted(df_logs['event'].unique().tolist())
            with _au_fc1:
                _au_evt_filter = st.selectbox("Event Type", _au_all_evts,
                                              key="rec_audit_evt_filter", label_visibility="collapsed")
            with _au_fc2:
                _au_search = st.text_input("Search", placeholder="🔍  Search by event or details…",
                                           key="rec_audit_search", label_visibility="collapsed")
            with _au_fc3:
                _au_show_n = st.selectbox("Show", [20, 50, 100, 200],
                                          key="rec_audit_show_n", label_visibility="collapsed")

            _au_filtered = df_logs.copy()
            if _au_evt_filter != "All Events":
                _au_filtered = _au_filtered[_au_filtered['event'] == _au_evt_filter]
            if _au_search:
                _au_filtered = _au_filtered[
                    _au_filtered['details'].str.contains(_au_search, case=False, na=False) |
                    _au_filtered['event'].str.contains(_au_search, case=False, na=False)
                ]
            _au_filtered = _au_filtered.head(_au_show_n)

            # Table header
            st.markdown("""
            <div style="display:grid;grid-template-columns:160px 190px 1fr;gap:0;
                        background:#F1F5F9;border:1px solid #E2E8F0;
                        border-radius:8px 8px 0 0;padding:8px 14px;margin-top:8px;">
                <span style="font-size:0.72rem;font-weight:700;color:#64748B;">DATE / TIME</span>
                <span style="font-size:0.72rem;font-weight:700;color:#64748B;">EVENT</span>
                <span style="font-size:0.72rem;font-weight:700;color:#64748B;">DETAILS</span>
            </div>""", unsafe_allow_html=True)

            _au_ecfg = {
                "SESSION_START":     ("#DCFCE7", "#15803D", "#F0FDF4", "Session Started"),
                "SESSION_STOP":      ("#FEF9C3", "#92400E", "#FFFBEB", "Session Stopped"),
                "SESSION_PAUSE":     ("#DBEAFE", "#1D4ED8", "#EFF6FF", "Session Paused"),
                "EMERGENCY_STOP":    ("#FEE2E2", "#991B1B", "#FFF5F5", "⚠ Emergency Stop"),
                "PARAM_CHANGE":      ("#EDE9FE", "#5B21B6", "#F5F3FF", "Parameter Adjusted"),
                "INBODY_OCR_UPLOAD": ("#CFFAFE", "#0E7490", "#F0FDFF", "InBody Scan Uploaded"),
                "APPROVAL_GRANTED":  ("#DCFCE7", "#15803D", "#F0FDF4", "✅ Approval Granted"),
                "APPROVAL_DECLINED": ("#FEF9C3", "#92400E", "#FFFBEB", "❌ Approval Declined"),
            }
            _au_default = ("#E2E8F0", "#475569", "#F8FAFC", "System Event")

            _au_rows_html = ""
            for _au_i, (_, _au_row) in enumerate(_au_filtered.iterrows()):
                _au_dc, _au_tc, _au_bg, _au_lbl = _au_ecfg.get(_au_row['event'], _au_default)
                _au_det = str(_au_row['details']) if _au_row['details'] else "—"
                _au_ts  = str(_au_row['ts'])[:19]
                _au_date = _au_ts[:10]
                _au_time = _au_ts[11:19] if len(_au_ts) > 10 else ""
                _au_border_top = "1px solid #E2E8F0" if _au_i > 0 else "none"
                _au_radius = "0 0 8px 8px" if _au_i == len(_au_filtered) - 1 else "0"
                _au_rows_html += (
                    f'<div style="display:grid;grid-template-columns:160px 190px 1fr;gap:0;'
                    f'background:{_au_bg};border-left:1px solid #E2E8F0;border-right:1px solid #E2E8F0;'
                    f'border-top:{_au_border_top};border-bottom:1px solid #E2E8F0;'
                    f'border-radius:{_au_radius};padding:9px 14px;align-items:center;">'
                    f'<div>'
                    f'<div style="font-size:0.78rem;font-weight:600;color:#1E293B;font-family:monospace;">{_au_date}</div>'
                    f'<div style="font-size:0.72rem;color:#94A3B8;font-family:monospace;">{_au_time}</div>'
                    f'</div>'
                    f'<div style="display:flex;align-items:center;gap:7px;">'
                    f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;'
                    f'background:{_au_dc};border:1.5px solid {_au_tc};flex-shrink:0;"></span>'
                    f'<span style="font-size:0.78rem;font-weight:600;color:{_au_tc};">{_au_lbl}</span>'
                    f'</div>'
                    f'<div style="font-size:0.77rem;color:#475569;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">'
                    f'{_au_det}</div>'
                    f'</div>'
                )
            st.markdown(_au_rows_html, unsafe_allow_html=True)
            st.caption(f"Showing {len(_au_filtered)} of {len(df_logs)} total events · Patient: {patient_id}")

        st.divider()
        _au_col1, _au_col2 = st.columns(2)
        with _au_col1:
            _au_csv = df_logs.to_csv(index=False).encode("utf-8") if not df_logs.empty else b""
            st.download_button(
                "⬇️ Export Audit Trail (CSV)", _au_csv,
                f"audit_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv", use_container_width=True, disabled=df_logs.empty
            )
        with _au_col2:
            if not df_logs.empty:
                _au_txt  = f"CLINICAL AUDIT TRAIL\n{'='*50}\n"
                _au_txt += f"Patient ID : {patient_id}\n"
                _au_txt += f"Generated  : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                _au_txt += f"{'='*50}\n\n"
                for _, _au_r in df_logs.iterrows():
                    _au_txt += f"[{_au_r['ts']}]  {_au_r['event']:<22}  {_au_r['details']}\n"
                st.download_button(
                    "📄 Export Formatted Report (TXT)",
                    _au_txt.encode("utf-8"),
                    f"audit_{patient_id}_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                    "text/plain", use_container_width=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # =====================================================
        # 7. EXPORT REPORT
        # =====================================================
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📄 Export Report")
        report_text = generate_report(patient_id, muscle_mass_df, pain_score_progress, fatigue_progress)
        st.download_button(label="📄 Download Progress Report (TXT)", data=report_text, file_name=f"Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.txt", mime="text/plain", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

       

    # ---------- TAB 5: CLINICAL AI CHAT (improved layout) ----------
    with tab_chat:
        # Custom CSS for chat styling
        st.markdown("""
        <style>
        /* Chat container styling */
        .chat-message-user {
            background-color: #EFF6FF;
            border-radius: 20px;
            padding: 12px 18px;
            margin: 8px 0;
            border-left: 5px solid #3B82F6;
            max-width: 85%;
            margin-left: auto;
            word-wrap: break-word;
        }
        .chat-message-assistant {
            background-color: #F8FAFC;
            border-radius: 20px;
            padding: 12px 18px;
            margin: 8px 0;
            border-left: 5px solid #10B981;
            max-width: 85%;
            margin-right: auto;
            word-wrap: break-word;
        }
        .suggestion-chip {
            background-color: #F1F5F9;
            border-radius: 40px;
            padding: 8px 16px;
            margin: 5px;
            display: inline-block;
            font-size: 0.9rem;
            font-weight: 500;
            color: #1E293B;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid #E2E8F0;
            text-align: center;
        }
        .suggestion-chip:hover {
            background-color: #E2E8F0;
            transform: translateY(-1px);
        }
        .chat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.markdown("### 🧠 AI Clinical Assistant")
        st.info("Ask any question about rehabilitation protocols, treatment guidelines, or patient management.")

        # Header with clear button
        col_title, col_clear = st.columns([3, 1])
        with col_clear:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.rag_messages = []
                st.rerun()

        # Suggested questions as clickable chips (using markdown + button simulation)
        st.markdown("#### 💡 Suggested questions")
        suggested = [
            "Contraindications of EMS therapy?",
            "EMS intensity for sarcopenia?",
            "Quadriceps electrode placement?",
            "Signs of muscle overwork?",
            "Difference between EMS and TENS?",
            "Frequency of EMS sessions?"
        ]
        # Render as columns of chips
        cols = st.columns(3)
        for i, q in enumerate(suggested):
            with cols[i % 3]:
                if st.button(q, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.rag_chat_input = q
                    st.rerun()

        # Initialize chat history
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = []

        # Display chat messages with custom styling
        for msg in st.session_state.rag_messages:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(msg["content"])
                    if "references" in msg and msg["references"]:
                        with st.expander("📚 References"):
                            for ref in msg["references"]:
                                st.caption(f"• {ref}")

        # Chat input (auto-populate from suggested chips)
        if "rag_chat_input" not in st.session_state:
            st.session_state.rag_chat_input = ""

        prompt = st.chat_input("Ask a clinical question...", key="rag_input")
        if prompt is None and st.session_state.rag_chat_input:
            prompt = st.session_state.rag_chat_input
            st.session_state.rag_chat_input = ""

        if prompt:
            # Add user message
            st.session_state.rag_messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get answer
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base..."):
                    answer, references = call_rag_api(prompt)
                    st.markdown(answer)
                    if references:
                        with st.expander("📚 References"):
                            for ref in references:
                                st.caption(f"• {ref}")

            # Store assistant message
            st.session_state.rag_messages.append({"role": "assistant", "content": answer, "references": references})

else:   # CAREGIVER VIEW – simplified, elderly‑friendly dashboard
    # ==========================================
    # SIMPLIFIED, ELDERLY-FRIENDLY DASHBOARD
    # ==========================================
    st.markdown("""
    <style>
        h1, h2, h3 {
            font-size: 2rem !important;
        }
        .care-card {
            background: white;
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 18px;
            text-align: center;
            border: 1px solid #E2E8F0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        .care-title {
            font-size: 1.3rem;
            color: #475569;
            font-weight: 700;
        }
        .care-value {
            font-size: 2.8rem;
            font-weight: 900;
            color: #0F172A;
            margin-top: 6px;
        }
        .care-desc {
            font-size: 1rem;
            color: #475569;
            margin-top: 6px;
        }
        div.stButton > button {
            font-size: 1.3rem !important;
            height: 60px !important;
            border-radius: 16px !important;
            font-weight: 700 !important;
        }
        div[data-testid="stSlider"] label {
            font-size: 1.3rem !important;
            font-weight: 700 !important;
        }
        .stAlert {
            font-size: 1.2rem !important;
        }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("## 👨‍👩‍👧 Caregiver View")
    st.caption("Simple monitoring screen for patient comfort and safety.")

    tele = st.session_state.telemetry
    latest_emg = tele['emg'].iloc[-1] if not tele.empty else 0
    pred_label, pred_icon, pred_desc = predict_muscle_state(latest_emg)

    # status emoji
    if pred_label == "Relaxed":
        status_emoji = "😌"
    elif pred_label == "Moderate Activity":
        status_emoji = "💪"
    elif pred_label == "Muscle Fatigue":
        status_emoji = "😩"
    else:
        status_emoji = "⚠️"

    # card background based on muscle state
    if pred_label in ["Muscle Fatigue", "Overexertion"]:
        status_color = "#FEE2E2"
        border_color = "#EF4444"
    elif pred_label == "Moderate Activity":
        status_color = "#FEF3C7"
        border_color = "#F59E0B"
    else:
        status_color = "#DCFCE7"
        border_color = "#22C55E"

    st.markdown(f"""
    <div class="care-card" style="background:{status_color}; border-color:{border_color};">
        <div style="font-size:3.5rem;">{status_emoji}</div>
        <div class="care-title">Current Muscle Condition</div>
        <div class="care-value">{pred_label}</div>
        <div class="care-desc">{pred_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    # Two big cards: EMG and Gait
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="care-card">
            <div class="care-title">📈 EMG Signal</div>
            <div class="care-value" style="color:#2A9D8F;">{latest_emg:.0f}</div>
            <div class="care-desc">microvolts (µV)</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        gait_status = st.session_state.ml_prediction
        if gait_status == "NORMAL":
            gait_text = "Normal"
            gait_icon = "✅"
            gait_bg = "#DCFCE7"
            gait_border = "#22C55E"
        else:
            gait_text = "Check Needed"
            gait_icon = "⚠️"
            gait_bg = "#FEE2E2"
            gait_border = "#EF4444"
        st.markdown(f"""
        <div class="care-card" style="background:{gait_bg}; border-color:{gait_border};">
            <div style="font-size:2.5rem;">{gait_icon}</div>
            <div class="care-title">Gait Pattern</div>
            <div class="care-value">{gait_text}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("## Patient Feeling")
    pain = st.slider("😖 Pain Level", 0, 10, value=st.session_state.get("live_pain", 2), key="caregiver_pain")
    fatigue = st.slider("😴 Fatigue Level", 0, 10, value=st.session_state.get("live_fatigue", 4), key="caregiver_fatigue")
    st.session_state.live_pain = pain
    st.session_state.live_fatigue = fatigue

    if pain > 7:
        st.error("🔴 High pain. Stop therapy and tell the clinician.")
    elif fatigue > 7:
        st.warning("🟡 Patient is very tired. Please take a rest.")
    elif pred_label in ["Muscle Fatigue", "Overexertion"]:
        st.warning("🟡 Muscle activity is high. Monitor the patient closely.")
    else:
        st.success("🟢 Patient condition looks okay.")

    # ===== EXPANDER: LEAN MUSCLE ANALYSIS (collapsible, less intrusive) =====
    with st.expander("💪 Show Muscle Health by Body Part (detailed)"):
        st.markdown("""
        <style>
            .muscle-card {
                background: white;
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 14px;
                border: 1px solid #E2E8F0;
            }
            .muscle-title {
                font-size: 1.2rem;
                font-weight: 700;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            .progress-bar-bg {
                background-color: #E2E8F0;
                border-radius: 30px;
                height: 32px;
                width: 100%;
                margin: 10px 0;
            }
            .progress-fill {
                height: 32px;
                border-radius: 30px;
                display: flex;
                align-items: center;
                justify-content: flex-end;
                padding-right: 12px;
                color: white;
                font-weight: 700;
                font-size: 1rem;
            }
            .muscle-stats {
                font-size: 0.95rem;
                color: #475569;
            }
        </style>
        """, unsafe_allow_html=True)

        segments = ["Left Arm", "Trunk", "Right Arm", "Left Leg", "Right Leg"]
        percentages = [65.3, 84.8, 69.8, 93.4, 93.9]
        masses = [1.13, 13.3, 1.21, 5.11, 5.13]
        icons = ["💪", "🎯", "💪", "🦵", "🦵"]

        for seg, pct, mass, icon in zip(segments, percentages, masses, icons):
            bar_color = "#22C55E" if pct >= 90 else "#EF4444"
            status_text = "Normal ✅" if pct >= 90 else "Weak ⚠️"
            st.markdown(f"""
            <div class="muscle-card">
                <div class="muscle-title">
                    <span style="font-size:1.8rem;">{icon}</span>
                    <span>{seg}</span>
                    <span style="margin-left: auto; color: {bar_color};">{status_text}</span>
                </div>
                <div class="progress-bar-bg">
                    <div class="progress-fill" style="background-color: {bar_color}; width: {pct}%;">
                        {pct:.1f}%
                    </div>
                </div>
                <div class="muscle-stats">
                    Mass: <strong>{mass} kg</strong> &nbsp; (vs ideal)
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.caption("✅ Normal = muscle mass ≥90% of ideal. ⚠️ Weak = below 90%.")

    st.info("Use START, PAUSE, or STOP buttons above. Press Emergency STOP if the patient feels unsafe.")

    with st.expander("📈 Show EMG trend"):
        if not tele.empty:
            st.line_chart(tele.set_index("t")["emg"], height=260)
        else:
            st.write("No data yet.")

# ==========================================
# 11. AUTO REFRESH
# ==========================================
if st.session_state.system_status == "ACTIVE":
    time.sleep(1.0)   # refresh telemetry chart every ~1 s; ML API is gated separately
    st.rerun()
