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
ss_init("esp_state", None)
ss_init("esp_mode", None)
ss_init("esp_channel", None)
ss_init("ml_summary", {})
ss_init("session_summary_generated", False)
ss_init("session_summary_text", "")
ss_init("last_ml_call_time", 0)
ML_CALL_INTERVAL = 2  # seconds between ML API calls
ss_init("ml_latest", {})
ss_init("ml_probabilities", [])
ss_init("ml_session", {})
ss_init("ml_thread", None)
ss_init("ml_pending", False)
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

SESSION_DURATION_MINUTES = 20   # fixed protocol duration in minutes

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

    # ── Derived stats ─────────────────────────────────────────────────────
    avg_emg   = tele['emg'].mean()
    max_emg   = tele['emg'].max()
    min_emg   = tele['emg'].min()
    std_emg   = tele['emg'].std()
    q         = max(1, len(tele) // 4)
    emg_early = tele['emg'].iloc[:q].mean()
    emg_late  = tele['emg'].iloc[-q:].mean()
    emg_trend = emg_late - emg_early
    trend_desc = (
        f"increased by {emg_trend:.1f} µV (progressive recruitment)"
        if emg_trend > 10 else
        f"decreased by {abs(emg_trend):.1f} µV (possible fatigue or accommodation)"
        if emg_trend < -10 else
        f"remained stable (Δ{emg_trend:+.1f} µV)"
    )
    n_total   = len(tele)
    n_relaxed = (tele['emg'] <= 300).sum()
    n_mod     = ((tele['emg'] > 300) & (tele['emg'] <= 500)).sum()
    n_fat     = ((tele['emg'] > 500) & (tele['emg'] <= 700)).sum()
    n_over    = (tele['emg'] > 700).sum()
    pct_act   = (n_mod + n_fat + n_over) / n_total * 100

    # ── Session params — read from frozen snapshot ────────────────────────
    intensity   = st.session_state.frozen_intensity
    frequency   = st.session_state.frozen_frequency
    pulse_width = st.session_state.frozen_pulse_width
    duty_on     = st.session_state.frozen_duty_on
    duty_off    = st.session_state.frozen_duty_off
    pain        = st.session_state.frozen_pain
    fatigue     = st.session_state.frozen_fatigue
    # ML values: read from live state (preserved on STOPPED) with frozen as fallback
    gait        = st.session_state.ml_prediction    if st.session_state.ml_prediction    not in ("WAITING", "")  else st.session_state.frozen_ml_prediction
    ml_conf     = st.session_state.ml_probability   if st.session_state.ml_probability   > 0                     else st.session_state.frozen_ml_probability
    ml_session  = st.session_state.ml_session       if st.session_state.ml_session                               else st.session_state.frozen_ml_session
    ml_latest   = st.session_state.ml_latest        if st.session_state.ml_latest                                else st.session_state.frozen_ml_latest
    ml_probs    = st.session_state.ml_probabilities if st.session_state.ml_probabilities                         else st.session_state.frozen_ml_probabilities
    ml_summary_snap = st.session_state.ml_summary   if st.session_state.ml_summary                              else st.session_state.frozen_ml_summary

    conf_pct    = ml_conf * 100 if ml_conf <= 1.0 else ml_conf
    rms_val     = ml_latest.get("rms_recto_femoral", "N/A")
    spread_val  = ml_latest.get("rms_signal_spread", "N/A")
    std_val     = ml_latest.get("rms_signal_std", "N/A")
    ml_avg      = ml_session.get("avg", "N/A")
    ml_min      = ml_session.get("min", "N/A")
    ml_max      = ml_session.get("max", "N/A")
    readings    = ml_session.get("count", "N/A")

    # Snapshot ML summary for display in Records tab
    st.session_state.last_ml_summary_snapshot = dict(ml_summary_snap)

    conditions = ', '.join(condition_tags) if condition_tags else 'general rehabilitation'

    # Use protocol intensity label — if frozen intensity is 0 it means the device
    # controlled current independently (ESP32 mode); use a neutral description
    intensity_desc = f"{intensity} mA" if intensity > 0 else "device-controlled current (intensity set by ESP32)"
    intensity_action = (
        f"set by the ESP32 controller — verify the actual delivered current level from the device log"
        if intensity == 0 else
        f"{intensity} mA"
    )

    # ── Build fallback (factual base the model must preserve) ─────────────
    fallback = {
        "title": f"{gait} recto femoris EMG pattern — {SESSION_DURATION_MINUTES}-min EMS session ({conditions})",
        "summary": (
            f"A {SESSION_DURATION_MINUTES}-minute EMS session was completed for patient {patient_id} "
            f"(age group {age_group}, {conditions}) at {frequency} Hz, "
            f"{pulse_width} µs pulse width, {duty_on}s ON / {duty_off}s OFF duty cycle"
            f"{f', {intensity} mA stimulation intensity' if intensity > 0 else ''}. "
            f"The recto femoris EMG averaged {avg_emg:.1f} µV (max {max_emg:.1f}, min {min_emg:.1f}, "
            f"STD {std_emg:.1f} µV) across {n_total} samples, with signal amplitude that {trend_desc}. "
            f"Active muscle recruitment occurred in {pct_act:.0f}% of the session "
            f"({n_mod} moderate, {n_fat} fatigue-range, {n_over} overexertion samples). "
            f"The ML gait classifier returned {gait} at {conf_pct:.1f}% confidence, "
            f"with final patient-reported pain of {pain}/10 and fatigue of {fatigue}/10."
        ),
        "interpretation": [
            f"Recto Femoris RMS = {rms_val} µV (session avg/min/max: {ml_avg}/{ml_min}/{ml_max}): "
            f"this average signal level reflects the muscle's response to {intensity_desc} "
            f"and should be compared to baseline for a {age_group}-year-old patient with {conditions}.",
            f"Signal Spread = {spread_val} and STD = {std_val}: the within-window variability indicates "
            f"{'stable, consistent motor unit firing' if float(str(std_val).replace('N/A','0') or 0) < 5 else 'notable signal fluctuation, suggesting irregular motor unit recruitment or movement artefact'}. "
            f"EMG STD across the full session was {std_emg:.1f} µV.",
            f"Gait classification: {gait} at {conf_pct:.1f}% confidence with "
            f"{n_relaxed} relaxed / {n_mod} moderate / {n_fat} fatigue-range / {n_over} overexertion samples — "
            f"active recruitment was {pct_act:.0f}% of session time, "
            f"and the EMG trend {trend_desc}."
        ],
        "actions": [
            f"{'Maintain' if gait == 'Normal' else 'Review'} the stimulation protocol "
            f"({frequency} Hz, {pulse_width} µs, {intensity_action}) for the next session — "
            f"{'Normal gait classification supports protocol continuity' if gait == 'Normal' else 'Abnormal gait classification warrants reassessment of electrode placement and stimulation parameters'}.",
            f"{'Increase stimulation intensity to improve recruitment' if pct_act < 50 else 'Active recruitment was sufficient'} "
            f"({pct_act:.0f}% of session {'is below' if pct_act < 50 else 'meets'} the >50% target for effective EMS rehabilitation"
            f"{f'; consider increasing above {intensity} mA' if pct_act < 50 and intensity > 0 else ''}).",
            f"{'Reduce duty cycle ON-time or allow longer rest between sets' if fatigue >= 6 else 'Maintain the current duty cycle'} "
            f"(currently {duty_on}s ON / {duty_off}s OFF): patient reported fatigue {fatigue}/10 "
            f"({'high — rest-to-work ratio should be reviewed' if fatigue >= 6 else 'acceptable'}).",
            f"{'Address pain management — consider adjusting electrode position or reducing stimulation intensity' if pain >= 5 else 'Continue documenting pain score each session'}: "
            f"patient reported pain {pain}/10 "
            f"({'exceeds comfort threshold of 5/10' if pain >= 5 else 'within acceptable range'})."
        ]
    }

    # ── Probability and feature lines for the prompt ───────────────────────
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
- Do not use vague advice such as "monitor closely" unless you explain what to monitor.
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

    # Parse JSON response — fall back to the pre-computed fallback if the model fails
    parsed = None
    if answer and "Error" not in answer:
        try:
            clean = answer.strip().replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean)
        except Exception:
            parsed = None

    st.session_state.session_summary_text = parsed if parsed else fallback

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
            st.session_state.intensity = 15   # reset to default on fresh start
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
        freeze_session_state()   # snapshot ML + params before clearing
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
        freeze_session_state()   # snapshot ML + params before clearing
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
        # Only wipe ML display state when truly idle/paused — not on STOPPED,
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
        st.session_state.elapsed_time = 0.0   # reset only after summary has captured the duration

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
        col_left, col_ml = st.columns([1, 1.6])

        with col_left:
            # ── Safety & Optimization ──────────────────────────────────────
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("🛡️ Safety & Rules")
            if st.session_state.system_status == "ACTIVE":
                pain_val = st.session_state.get("live_pain", 0)
                fatigue_val = st.session_state.get("live_fatigue", 0)
                if pain_val >= 6:
                    st.markdown("""<div class="alert-box alert-risk"><strong>High Pain Detected</strong><br>Pain Score > 6<br>Action: Reduce intensity 20% (PAIN-01)</div>""", unsafe_allow_html=True)
                elif fatigue_val >= 7:
                    st.markdown("""<div class="alert-box alert-info"><strong>High Fatigue</strong><br>Fatigue > 7<br>Action: Increase OFF time (ONOFF-04)</div>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<div class="alert-box alert-safe"><strong>System Nominal</strong><br>All parameters within safety limits.<br>Action: Maintain current protocol (MAIN-01)</div>""", unsafe_allow_html=True)
            else:
                st.caption("System inactive — start session to monitor safety rules.")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── Patient Feedback ───────────────────────────────────────────
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("💬 Patient Feedback")
            pain    = st.slider("Pain Score (0–10)",    0, 10, value=st.session_state.get("live_pain",    2), key="live_pain")
            fatigue = st.slider("Fatigue Level (0–10)", 0, 10, value=st.session_state.get("live_fatigue", 4), key="live_fatigue")
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
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("Gait Pathology (ML Engine)")
            if st.session_state.get("ml_pending", False):
                st.caption("🔄 Fetching latest prediction…")

            res = st.session_state.ml_prediction
            prob = st.session_state.ml_probability

            if st.session_state.system_status == "ACTIVE":

                ERROR_PREFIXES = ("API Timeout", "API Offline", "API Error", "Invalid API Response")
                is_error = any(res.startswith(p) for p in ERROR_PREFIXES) if isinstance(res, str) else False

                # Show loading placeholder until first real result arrives
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
                    tip  = "The ngrok tunnel may be down or the Flask backend is not running." if "Offline" in res else \
                           "The backend took too long to respond. It will retry automatically." if "Timeout" in res else \
                           "Check the backend logs for details."
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

                    if conf_pct >= 75:
                        conf_note = "High confidence. The model result is relatively stable for this reading."
                    elif conf_pct >= 50:
                        conf_note = "Moderate confidence. Consider monitoring additional readings."
                    else:
                        conf_note = "Low confidence. Result may be unreliable — check sensor placement."

                    # ── Predicted Class + Confidence ──────────────────────────
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

                    # ── Three feature metric cards ─────────────────────────────
                    if latest:
                        rms_val    = latest.get('rms_recto_femoral', 0)
                        spread_val = latest.get('rms_signal_spread', 0)
                        std_val    = latest.get('rms_signal_std', 0)

                        st.markdown(f"""
                        <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:14px;">
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                                        border-radius:12px; padding:14px 10px;">
                                <div style="font-size:0.68rem; font-weight:700; color:#78909C;
                                            letter-spacing:0.07em; margin-bottom:6px;">
                                    RECTO FEMORIS RMS
                                </div>
                                <div style="font-size:1.6rem; font-weight:800; color:#0F172A;
                                            margin-bottom:6px;">{rms_val:.2f}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; line-height:1.4;">
                                    Average EMG level from the recto femoris sensor in the latest cleaned 1-second window.
                                </div>
                            </div>
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                                        border-radius:12px; padding:14px 10px;">
                                <div style="font-size:0.68rem; font-weight:700; color:#78909C;
                                            letter-spacing:0.07em; margin-bottom:6px;">
                                    SIGNAL SPREAD
                                </div>
                                <div style="font-size:1.6rem; font-weight:800; color:#0F172A;
                                            margin-bottom:6px;">{spread_val:.0f}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; line-height:1.4;">
                                    Formula: max − min after cleaning. Shows the signal range within the 1-second window.
                                </div>
                            </div>
                            <div style="background:#F8FAFC; border:1px solid #E2E8F0;
                                        border-radius:12px; padding:14px 10px;">
                                <div style="font-size:0.68rem; font-weight:700; color:#78909C;
                                            letter-spacing:0.07em; margin-bottom:6px;">
                                    SIGNAL STD
                                </div>
                                <div style="font-size:1.6rem; font-weight:800; color:#0F172A;
                                            margin-bottom:6px;">{std_val:.4f}</div>
                                <div style="font-size:0.72rem; color:#94A3B8; line-height:1.4;">
                                    Standard deviation of cleaned EMG samples. Shows how much the signal fluctuates around the average.
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Probability bars ───────────────────────────────────────
                    if probabilities:
                        st.markdown("**Prediction Probability**")
                        for p in probabilities:
                            label_p = p.get("label", "Unknown")
                            prob_p  = float(p.get("probability", 0))
                            st.progress(prob_p / 100, text=f"{label_p}: {prob_p:.2f}%")

                    # ── Session Summary row ────────────────────────────────────
                    if session:
                        st.markdown(f"""
                        <div style="background:#F1F5F9; border:1px solid #E2E8F0; border-radius:10px;
                                    padding:12px 16px; margin:10px 0;">
                            <div style="font-weight:700; font-size:0.9rem; color:#1E293B;
                                        margin-bottom:4px;">Session Summary</div>
                            <div style="font-size:0.85rem; color:#475569;">
                                Readings: <strong>{session.get('count', '—')}</strong> &nbsp;|&nbsp;
                                Average: <strong>{session.get('avg', '—')}</strong> &nbsp;|&nbsp;
                                Min: <strong>{session.get('min', '—')}</strong> &nbsp;|&nbsp;
                                Max: <strong>{session.get('max', '—')}</strong>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.caption("📋 Full AI analysis available in the **Records & Reports** tab after session ends.")

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
                            st.toast("InBody OCR data saved!", icon="✅")
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
        sess_summary = st.session_state.session_summary_text
        if sess_summary:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("📝 AI Session Summary")

            # ── Session stats header row ──────────────────────────────────
            tele_snap = st.session_state.telemetry
            gait_res  = st.session_state.ml_prediction
            is_abn    = gait_res in ("ABNORMAL", "Abnormal")

            stat_cols = st.columns(5)
            stat_cols[0].metric("⏱️ Duration",    f"{SESSION_DURATION_MINUTES} min")
            stat_cols[1].metric("📈 Avg EMG",     f"{tele_snap['emg'].mean():.1f} µV" if not tele_snap.empty else "—")
            stat_cols[2].metric("🦶 Gait Result",  gait_res)
            stat_cols[3].metric("😣 Pain",         f"{st.session_state.live_pain}/10")
            stat_cols[4].metric("😴 Fatigue",      f"{st.session_state.live_fatigue}/10")

            st.divider()

            # ── Clinical session summary (structured JSON from RAG) ────────
            st.markdown("#### 🏥 Clinical Session Summary")

            if isinstance(sess_summary, dict) and "error" not in sess_summary:
                # Title
                if sess_summary.get("title"):
                    st.markdown(f"**{sess_summary['title']}**")

                # Summary paragraph
                if sess_summary.get("summary"):
                    st.markdown(
                        f"<p style='font-size:0.88rem; color:#475569; line-height:1.7; margin:10px 0 14px;'>"
                        f"{sess_summary['summary']}</p>",
                        unsafe_allow_html=True
                    )

                # Interpretation + Actions side by side
                col_interp, col_actions = st.columns(2)
                with col_interp:
                    if sess_summary.get("interpretation"):
                        st.markdown("**Signal Interpretation**")
                        for item in sess_summary["interpretation"]:
                            st.markdown(f"- {item}")
                with col_actions:
                    if sess_summary.get("actions"):
                        st.markdown("**Recommended Actions**")
                        for item in sess_summary["actions"]:
                            st.markdown(f"- {item}")

            elif isinstance(sess_summary, dict) and "error" in sess_summary:
                st.warning(sess_summary["error"])
            else:
                # Legacy plain text fallback
                st.markdown(sess_summary)

            st.divider()
            if st.button("🔄 Regenerate Summary", key="regenerate_summary"):
                st.session_state.session_summary_generated = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Segmental Lean Mass Analysis 
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("💪 Segmental Lean Mass Analysis")
        st.caption("Soft Lean Mass — percentage of ideal muscle mass per body segment (≥90% = Normal, <90% = Under)")

        segments = ["Left Arm", "Trunk",  "Right Arm", "Left Leg", "Right Leg"]
        pcts     = [65.3,        84.8,     69.8,         93.4,       93.9      ]
        statuses = ["Under",    "Under",   "Under",      "Normal",   "Normal"  ]
        masses   = [1.13,        13.3,     1.21,         5.11,       5.13      ]
        changes  = ["0.00",     "-0.1",   "0.00",       "+0.04",    "+0.05"   ]

        def seg_color(p):
            return "#EF5350" if p < 90 else "#2A9D8F"

        def change_meta(c):
            if c.startswith("-"):      return "#DC2626", "↓"
            elif c in ("0.00", "0"):   return "#94A3B8", "→"
            else:                      return "#16A34A", "↑"

        la_c = seg_color(pcts[0]); tr_c = seg_color(pcts[1])
        ra_c = seg_color(pcts[2]); ll_c = seg_color(pcts[3]); rl_c = seg_color(pcts[4])

        col_fig, col_cards = st.columns([1, 1], gap="large")

        with col_fig:
            body_svg = (
                f'<svg viewBox="0 0 100 200" width="110" xmlns="http://www.w3.org/2000/svg">'
                f'<circle cx="50" cy="18" r="13" fill="#B0BEC5"/>'
                f'<rect x="30" y="32" width="40" height="55" rx="8" fill="{tr_c}" opacity="0.75"/>'
                f'<rect x="12" y="34" width="16" height="48" rx="7" fill="{la_c}" opacity="0.75"/>'
                f'<rect x="72" y="34" width="16" height="48" rx="7" fill="{ra_c}" opacity="0.75"/>'
                f'<rect x="27" y="90" width="18" height="70" rx="7" fill="{ll_c}" opacity="0.75"/>'
                f'<rect x="55" y="90" width="18" height="70" rx="7" fill="{rl_c}" opacity="0.75"/>'
                f'</svg>'
            )

            label_style = "font-size:0.78rem; color:#546E7A;"
            status_bold = "font-size:0.75rem; font-weight:700;"
            seg_bold    = "font-size:0.82rem; font-weight:600; color:#1E293B;"

            trunk_label = (
                f'<div style="text-align:center; margin-bottom:4px;">'
                f'<span style="{label_style}">{pcts[1]:.1f}%</span><br>'
                f'<span style="{status_bold} color:{tr_c};">{statuses[1]}</span><br>'
                f'<span style="{seg_bold}">Trunk</span>'
                f'</div>'
            )
            la_label = (
                f'<div style="text-align:center; width:28%;">'
                f'<span style="{label_style}">{pcts[0]:.1f}%</span><br>'
                f'<span style="{status_bold} color:{la_c};">{statuses[0]}</span><br>'
                f'<span style="{seg_bold}">Left Arm</span>'
                f'</div>'
            )
            ra_label = (
                f'<div style="text-align:center; width:28%;">'
                f'<span style="{label_style}">{pcts[2]:.1f}%</span><br>'
                f'<span style="{status_bold} color:{ra_c};">{statuses[2]}</span><br>'
                f'<span style="{seg_bold}">Right Arm</span>'
                f'</div>'
            )
            ll_label = (
                f'<div style="text-align:center; width:40%;">'
                f'<span style="{label_style}">{pcts[3]:.1f}%</span><br>'
                f'<span style="{status_bold} color:{ll_c};">{statuses[3]}</span><br>'
                f'<span style="{seg_bold}">Left Leg</span>'
                f'</div>'
            )
            rl_label = (
                f'<div style="text-align:center; width:40%;">'
                f'<span style="{label_style}">{pcts[4]:.1f}%</span><br>'
                f'<span style="{status_bold} color:{rl_c};">{statuses[4]}</span><br>'
                f'<span style="{seg_bold}">Right Leg</span>'
                f'</div>'
            )

            html = (
                f'<div style="position:relative; width:100%; padding-bottom:8px;">'
                f'{trunk_label}'
                f'<div style="display:flex; align-items:center; justify-content:space-between; margin:6px 0;">'
                f'{la_label}'
                f'<div style="width:44%; display:flex; justify-content:center;">{body_svg}</div>'
                f'{ra_label}'
                f'</div>'
                f'<div style="display:flex; justify-content:space-around; margin-top:4px;">'
                f'{ll_label}{rl_label}'
                f'</div>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)

        with col_cards:
            for seg, mass, change, pct, stat in zip(segments, masses, changes, pcts, statuses):
                c_col, c_arrow = change_meta(change)
                s_col = seg_color(pct)
                disp = change if change.startswith("-") else (f"+{change}" if change != "0.00" else "0.0")
                card = (
                    f'<div style="background:#fff; border:1px solid #E2E8F0; border-radius:12px;'
                    f' padding:10px 14px; margin-bottom:8px;'
                    f' display:flex; align-items:center; justify-content:space-between;">'
                    f'<div>'
                    f'<div style="font-size:0.72rem; color:#94A3B8; font-weight:600; letter-spacing:0.04em;">{seg.upper()}</div>'
                    f'<div style="font-size:1.45rem; font-weight:800; color:#0F172A; line-height:1.2;">'
                    f'{mass} <span style="font-size:0.85rem; font-weight:500; color:#64748B;">kg</span></div>'
                    f'<div style="font-size:0.72rem; font-weight:700; color:{s_col}; margin-top:1px;">{stat} · {pct:.1f}%</div>'
                    f'</div>'
                    f'<div style="background:#F8FAFC; border-radius:8px; padding:4px 10px; text-align:center; min-width:48px;">'
                    f'<div style="font-size:1rem; color:{c_col};">{c_arrow}</div>'
                    f'<div style="font-size:0.72rem; font-weight:700; color:{c_col};">{disp}</div>'
                    f'</div>'
                    f'</div>'
                )
                st.markdown(card, unsafe_allow_html=True)
            st.caption("Change from previous measurement")
        st.markdown('</div>', unsafe_allow_html=True)

        # Long-Term Musculoskeletal Health
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

        # Session Summary & Progress
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📊 Session Summary & Progress")
        col_p1, col_p2 = st.columns(2)
        pain_score_progress = pd.DataFrame({'Time': ['0 min', '5 min', '10 min', '15 min', '20 min'], 'Pain Score': [5, 4, 3, 2, 1]})
        fatigue_progress = pd.DataFrame({'Time': ['0 min', '5 min', '10 min', '15 min', '20 min'], 'Fatigue Level': [6, 5, 4, 3, 2]})
        with col_p1:
            st.markdown("**Pain Score Trend**")
            st.line_chart(pain_score_progress.set_index('Time'), color="#E57373", height=250)
        with col_p2:
            st.markdown("**Fatigue Level Trend**")
            st.line_chart(fatigue_progress.set_index('Time'), color="#64B5F6", height=250)
        st.markdown("**Muscle Activation (EMG) - Session Overview**")
        emg_progress = pd.DataFrame({'Time': ['0 min', '5 min', '10 min', '15 min', '20 min'], 'EMG Amplitude': [15, 18, 20, 22, 24]})
        st.bar_chart(emg_progress.set_index('Time'), color="#2A9D8F", height=250)
        st.markdown('</div>', unsafe_allow_html=True)

        # Session Audit Trail
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("📋 Session Audit Trail")
        df_logs = read_logs(patient_id)
        st.dataframe(df_logs, use_container_width=True, height=300)
        csv = df_logs.to_csv(index=False).encode("utf-8")
        st.download_button("Download Audit Trail in CSV", csv, f"audit_{patient_id}.csv", "text/csv")
        st.markdown('</div>', unsafe_allow_html=True)

        # Export Report
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
