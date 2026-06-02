import requests
import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
import joblib
import json
import os
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
            timeout=20,
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
ss_init("ml_latest", {})
ss_init("ml_probabilities", [])
ss_init("ml_session", {})

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
        st.session_state.session_summary_text = "No telemetry data collected during this session."
        return

    # Compute session stats
    avg_emg = tele['emg'].mean()
    max_emg = tele['emg'].max()
    min_emg = tele['emg'].min()
    duration = st.session_state.elapsed_time  # in seconds
    pain = st.session_state.live_pain
    fatigue = st.session_state.live_fatigue
    gait = st.session_state.ml_prediction  # "NORMAL" or "ABNORMAL"
    
    # Build prompt for RAG API
    prompt = f"""
    You are a clinical assistant. Write a short, professional summary of the following EMS therapy session:

    - Duration: {duration:.0f} seconds (approx {duration/60:.1f} minutes)
    - Average EMG: {avg_emg:.1f} µV
    - Maximum EMG: {max_emg:.1f} µV
    - Minimum EMG: {min_emg:.1f} µV
    - Final patient pain score: {pain}/10
    - Final patient fatigue score: {fatigue}/10
    - Gait pathology classification: {gait}
    
    Provide a conclusion and any recommendations.
    """
    
    answer, _ = call_rag_api(prompt)
    if not answer or "Error" in answer:
        answer = "Could not generate AI summary. The RAG service may be offline."
    
    st.session_state.session_summary_text = answer

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
        response = requests.get(url)
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

    current_time = time.time()

    # Call remote ML backend every 3 seconds only
    if st.session_state.system_status == "ACTIVE":

    current_time = time.time()

    # Call ML backend every 3 seconds only
    if current_time - st.session_state.last_ml_call_time >= 3:

        prediction, confidence, summary, latest, probabilities, session = call_ml_api()

        st.session_state.ml_prediction = prediction
        st.session_state.ml_probability = confidence
        st.session_state.ml_summary = summary
        st.session_state.ml_latest = latest
        st.session_state.ml_probabilities = probabilities
        st.session_state.ml_session = session
        st.session_state.last_ml_call_time = current_time
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
# 5. DIALOGS
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
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        st.session_state.elapsed_time = 0.0 
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
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        st.session_state.elapsed_time = 0.0 
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

    # 2. If session is active, run ML prediction on recent EMG values
    if st.session_state.system_status == "ACTIVE":
        emg_values = st.session_state.telemetry['emg'].tail(50).values
        if len(emg_values) > 0:
            rms = np.sqrt(np.mean(emg_values**2))
            spread = np.max(emg_values) - np.min(emg_values)
            std = np.std(emg_values)
            prediction, confidence, summary = call_ml_api(rms, spread, std)
            st.session_state.ml_prediction = prediction
            st.session_state.ml_probability = confidence 
            st.session_state.ml_summary = summary
        else:
            st.session_state.ml_prediction = "WAITING"
            st.session_state.ml_probability = 0.0
            st.session_state.ml_summary = {}

    # 3. Detect session end and generate AI summary (once)
    if st.session_state.system_status == "STOPPED" and not st.session_state.session_summary_generated:
        generate_session_summary()
        st.session_state.session_summary_generated = True

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
        col_rag, col_ml = st.columns(2)
        with col_rag:
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
        with col_ml:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Gait Pathology (ML Engine)")

    res = st.session_state.ml_prediction
    prob = st.session_state.ml_probability

    if st.session_state.system_status == "ACTIVE":

        # ===============================
        # MAIN ML RESULT CARD
        # ===============================
        if res == "ABNORMAL" or res == "Abnormal":
            st.markdown(f"""
            <div class="alert-box alert-risk">
                <h3 style="color:#B71C1C; margin:0;">PATHOLOGY DETECTED</h3>
                <p>Confidence: {prob:.1%}</p>
                <hr>
                <p><strong>Recommendation:</strong> Evaluate electrode placement or reduce frequency.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="alert-box alert-safe">
                <h3 style="color:#1B5E20; margin:0;">NORMAL GAIT</h3>
                <p>Confidence: {prob:.1%}</p>
                <hr>
                <p><strong>Recommendation:</strong> Continue current protocol.</p>
            </div>
            """, unsafe_allow_html=True)

        # ===============================
        # ADD THIS PART HERE
        # ===============================
        latest = st.session_state.get("ml_latest", {})
        session = st.session_state.get("ml_session", {})
        probabilities = st.session_state.get("ml_probabilities", [])

        if latest:
            st.markdown("#### Latest ML Features")

            f1, f2, f3 = st.columns(3)

            with f1:
                st.metric(
                    "Recto Femoris RMS",
                    f"{latest.get('rms_recto_femoral', 0):.2f}"
                )

            with f2:
                st.metric(
                    "Signal Spread",
                    f"{latest.get('rms_signal_spread', 0):.2f}"
                )

            with f3:
                st.metric(
                    "Signal STD",
                    f"{latest.get('rms_signal_std', 0):.2f}"
                )

        if probabilities:
            st.markdown("#### Prediction Probability")

            for p in probabilities:
                label = p.get("label", "Unknown")
                probability = float(p.get("probability", 0))

                st.progress(
                    probability / 100,
                    text=f"{label}: {probability:.2f}%"
                )

        if session:
            st.markdown("#### Session Summary")

            st.write(
                f"Readings: {session.get('count', '-')}"
                f" | Average: {session.get('avg', '-')}"
                f" | Min: {session.get('min', '-')}"
                f" | Max: {session.get('max', '-')}"
            )

        # ===============================
        # EXISTING AI SUMMARY
        # ===============================
        if st.session_state.ml_summary:
            with st.expander("📋 AI Summary & Recommendations"):
                summary = st.session_state.ml_summary
                st.markdown(f"**{summary.get('title', '')}**")
                st.markdown(summary.get('summary', ''))

                if summary.get('interpretation'):
                    st.markdown("**Interpretation:**")
                    for item in summary['interpretation']:
                        st.markdown(f"- {item}")

                if summary.get('actions'):
                    st.markdown("**Recommended Actions:**")
                    for item in summary['actions']:
                        st.markdown(f"- {item}")

                st.caption(summary.get('disclaimer', ''))

    else:
        st.info("Start session to enable ML analysis.")

    st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 💬 Patient Feedback")
        col_feedback = st.columns(1)[0]
        with col_feedback:
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
        if st.session_state.session_summary_text:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.subheader("📝 AI Session Summary")
            st.markdown(st.session_state.session_summary_text)
            if st.button("Regenerate Summary", key="regenerate_summary"):
                st.session_state.session_summary_generated = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        # Segmental Lean Mass Analysis 
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("💪 Segmental Lean Mass Analysis ")
        st.caption("Percentage of ideal muscle mass per body segment (≥90% = Normal, <90% = Under)")
        segments = ["Left Arm", "Trunk", "Right Arm", "Left Leg", "Right Leg"]
        percentages = [65.3, 84.8, 69.8, 93.4, 93.9]
        statuses = ["Under", "Under", "Under", "Normal", "Normal"]
        masses = [1.13, 13.3, 1.21, 5.11, 5.13]
        changes = ["0.00", "-0.1", "0.00", "+0.04", "+0.09"]
        icons = ["💪", "🎯", "💪", "🦵", "🦵"]
        col_left, col_right = st.columns([1.2, 0.9], gap="medium")
        with col_left:
            st.markdown("#### Performance vs Ideal")
            for seg, pct, stat, icon in zip(segments, percentages, statuses, icons):
                bar_color = "#EF5350" if pct < 90 else "#2A9D8F"
                status_color = "#EF5350" if pct < 90 else "#2A9D8F"
                st.markdown(f"""
                <div style="margin-bottom: 22px;">
                    <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 5px;">
                        <span style="font-size: 1.1rem;">{icon}</span>
                        <span style="font-weight: 600; font-size: 0.9rem;">{seg}</span>
                        <span style="margin-left: auto; font-size: 0.75rem; background-color: #F1F5F9; padding: 2px 8px; border-radius: 20px; color: {status_color}; font-weight: 500;">{stat}</span>
                    </div>
                    <div style="background-color: #E2E8F0; border-radius: 12px; height: 28px; width: 100%; position: relative;">
                        <div style="background: linear-gradient(90deg, {bar_color}, {bar_color}CC); width: {pct}%; height: 28px; border-radius: 12px; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px;">
                            <span style="color: white; font-size: 0.75rem; font-weight: 600;">{pct:.1f}%</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        with col_right:
            st.markdown("#### Soft Lean Mass (kg)")
            for seg, mass, change, icon in zip(segments, masses, changes, icons):
                if change.startswith("-"):
                    change_color = "#DC2626"
                    arrow = "↓"
                elif change == "0.00":
                    change_color = "#64748B"
                    arrow = "→"
                else:
                    change_color = "#16A34A"
                    arrow = "↑"
                sign = "" if change.startswith("-") else "+" if change != "0.00" else ""
                display_change = f"{sign}{change}" if change != "0.00" else "0.00"
                st.markdown(f"""
                <div style="background-color: #F8FAFC; border-radius: 12px; padding: 10px 12px; margin-bottom: 12px; border-left: 4px solid {change_color};">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span style="font-size: 1.1rem;">{icon}</span>
                            <span style="font-weight: 500;">{seg}</span>
                        </div>
                        <div style="text-align: right;">
                            <span style="font-size: 1.3rem; font-weight: 700;">{mass}</span>
                            <span style="font-size: 0.8rem;"> kg</span>
                            <div style="font-size: 0.7rem; color: {change_color};">
                                {arrow} {display_change} vs previous
                            </div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.caption("📈 Change from previous measurement (demo data)")
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
    time.sleep(0.2)
    st.rerun()

