import streamlit as st
import pandas as pd
import numpy as np
import time
import sqlite3
import joblib
import json
import os
import plotly.graph_objects as go
from datetime import datetime
from collections import deque

# ==========================================
# 1. PAGE CONFIGURATION & DATABASE
# ==========================================
st.set_page_config(
    page_title="NeuroFlex AI-EMS Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- SQLite Audit Log Setup ---
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

# ==========================================
# 2. ML MODEL LOADING
# ==========================================
@st.cache_resource
def load_model_assets():
    try:
        model = joblib.load("model.pkl")
        with open("feature_cols.json", "r") as f:
            features = json.load(f)
        return model, features, "Online"
    except FileNotFoundError:
        return None, None, "Offline (Files Missing)"
    except Exception as e:
        return None, None, f"Error: {e}"

rf_model, feature_names, model_status = load_model_assets()

# ==========================================
# 3. SESSION STATE INITIALIZATION
# ==========================================
def ss_init(key, value):
    if key not in st.session_state:
        st.session_state[key] = value

ss_init("system_status", "IDLE")
ss_init("connected", True)
ss_init("session_start_time", None)
ss_init("intensity", 15)
ss_init("frequency", 40)
ss_init("pulse_width", 300)
ss_init("duty_on", 10)
ss_init("duty_off", 20)
# Telemetry for the live line chart (Single channel overview)
ss_init("telemetry", pd.DataFrame(columns=["t", "emg", "hr", "imp"]))
# Buffer for the ML Model (4-channel window)
ss_init("ml_window", None)
ss_init("ml_prediction", "WAITING")
ss_init("ml_probability", 0.0)

# ==========================================
# 4. HELPER FUNCTIONS (Data Generation)
# ==========================================
def generate_ml_window(status):
    """Generates 200 samples of 4-channel data for the Random Forest."""
    window_size = 200
    noise_level = 0.5
    data = {}
    muscles = ["Recto Femoral", "Biceps Femoral", "Vasto Medial", "EMG Semitendinoso"]
    
    for muscle in muscles:
        t = np.linspace(0, 10, window_size)
        base = np.sin(t) * 5 + np.random.normal(0, noise_level, window_size)
        # Simulate Pathology
        if status == "Risk (Abnormal)":
            base = base * np.random.uniform(1.5, 3.0) + np.random.normal(0, 2, window_size)
        data[muscle] = base
    return pd.DataFrame(data)

def extract_features(raw_window_df):
    """Calculates RMS features exactly as the model expects."""
    feats = {}
    feats['rms_recto_femoral'] = np.sqrt(np.mean(raw_window_df["Recto Femoral"]**2))
    feats['rms_biceps_femoral'] = np.sqrt(np.mean(raw_window_df["Biceps Femoral"]**2))
    feats['rms_vasto_medial'] = np.sqrt(np.mean(raw_window_df["Vasto Medial"]**2))
    feats['rms_emg_semitendinoso'] = np.sqrt(np.mean(raw_window_df["EMG Semitendinoso"]**2))
    return pd.DataFrame([feats])

# --- B. For the Live Plot (Single Point Stream) ---
def update_telemetry_stream():
    df = st.session_state.telemetry.copy()
    now = datetime.now().strftime("%H:%M:%S")

    if st.session_state.system_status == "ACTIVE":
        emg = np.random.normal(st.session_state.intensity * 2, 4)
        hr = int(np.clip(np.random.normal(74, 3), 60, 110))
        imp = float(np.clip(np.random.normal(1.2, 0.1), 0.7, 2.5))
    else:
        emg = np.random.normal(0, 2)
        hr = int(np.random.normal(72, 2))
        imp = float(np.random.normal(1.2, 0.1))

    new_row = pd.DataFrame([{"t": now, "emg": emg, "hr": hr, "imp": imp}])
    df = pd.concat([df, new_row], ignore_index=True)
    st.session_state.telemetry = df.tail(60) # Keep last 60 seconds

# ==========================================
# 5. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("NeuroFlex System")
    st.caption(f"ML Model: {model_status}")

    st.subheader("Patient Profile")
    patient_id = st.text_input("Patient ID", value="PT-2024-89")
    age_group = st.selectbox("Age Group", ["60-69", "70-79", "80+"])
    condition_tags = st.multiselect("Conditions", ["Sarcopenia", "Post-Stroke", "Osteoarthritis", "Parkinson's Disease", "Muscle Atrophy"], default=["Sarcopenia"])
    mobility = st.selectbox("Mobility", ["Independent", "Assisted", "Wheelchair"])
    
    # Display height and weight for normalization, no user input for these
    height_display = 170  # Static for normalization purposes
    weight_display = 70   # Static for normalization purposes
    st.markdown(f"**Height**: {height_display} cm")
    st.markdown(f"**Weight**: {weight_display} kg")
    
    # --- User Role Selection (Doctor vs Caregiver) ---
    user_role = st.selectbox("Select User Role", ["Doctor", "Caregiver"])
    
    st.divider()

    st.subheader("Simulation Control")
    # This controls the "Hidden" data state for the ML model
    sim_mode = st.radio("Simulate Patient State:", ["Normal", "Risk (Abnormal)"])
    st.session_state.connected = st.toggle("Device Connected", value=True)

    st.divider()

    st.subheader("Session Setup")
    protocol = st.selectbox("Therapy Protocol", ["Muscle Stimulation"])
    
    col_chk1, col_chk2 = st.columns(2)
    electrode_check = col_chk1.checkbox("Electrodes OK")
    skin_check = col_chk2.checkbox("Skin OK")
    
    # Double Confirmation for Electrode and Skin Condition
    can_start = False
    if electrode_check and skin_check:
        if st.button("Confirm Electrode and Skin Condition"):
            can_start = True
            st.success("Electrode and Skin Condition Confirmed")
        else:
            st.warning("Please confirm the electrode placement and skin condition.")
    
    if st.button("▶ Start Session", disabled=not can_start, type="primary"):
        st.session_state.system_status = "ACTIVE"
        st.session_state.session_start_time = time.time()
        log_event(patient_id, "SESSION_START", f"Protocol={protocol}, Mode={sim_mode}")

    if st.button("⏸ Pause Session", disabled=st.session_state.system_status != "ACTIVE"):
        st.session_state.system_status = "PAUSED"
        log_event(patient_id, "SESSION_PAUSE")

    if st.button("⏹ Stop Session", disabled=st.session_state.system_status not in ["ACTIVE", "PAUSED"]):
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        log_event(patient_id, "SESSION_STOP")

# ==========================================
# 6. HEADER & EMERGENCY
# ==========================================
header_cols = st.columns([3, 1, 1, 1])
with header_cols[0]:
    st.title("Clinical Dashboard")
    st.caption(f"Patient: {patient_id} | Protocol: {protocol}")

with header_cols[1]:
    color = "green" if st.session_state.system_status == "ACTIVE" else "gray"
    st.markdown(f"**Status:** :{color}[{st.session_state.system_status}]")

with header_cols[2]:
    timer = "--:--"
    if st.session_state.session_start_time and st.session_state.system_status == "ACTIVE":
        elapsed = int(time.time() - st.session_state.session_start_time)
        timer = f"{elapsed//60:02d}:{elapsed%60:02d}"
    st.metric("Time", timer)

with header_cols[3]:
    if st.button("EMERGENCY STOP", type="primary"):
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        log_event(patient_id, "EMERGENCY_STOP")

st.divider()

# ==========================================
# 7. MAIN LOGIC LOOP (Data Processing)
# ==========================================
if st.session_state.connected:
    # 1. Update the visual line chart
    update_telemetry_stream()
    
    # 2. Update the ML Model Data (only if Active)
    if st.session_state.system_status == "ACTIVE":
        # Generate 4-channel window
        raw_ml_df = generate_ml_window(sim_mode)
        st.session_state.ml_window = raw_ml_df
        
        # Run Prediction
        if rf_model:
            feats = extract_features(raw_ml_df)
            # Ensure column order matches training
            feats = feats[feature_names]
            
            pred = rf_model.predict(feats)[0]
            prob = rf_model.predict_proba(feats)[0][1]
            
            st.session_state.ml_prediction = "ABNORMAL" if pred == 1 else "NORMAL"
            st.session_state.ml_probability = prob

# ==========================================
# 8. TABS
# ==========================================
tab_live, tab_ai, tab_ctrl, tab_logs = st.tabs(
    ["📊 Live Monitoring", "🧠 AI & RAG Analysis", "🎛 Device Control", "📑 Audit Logs"]
)

# --- TAB 1: LIVE MONITORING ---
with tab_live:
    col_main_plot, col_mini_plot = st.columns([3, 1])
    
    with col_main_plot:
        st.subheader("Global EMG Telemetry")
        tele = st.session_state.telemetry
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tele["t"], y=tele["emg"], mode="lines", name="EMG (RMS)", line=dict(color='#008080')))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="Amplitude (uV)")
        st.plotly_chart(fig, use_container_width=True)

    with col_mini_plot:
        st.subheader("Vitals")
        last_hr = tele['hr'].iloc[-1] if len(tele) > 0 else 0
        last_imp = tele['imp'].iloc[-1] if len(tele) > 0 else 0
        st.metric("Heart Rate", f"{last_hr} BPM")
        st.metric("Impedance", f"{last_imp:.1f} kΩ")
        
        st.markdown("---")
        st.caption("Detailed 4-Channel View")
        if st.session_state.ml_window is not None:
            st.line_chart(st.session_state.ml_window, height=120)
        else:
            st.info("No active data")

    st.subheader("Patient Feedback Input")
    c_feed1, c_feed2 = st.columns(2)
    pain_score = c_feed1.slider("Pain / Discomfort (0–10)", 0, 10, 2)
    fatigue_score = c_feed2.slider("Fatigue Level (0–10)", 0, 10, 4)

# --- TAB 2: AI & RAG ANALYSIS ---
with tab_ai:
    col_rag, col_ml = st.columns(2)
    
    # --- A. Rule-Based RAG (Safety) ---
    with col_rag:
        st.subheader("🛡️ Safety & Optimization (Rules)")
        
        # RAG Logic
        if st.session_state.system_status == "ACTIVE":
            if pain_score >= 6:
                obs = f"High pain reported ({pain_score}/10)."
                act = "Reduce stimulation intensity by 20%."
                ref = "Safety Rule: PAIN-01"
                rag_color = "orange"
            elif fatigue_score >= 7:
                obs = "Patient reported high fatigue."
                act = "Increase OFF time in duty cycle."
                ref = "Fatigue Rule: ONOFF-04"
                rag_color = "blue"
            else:
                obs = "Patient comfort within range."
                act = "Maintain current parameters."
                ref = "Optimization Rule: MAINTAIN-01"
                rag_color = "green"
        else:
            obs = "Session inactive."
            act = "No action."
            ref = "N/A"
            rag_color = "gray"

        st.info(f"**Observation:** {obs}")
        st.markdown(f"**Recommended Action:** :{rag_color}[{act}]")
        with st.expander("View Protocol Reference"):
            st.write(ref)

    # --- B. ML Model (Gait Analysis) ---
    with col_ml:
        st.subheader("🤖 Gait Pathology (ML Model)")
        
        res = st.session_state.ml_prediction
        prob = st.session_state.ml_probability
        
        if st.session_state.system_status == "ACTIVE":
            if res == "ABNORMAL":
                box_color = "#FFCDD2"  # Light red (deeper red)
                text_color = "black"   # Changed to black for "Abnormal"
                msg = "Gait Pathology Detected"
                rec = "Evaluate electrode placement or reduce frequency."
            else:
                box_color = "#E8F5E9"  # Light green
                text_color = "green"
                msg = "Normal Gait Pattern"
                rec = "Continue current protocol."

            st.markdown(f"""
            <div style="background-color: {box_color}; padding: 15px; border-radius: 10px; border-left: 5px solid {text_color};">
                <h3 style="color: {text_color}; margin:0;">{res}</h3>
                <p>Confidence: {prob:.1%}</p>
                <hr>
                <p><strong>Analysis:</strong> {msg}</p>
                <p><strong>AI Suggestion:</strong> {rec}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show live feature data
            if st.checkbox("Show Raw Features"):
                if st.session_state.ml_window is not None:
                    feat_view = extract_features(st.session_state.ml_window)
                    st.dataframe(feat_view, hide_index=True)

        else:
            st.warning("Start session to enable ML analysis.")

# --- TAB 3: DEVICE CONTROL ---
with tab_ctrl:
    st.subheader("Stimulation Parameters")
    c1, c2, c3 = st.columns(3)
    c1.metric("Intensity (mA)", st.session_state.intensity)
    c2.metric("Frequency (Hz)", st.session_state.frequency)
    c3.metric("Pulse Width (us)", st.session_state.pulse_width)
    
    st.progress(0.5, text=f"Duty Cycle: {st.session_state.duty_on}s ON / {st.session_state.duty_off}s OFF")
    
    # Restrict intensity adjustment based on user role
    if user_role == "Doctor":
        new_int = st.number_input("Adjust Intensity", 0, 100, st.session_state.intensity)
        if new_int != st.session_state.intensity:
            if st.button("Confirm Adjustment"):
                st.session_state.intensity = new_int
                log_event(patient_id, "PARAM_CHANGE", f"Intensity set to {new_int}")
                st.success(f"Intensity adjusted to {new_int} mA")
                st.rerun()
    else:
        st.warning("Only doctors can adjust the intensity.")

# --- TAB 4: LOGS ---
with tab_logs:
    st.subheader("Session Audit Trail")
    df_logs = read_logs(patient_id)
    st.dataframe(df_logs, use_container_width=True)
    
    csv = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"audit_{patient_id}.csv", "text/csv")

# ==========================================
# 9. AUTO REFRESH
# ==========================================
if st.session_state.system_status == "ACTIVE":
    time.sleep(1)
    st.rerun()
