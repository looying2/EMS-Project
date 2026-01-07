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

# ==========================================
# 1. PAGE CONFIGURATION & AESTHETICS
# ==========================================
st.set_page_config(
    page_title="NeuroFlex AI-EMS Clinical Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🩺"
)

# Custom CSS for "Medical Modern" Aesthetic
st.markdown("""
<style>
    /* --- MAIN LAYOUT & BACKGROUND --- */
    .stApp {
        background-color: #F0F2F6; /* Soft Clinical Grey-Blue */
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* --- HEADINGS --- */
    h1, h2, h3 {
        color: #264653; /* Dark Slate Blue */
        font-weight: 600;
    }

    /* --- CUSTOM DASHBOARD CARDS --- */
    .dashboard-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        margin-bottom: 20px;
        border: 1px solid #E0E0E0;
    }

    /* --- METRIC BOXES --- */
    div[data-testid="stMetric"] {
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
        color: #2A9D8F; /* Medical Teal */
        font-weight: 700;
    }

    /* --- BUTTON STYLING --- */
    /* Primary (Start) Buttons */
    div.stButton > button {
        border-radius: 25px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Emergency Button Specific Styling */
    div.stButton > button[kind="primary"] {
         background-color: #EF5350 !important;
         color: white !important;
         border: none;
    }

    /* --- SIDEBAR --- */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }

    /* --- ALERTS --- */
    .alert-box {
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        border-left: 5px solid;
    }
    .alert-safe { background-color: #E8F5E9; border-color: #4CAF50; color: #1B5E20; }
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
# 3. SESSION STATE
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
ss_init("telemetry", pd.DataFrame(columns=["t", "emg", "hr", "imp"]))
ss_init("ml_window", None)
ss_init("ml_prediction", "WAITING")
ss_init("ml_probability", 0.0)

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
def generate_ml_window(status):
    window_size = 200
    noise_level = 0.5
    data = {}
    muscles = ["Recto Femoral", "Biceps Femoral", "Vasto Medial", "EMG Semitendinoso"]
    
    for muscle in muscles:
        t = np.linspace(0, 10, window_size)
        base = np.sin(t) * 5 + np.random.normal(0, noise_level, window_size)
        if status == "Risk (Abnormal)":
            base = base * np.random.uniform(1.5, 3.0) + np.random.normal(0, 2, window_size)
        data[muscle] = base
    return pd.DataFrame(data)

def extract_features(raw_window_df):
    feats = {}
    feats['rms_recto_femoral'] = np.sqrt(np.mean(raw_window_df["Recto Femoral"]**2))
    feats['rms_biceps_femoral'] = np.sqrt(np.mean(raw_window_df["Biceps Femoral"]**2))
    feats['rms_vasto_medial'] = np.sqrt(np.mean(raw_window_df["Vasto Medial"]**2))
    feats['rms_emg_semitendinoso'] = np.sqrt(np.mean(raw_window_df["EMG Semitendinoso"]**2))
    return pd.DataFrame([feats])

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
    st.session_state.telemetry = df.tail(60)

def generate_report(pid, mass_df, pain_df, fatigue_df):
    """Generates a text report for download."""
    report_buffer = io.StringIO()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report_buffer.write(f"NEUROFLEX CLINICAL PROGRESS REPORT\n")
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
    # Taking last 5 entries for brevity in text report
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

# ==========================================
# 5. DIALOGS (Confirmations)
# ==========================================
# 5a. Start Session Dialog
@st.dialog("Start Session Confirmation")
def show_start_confirmation(pid, proto):
    st.write("### Safety Check")
    st.info("Please confirm that electrode placement and skin conditions have been verified manually.")
    st.warning("Ensure patient is ready for stimulation.")
    
    col_d1, col_d2 = st.columns(2)
    if col_d1.button("Yes (Start)", type="primary"):
        st.session_state.system_status = "ACTIVE"
        st.session_state.session_start_time = time.time()
        log_event(pid, "SESSION_START", f"Protocol={proto}")
        st.rerun() # Auto-close and refresh

    if col_d2.button("No (Cancel)"):
        st.rerun() # Auto-close and refresh

# 5b. Intensity Adjustment Dialog
@st.dialog("Confirm Intensity Adjustment")
def show_intensity_confirmation(pid, new_val):
    st.write(f"### Adjust Intensity?")
    st.write(f"You are changing the intensity to **{new_val} mA**.")
    st.warning("Please verify this level is safe for the patient.")

    col_i1, col_i2 = st.columns(2)
    
    if col_i1.button("Accept", type="primary"):
        st.session_state.intensity = new_val
        log_event(pid, "PARAM_CHANGE", f"Intensity set to {new_val}")
        st.success("Updated")
        st.rerun() # Auto-close and refresh

    if col_i2.button("Deny"):
        st.rerun() # Auto-close and refresh (keeps old value)


# ==========================================
# 6. SIDEBAR
# ==========================================
with st.sidebar:
    st.title("NeuroFlex System")
    st.caption(f"ML Engine: {model_status}")
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
    
    # Styled buttons using columns for layout
    c1, c2 = st.columns(2)
    
    # Logic: Show different Start Button based on Status
    if st.session_state.system_status == "ACTIVE":
        # Visual Change: Green, Disabled button indicating running state
        st.button("Session Running...", disabled=True, use_container_width=True)
    else:
        # Standard Start Button that triggers Dialog
        if st.button("▶ START", use_container_width=True):
            show_start_confirmation(patient_id, protocol)

    # Pause Button
    if st.button("⏸ PAUSE", disabled=st.session_state.system_status != "ACTIVE", use_container_width=True):
        st.session_state.system_status = "PAUSED"
        log_event(patient_id, "SESSION_PAUSE")
        st.rerun()

    # Stop Button
    if st.button("⏹ STOP SESSION", disabled=st.session_state.system_status not in ["ACTIVE", "PAUSED"], use_container_width=True):
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        log_event(patient_id, "SESSION_STOP")
        st.rerun()

# ==========================================
# 7. HEADER & STATUS
# ==========================================
# Wrap header in a card for better visuals
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
    timer = "--:--"
    if st.session_state.session_start_time and st.session_state.system_status == "ACTIVE":
        elapsed = int(time.time() - st.session_state.session_start_time)
        timer = f"{elapsed//60:02d}:{elapsed%60:02d}"
    st.metric("Session Time", timer)

with header_cols[3]:
    if st.button("Emergency STOP", type="primary", use_container_width=True):
        st.session_state.system_status = "STOPPED"
        st.session_state.intensity = 0
        log_event(patient_id, "EMERGENCY_STOP")
        st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 8. MAIN LOGIC LOOP
# ==========================================
if st.session_state.connected:
    update_telemetry_stream()
    
    if st.session_state.system_status == "ACTIVE":
        raw_ml_df = generate_ml_window(sim_mode)
        st.session_state.ml_window = raw_ml_df
        
        if rf_model:
            feats = extract_features(raw_ml_df)
            feats = feats[feature_names]
            pred = rf_model.predict(feats)[0]
            prob = rf_model.predict_proba(feats)[0][1]
            
            st.session_state.ml_prediction = "NORMAL" if pred == 1 else "ABNORMAL"
            st.session_state.ml_probability = prob

# ==========================================
# 9. TABS
# ==========================================
tab_live, tab_ai, tab_ctrl, tab_logs, tab_progress = st.tabs(
    ["Live Monitoring", "AI & RAG Analysis", "Device Control", "Audit Logs", "Progress Summary"]
)

# --- TAB 1: LIVE MONITORING ---
with tab_live:
    col_main_plot, col_mini_plot = st.columns([3, 1])
    
    with col_main_plot:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Global EMG Telemetry (RMS)")
        tele = st.session_state.telemetry
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tele["t"], y=tele["emg"], mode="lines", fill='tozeroy', name="EMG", line=dict(color='#2A9D8F', width=2)))
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=10, b=20), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', yaxis_title="Amplitude (uV)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_mini_plot:
        # Vitals stack
        last_hr = tele['hr'].iloc[-1] if len(tele) > 0 else 0
        last_imp = tele['imp'].iloc[-1] if len(tele) > 0 else 0
        
        st.metric("Heart Rate", f"{last_hr} BPM")
        st.metric("Impedance", f"{last_imp:.1f} kΩ")
        
        st.markdown('<div class="dashboard-card" style="margin-top:20px; padding:15px;">', unsafe_allow_html=True)
        st.caption("Raw Signal (4-CH)")
        if st.session_state.ml_window is not None:
            st.line_chart(st.session_state.ml_window, height=100)
        else:
            st.info("No active data")
        st.markdown('</div>', unsafe_allow_html=True)

    # Feedback Section
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Patient Feedback Input")
    c_feed1, c_feed2 = st.columns(2)
    pain_score = c_feed1.slider("Pain / Discomfort (0–10)", 0, 10, 2)
    fatigue_score = c_feed2.slider("Fatigue Level (0–10)", 0, 10, 4)
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 2: AI & RAG ANALYSIS ---
with tab_ai:
    col_rag, col_ml = st.columns(2)
    
    # --- A. Rule-Based RAG (Safety) ---
    with col_rag:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Safety & Optimization (Rules)")
        
        if st.session_state.system_status == "ACTIVE":
            if pain_score >= 6:
                st.markdown("""<div class="alert-box alert-risk">
                <strong>High Pain Detected</strong><br>
                Observation: Pain Score > 6<br>
                Action: Reducing intensity by 20% (Rule PAIN-01)
                </div>""", unsafe_allow_html=True)
            elif fatigue_score >= 7:
                st.markdown("""<div class="alert-box alert-info">
                <strong>High Fatigue</strong><br>
                Observation: Patient reported fatigue > 7<br>
                Action: Increasing OFF time (Rule ONOFF-04)
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="alert-box alert-safe">
                <strong>System Nominal</strong><br>
                All parameters within safety limits.<br>
                Action: Maintain current protocol (Rule MAIN-01)
                </div>""", unsafe_allow_html=True)
        else:
            st.caption("System Inactive - Start session to monitor safety rules.")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- B. ML Model (Gait Analysis) ---
    with col_ml:
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.subheader("Gait Pathology (ML Engine)")
        
        res = st.session_state.ml_prediction
        prob = st.session_state.ml_probability
        
        if st.session_state.system_status == "ACTIVE":
            if res == "ABNORMAL":
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
            
            with st.expander("View Raw Features"):
                if st.session_state.ml_window is not None:
                    feat_view = extract_features(st.session_state.ml_window)
                    st.dataframe(feat_view, hide_index=True)
        else:
            st.info("Start session to enable ML analysis.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 3: DEVICE CONTROL ---
with tab_ctrl:
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
                # Trigger the new Intensity Confirmation Dialog
                if st.button("Apply Changes", type="primary"):
                    show_intensity_confirmation(patient_id, new_int)
    else:
        st.warning("Intensity adjustments are locked for Caregiver role.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 4: LOGS ---
with tab_logs:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Session Audit Trail")
    df_logs = read_logs(patient_id)
    st.dataframe(df_logs, use_container_width=True, height=300)
    
    csv = df_logs.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV", csv, f"audit_{patient_id}.csv", "text/csv")
    st.markdown('</div>', unsafe_allow_html=True)

# --- TAB 5: PROGRESS SUMMARY ---
with tab_progress:
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
    st.subheader("Session Summary & Progress")
    
    col_p1, col_p2 = st.columns(2)
    
    # 1. Pain and Fatigue Data
    pain_score_progress = pd.DataFrame({
        'Time': ['0 min', '5 min', '10 min', '15 min', '20 min'],
        'Pain Score': [5, 4, 3, 2, 1]
    })
    
    fatigue_progress = pd.DataFrame({
        'Time': ['0 min', '5 min', '10 min', '15 min', '20 min'],
        'Fatigue Level': [6, 5, 4, 3, 2]
    })

    with col_p1:
        st.markdown("**Pain Score Trend**")
        st.line_chart(pain_score_progress.set_index('Time'), color="#E57373")
        
    with col_p2:
        st.markdown("**Fatigue Level Trend**")
        st.line_chart(fatigue_progress.set_index('Time'), color="#64B5F6")
        
    st.markdown("**Muscle Activation (EMG) - Session Overview**")
    emg_progress = pd.DataFrame({
        'Time': ['0 min', '5 min', '10 min', '15 min', '20 min'],
        'EMG Amplitude': [15, 18, 20, 22, 24]
    })
    st.bar_chart(emg_progress.set_index('Time'), color="#2A9D8F")
    
    st.divider()
    
    # 2. Muscle Mass Trend (Analysis over Past Sessions)
    st.subheader("Long-Term Musculoskeletal Health")
    st.markdown("**Muscle Mass Trend (Last 30 Days)**")
    
    # Simulate historical data for muscle mass (Hypertrophy trend)
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    base_mass = np.linspace(68, 69.8, 30) # Gradual increase
    noise = np.random.normal(0, 0.1, 30)
    mass_values = base_mass + noise
    
    muscle_mass_df = pd.DataFrame({
        'Date': dates,
        'Muscle Mass (kg)': mass_values
    })
    
    # Using an area chart for the trend
    st.area_chart(muscle_mass_df.set_index('Date'), color="#FF9800")
    st.caption("Showing estimated lean muscle mass trajectory based on bio-impedance analysis.")
    
    st.divider()
    
    # 3. Download Report Feature
    st.subheader("Export Report")
    
    # Generate the report string
    report_text = generate_report(patient_id, muscle_mass_df, pain_score_progress, fatigue_progress)
    
    st.download_button(
        label="📄 Download Progress Report (TXT)",
        data=report_text,
        file_name=f"Report_{patient_id}_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        type="primary"
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 10. AUTO REFRESH
# ==========================================
if st.session_state.system_status == "ACTIVE":
    time.sleep(1)
    st.rerun()

