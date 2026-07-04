import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import json
import os

st.set_page_config(page_title="Refrigerator Simulator Hub", layout="wide")

MEMORY_FILE = "simulator_memory.json"
REVIEWER_PASSWORD = "Admin@Cooling2026"

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

if 'db' not in st.session_state:
    st.session_state.db = load_memory()

if 'reviewer_logged_in' not in st.session_state:
    st.session_state.reviewer_logged_in = False

tc_features = ['tf-1', 'tf-2', 'tf-3', 'tf-4', 'tf-5', 'tc-1', 'tc-2', 'tc-3', 'tvc', 'S2']
metric_types = ['mean', 'min', 'max', '(max+min)/2']

# --- ADVANCED INTERPOLATION ENGINE ---
def run_automated_simulation(volume_records, new_pulldown, target_sensors):
    results_map = {}
    
    # Run separate predictions for each sensor value input by the user
    for idx, target_s in enumerate(target_sensors):
        predicted_modes = []
        
        # We model behavior across available historical test flags (Boost Cooling, Level-5, etc.)
        all_flags = set()
        for record in volume_records:
            for flag in record['cpt_data'].keys():
                all_flags.add(flag)
                
        for flag in sorted(all_flags):
            for metric in metric_types:
                X_train, y_train = [], []
                
                for record in volume_records:
                    if flag not in record['cpt_data'] or metric not in record['cpt_data'][flag]:
                        continue
                    
                    # Feature vector mapping base criteria
                    p_base = [record['pulldown_data'][f] for f in tc_features] + [record['pulldown_baseline_sensor']]
                    outputs = record['cpt_data'][flag][metric]
                    y_vector = [outputs.get(f, 0.0) for f in tc_features] + [outputs.get('S2', 0.0), outputs.get('Sensor', 0.0)]
                    
                    X_train.append(p_base)
                    y_train.append(y_vector)
                
                if len(X_train) >= 1:
                    knn = KNeighborsRegressor(n_neighbors=min(2, len(X_train)), weights='distance')
                    knn.fit(X_train, y_train)
                    
                    # Complete user evaluation array
                    query_vector = np.array(new_pulldown + [target_s]).reshape(1, -1)
                    prediction = knn.predict(query_vector)[0]
                    
                    row = {
                        "TestFlag": flag,
                        "Data Criteria": metric,
                        "tf-1": round(prediction[0], 2), "tf-2": round(prediction[1], 2),
                        "tf-3": round(prediction[2], 2), "tf-4": round(prediction[3], 2),
                        "tf-5": round(prediction[4], 2), "tc-1": round(prediction[5], 2),
                        "tc-2": round(prediction[6], 2), "tc-3": round(prediction[7], 2),
                        "tvc": round(prediction[8], 2),  "S2": round(prediction[9], 2),
                        "Sensor": round(prediction[10], 2)
                    }
                    predicted_modes.append(row)
                    
        if predicted_modes:
            results_map[f"Target Sensor Set {idx+1} ({target_s}°C)"] = pd.DataFrame(predicted_modes)
            
    return results_map

# --- SIDEBAR WORKSPACE MANAGER ---
st.sidebar.header("📁 Volume Profile Manager")
existing_volumes = list(st.session_state.db.keys()) or ["365L"]
selected_volume = st.sidebar.selectbox("Active Refrigerator Volume Model:", existing_volumes)

new_vol = st.sidebar.text_input("➕ Register New Volume Model:")
if st.sidebar.button("Add Volume Segment"):
    if new_vol and new_vol not in st.session_state.db:
        st.session_state.db[new_vol] = []
        save_memory(st.session_state.db)
        st.success(f"Model {new_vol} registered.")
        st.rerun()

st.sidebar.markdown("---")

tab1, tab2, tab3 = st.tabs(["🎛️ Run Automated Simulator", "🛠️ Data Repository Room", "🔍 Reviewer Dashboard"])

# ================= TAB 1: RUN AUTOMATED SIMULATOR =================
with tab1:
    st.subheader(f"Predict Multilevel CPT Matrices ({selected_volume})")
    vol_records = st.session_state.db.get(selected_volume, [])
    
    if not vol_records:
        st.warning(f"⚠️ No training data has been saved for **[{selected_volume}]** yet. Please visit the Data Repository Room to parse initial reports.")
    else:
        st.markdown("#### Step 1: Input Current Pulldown Telemetry Vector")
        
        sim_pulldown_file = st.file_uploader(
            "Optional: Upload a Pulldown Excel Report to auto-fill telemetry cells (.xlsx)", 
            type=["xlsx"], 
            key="sim_tab_pulldown_uploader"
        )
        
        if 'active_pulldown_form' not in st.session_state:
            st.session_state.active_pulldown_form = {feat: 0.0 for feat in tc_features}
            st.session_state.last_uploaded_sim_file = None

        if sim_pulldown_file and sim_pulldown_file != st.session_state.last_uploaded_sim_file:
            try:
                df_sim_p = pd.read_excel(sim_pulldown_file, sheet_name='Summary', header=1)
                df_sim_p.columns = [str(c).strip() for c in df_sim_p.columns]
                df_sim_p.iloc[:, 0] = df_sim_p.iloc[:, 0].astype(str).str.strip()
                df_sim_p.set_index(df_sim_p.columns[0], inplace=True)
                
                # Included S2 mapping parameter row connection
                mapping_keys = {
                    'tf-1':'tf1', 'tf-2':'tf2', 'tf-3':'tf3', 'tf-4':'tf4', 'tf-5':'tf5', 
                    'tc-1':'tc1', 'tc-2':'tc2', 'tc-3':'tc3', 'tvc':'tvc', 'S2':'S2'
                }
                
                for feat, xl_label in mapping_keys.items():
                    if xl_label in df_sim_p.index:
                        st.session_state.active_pulldown_form[feat] = float(df_sim_p.loc[xl_label, 'avg'])
                
                st.session_state.last_uploaded_sim_file = sim_pulldown_file
                st.toast("🟢 Telemetry values successfully pulled from spreadsheet!", icon="📊")
            except Exception as e:
                st.error(f"Error parsing local test profile: {str(e)}")

        if not sim_pulldown_file and st.session_state.last_uploaded_sim_file is not None:
            st.session_state.active_pulldown_form = {feat: 0.0 for feat in tc_features}
            st.session_state.last_uploaded_sim_file = None

        # Split horizontally across 10 dynamic grid columns now
        u_cols = st.columns(10)
        new_pulldown_input = []
        
        default_defaults = {
            'tf-1': -24.4, 'tf-2': -21.8, 'tf-3': -22.8, 
            'tf-4': -26.2, 'tf-5': -26.4, 'tc-1': 1.9, 
            'tc-2': 1.6,   'tc-3': 0.5,   'tvc': 8.1, 'S2': 41.1
        }
        
        for i, feat in enumerate(tc_features):
            current_stored_val = st.session_state.active_pulldown_form.get(feat, 0.0)
            if current_stored_val == 0.0:
                current_stored_val = default_defaults.get(feat, 0.0)
                
            val = u_cols[i].number_input(
                f"{feat}:", 
                value=current_stored_val, 
                key=f"automated_sim_input_{feat}"
            )
            new_pulldown_input.append(val)
            
        st.markdown("---")
        st.markdown("#### Step 2: Set Multi-Sensor Simulation Steps")
        num_targets = st.number_input("Number of target sensor variables to predict (1 to 7):", min_value=1, max_value=7, value=3, step=1)
        
        target_sensors = []
        s_cols = st.columns(int(num_targets))
        for idx in range(int(num_targets)):
            s_val = s_cols[idx].number_input(f"Sensor Query Point {idx+1} (°C):", value=-10.0 + (idx * 2.5), key=f"query_s_{idx}")
            target_sensors.append(s_val)
            
        if st.button("🚀 Generate Predictive CPT Dataset Matrices"):
            with st.spinner("Interpolating physical thermal maps..."):
                simulation_outputs = run_automated_simulation(vol_records, new_pulldown_input, target_sensors)
                
                if not simulation_outputs:
                    st.error("Simulation engine failed. Verify saved data metrics.")
                else:
                    for key_title, df_res in simulation_outputs.items():
                        st.markdown(f"### 📊 Predictions for {key_title}")
                        st.dataframe(df_res, use_container_width=True, hide_index=True)
# ================= TAB 2: DATA REPOSITORY ROOM =================
# Replace the file-processing logic inside your Tab 2 button with this:
try:
    # 1. PARSE PULLDOWN
    df_p_sum = pd.read_excel(pulldown_file, sheet_name='Summary', header=1)
    df_p_sum.columns = [str(c).strip() for c in df_p_sum.columns]
    df_p_sum.iloc[:, 0] = df_p_sum.iloc[:, 0].astype(str).str.strip()
    df_p_sum.set_index(df_p_sum.columns[0], inplace=True)
    df_p_sum.index.name = 'Metric'
    
    # S2 explicitly included in extraction keys matrix
    mapping_keys = {
        'tf-1':'tf1', 'tf-2':'tf2', 'tf-3':'tf3', 'tf-4':'tf4', 'tf-5':'tf5', 
        'tc-1':'tc1', 'tc-2':'tc2', 'tc-3':'tc3', 'tvc':'tvc', 'S2':'S2'
    }
    p_extracted = {f: float(df_p_sum.loc[xl, 'avg']) if xl in df_p_sum.index else 0.0 for f, xl in mapping_keys.items()}
    
    # Falling calculation strategy logic loop for sensor selection
    if 'Sensor' in df_p_sum.index and pd.notna(df_p_sum.loc['Sensor', 'avg']):
        resolved_sensor = float(df_p_sum.loc['Sensor', 'avg'])
    elif 'Eva Out100' in df_p_sum.index and 'Eva Out' in df_p_sum.index:
        resolved_sensor = (float(df_p_sum.loc['Eva Out100', 'avg']) + float(df_p_sum.loc['Eva Out', 'avg'])) / 2.0
    elif 'Eva Out' in df_p_sum.index:
        resolved_sensor = float(df_p_sum.loc['Eva Out', 'avg'])
    else:
        resolved_sensor = 0.0
    
    # 2. PARSE CPT DATA SHEETS
    df_cpt = pd.read_excel(cpt_file, sheet_name='Report', skiprows=8)
    df_cpt.dropna(subset=[df_cpt.columns[1], df_cpt.columns[2]], inplace=True)
    
    cpt_structured = {}
    current_flag = "Unknown"
    
    for _, r in df_cpt.iterrows():
        val_f1 = str(r.iloc[1]).strip()
        val_crit = str(r.iloc[2]).strip().lower()
        
        if val_f1 and val_f1 != 'nan' and val_f1 != current_flag:
            current_flag = val_f1
            
        if current_flag not in cpt_structured:
            cpt_structured[current_flag] = {}
            
        if val_crit in metric_types:
            cpt_structured[current_flag][val_crit] = {
                "tf-1": float(r.iloc[3]), "tf-2": float(r.iloc[4]), "tf-3": float(r.iloc[5]),
                "tf-4": float(r.iloc[6]), "tf-5": float(r.iloc[7]), "tc-1": float(r.iloc[9]),
                "tc-2": float(r.iloc[10]), "tc-3": float(r.iloc[11]), "tvc": float(r.iloc[13]),
                "S2": float(r.iloc[17]) if len(r) > 17 and pd.notna(r.iloc[17]) else 0.0,
                "Sensor": float(r.iloc[17]) if len(r) > 17 and pd.notna(r.iloc[17]) else 0.0
            }

    new_block = {
        "pulldown_data": p_extracted,
        "pulldown_baseline_sensor": resolved_sensor,
        "cpt_data": cpt_structured
    }
    
    if selected_volume not in st.session_state.db or isinstance(st.session_state.db[selected_volume], dict):
        st.session_state.db[selected_volume] = []
        
    st.session_state.db[selected_volume].append(new_block)
    st.session_state.db[selected_volume] = st.session_state.db[selected_volume][-5:]
    
    save_memory(st.session_state.db)
    st.success(f"📦 Pair saved! Core memory contains {len(st.session_state.db[selected_volume])}/5 active profile slots with S2 tracking active.")
    st.rerun()
    
except Exception as e:
    st.error(f"Error compiling automation metrics layout: {str(e)}")

# ================= TAB 3: REVIEWER DASHBOARD =================
with tab3:
    st.subheader("🔒 Repository Memory Inspection")
    if not st.session_state.reviewer_logged_in:
        pwd = st.text_input("Enter Reviewer Code Key:", type="password")
        if st.button("Unlock Database"):
            if pwd == REVIEWER_PASSWORD:
                st.session_state.reviewer_logged_in = True
                st.rerun()
    else:
        if st.button("Close Secure Lock"):
            st.session_state.reviewer_logged_in = False
            st.rerun()
        st.json(st.session_state.db.get(selected_volume, []))
