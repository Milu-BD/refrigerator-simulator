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

# Mapping keys to standardize features across different sheets
tc_features = ['tf-1', 'tf-2', 'tf-3', 'tf-4', 'tf-5', 'tc-1', 'tc-2', 'tc-3', 'tvc']
metric_types = ['Mean', 'Min', 'Max', '(Max+Min)/2']

# --- FILE PARSING ENGINES ---
def parse_pulldown_file(uploaded_file):
    """Extracts average validation vectors from files like CPT Report_365L_F-11708_R028442.xlsx"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Summary')
        # Standardize structure formatting
        df.columns = [str(c).strip() for c in df.iloc[0]]
        df = df.iloc[1:].copy()
        df.rename(columns={df.columns[0]: 'Feature'}, inplace=True)
        df['Feature'] = df['Feature'].astype(str).str.strip().str.lower()
        
        # Create mapping dictionary from the 'avg' column
        avg_map = dict(zip(df['Feature'], pd.to_numeric(df['avg'], errors='coerce')))
        
        # Align features safely
        extracted = {
            'tf-1': avg_map.get('tf1', -25.0),
            'tf-2': avg_map.get('tf2', -25.0),
            'tf-3': avg_map.get('tf3', -25.0),
            'tf-4': avg_map.get('tf4', -25.0),
            'tf-5': avg_map.get('tf5', -25.0),
            'tc-1': avg_map.get('tc1', 0.0),
            'tc-2': avg_map.get('tc2', 0.0),
            'tc-3': avg_map.get('tc3', 0.0),
            'tvc': avg_map.get('tvc', 0.0),
            'Sensor': avg_map.get('sensor', None),
            'Eva Out': avg_map.get('eva out', None),
            'Eva Out100': avg_map.get('eva out100', None)
        }
        return extracted
    except Exception as e:
        st.error(f"Error parsing Pulldown File template format: {e}")
        return None

def parse_cpt_file(uploaded_file):
    """Extracts stabilization performance profiles from files like CPT Report_365L_F-11708_R028323.xlsx"""
    try:
        df = pd.read_excel(uploaded_file, sheet_name='Report')
        
        # Locate performance level data rows securely
        records = {}
        current_level = None
        
        for idx, row in df.iterrows():
            val_0 = str(row.iloc[0]).strip()
            val_1 = str(row.iloc[1]).strip()
            val_2 = str(row.iloc[2]).strip().lower()
            
            if "Level-" in val_1:
                current_level = val_1
                records[current_level] = []
                
            if current_level and val_2 in ['mean', 'min', 'max', '(max+min)/2']:
                metric_name = 'Mean' if val_2=='mean' else ('Min' if val_2=='min' else ('Max' if val_2=='max' else '(Max+Min)/2'))
                
                # Extract indices from spreadsheet alignment
                m_data = {
                    "Metric_Type": metric_name,
                    "tf-1": pd.to_numeric(row.iloc[3], errors='coerce'),
                    "tf-2": pd.to_numeric(row.iloc[4], errors='coerce'),
                    "tf-3": pd.to_numeric(row.iloc[5], errors='coerce'),
                    "tf-4": pd.to_numeric(row.iloc[6], errors='coerce'),
                    "tf-5": pd.to_numeric(row.iloc[7], errors='coerce'),
                    "tc-1": pd.to_numeric(row.iloc[9], errors='coerce'),
                    "tc-2": pd.to_numeric(row.iloc[10], errors='coerce'),
                    "tc-3": pd.to_numeric(row.iloc[11], errors='coerce'),
                    "tvc": pd.to_numeric(row.iloc[13], errors='coerce'), # tvc1 used as primary index baseline
                    "Sensor": pd.to_numeric(row.iloc[17], errors='coerce')
                }
                records[current_level].append(m_data)
        return records
    except Exception as e:
        st.error(f"Error parsing CPT Matrix Report layout: {e}")
        return None


# --- PREDICTION CORE ---
def run_automated_simulation(cpt_matrix, pulldown_vector, target_sensor):
    simulated_results = []
    for metric in metric_types:
        X_train, y_train = [], []
        
        for lvl_name, rows in cpt_matrix.items():
            matched_row = next((r for r in rows if r['Metric_Type'] == metric), None)
            if not matched_row or pd.isna(matched_row['Sensor']):
                continue
                
            X_train.append([pulldown_vector[f] for f in tc_features])
            y_train.append([matched_row[f] for f in tc_features])
            
        if not X_train:
            continue
            
        knn = KNeighborsRegressor(n_neighbors=min(2, len(X_train)), weights='distance')
        knn.fit(X_train, y_train)
        
        user_vector = np.array([pulldown_vector[f] for f in tc_features]).reshape(1, -1)
        predicted_values = knn.predict(user_vector)[0]
        simulated_results.append([metric] + list(np.round(predicted_values, 2)))
        
    if simulated_results:
        return pd.DataFrame(simulated_results, columns=['Metric_Type'] + tc_features)
    return None


# --- USER INTERFACE SIDEBAR ---
st.sidebar.header("📁 System Volume Profile")
existing_volumes = list(st.session_state.db.keys()) if st.session_state.db else ["365L"]
selected_volume = st.sidebar.selectbox("Active Refrigerator Model Volume:", existing_volumes)

# --- TAB CONTROL ---
tab1, tab2, tab3 = st.tabs(["🎛️ Run Automated Simulator", "🛠️ Data Repository Room", "🔍 Reviewer Dashboard"])

# ================= TAB 1: RUN SIMULATOR =================
with tab1:
    st.subheader("Simulate Interpolated Performance Levels")
    
    vol_data = st.session_state.db.get(selected_volume, {})
    if "cpt_matrix" not in vol_data:
        st.warning("⚠️ No active validation dataset repository found. Please head over to 'Data Repository Room' tab to upload spreadsheet profiles first.")
    else:
        st.markdown("### Step 1: Provide Live Pulldown Validation Source File")
        pulldown_file = st.file_uploader("Upload Pulldown Data Excel (.xlsx)", type=["xlsx"], key="run_pulldown_upload")
        
        if pulldown_file:
            pulldown_data = parse_pulldown_file(pulldown_file)
            
            if pulldown_data:
                st.success("Pulldown data vector parsed successfully.")
                
                # Core Option Logic Selection UI
                st.markdown("### Step 2: Choose Interpolation Logic Route")
                
                has_sensor = pulldown_data['Sensor'] is not None and not pd.isna(pulldown_data['Sensor'])
                
                options_list = ["Predict CPT by Sensor", "Predict CPT by Eva out"]
                selected_strategy = st.radio("Available Computation Routes:", options_list, index=0 if has_sensor else 1)
                
                computed_target_sensor = None
                
                if selected_strategy == "Predict CPT by Sensor":
                    if has_sensor:
                        computed_target_sensor = pulldown_data['Sensor']
                        st.info(f"Using parsed file Sensor baseline target value: `{computed_target_sensor}°C`")
                    else:
                        st.error("No valid Sensor metric column discovered inside the uploaded summary file. Please fall back to 'Predict CPT by Eva out' mode below.")
                
                if selected_strategy == "Predict CPT by Eva out" or computed_target_sensor is None:
                    eva_out = pulldown_data.get('Eva Out')
                    eva_out100 = pulldown_data.get('Eva Out100')
                    
                    if eva_out100 is not None and not pd.isna(eva_out100) and eva_out is not None and not pd.isna(eva_out):
                        computed_target_sensor = (eva_out100 + eva_out) / 2
                        st.info(f"Target Sensor derived from formula calculation `(Eva Out100 + Eva Out) / 2`: `{computed_target_sensor:.2f}°C`")
                    elif eva_out is not None and not pd.isna(eva_out):
                        computed_target_sensor = eva_out
                        st.info(f"Eva Out100 column missing or corrupted. Fallback single target `Eva Out` captured: `{computed_target_sensor:.2f}°C`")
                    else:
                        st.error("Critical calculation error: Both Eva Out and Eva Out100 data values are absent in this file dataset summary.")
                
                if computed_target_sensor is not None:
                    st.markdown("---")
                    st.markdown(f"#### 📊 Automated Target Prediction Matrix (Evaluated Target: `{computed_target_sensor:.2f}°C`)")
                    
                    out_table = run_automated_simulation(vol_data['cpt_matrix'], pulldown_data, computed_target_sensor)
                    if out_table is not None:
                        st.markdown("`Metric_Type` | `tf-1` | `tf-2` | `tf-3` | `tf-4` | `tf-5` | `tc-1` | `tc-2` | `tc-3` | `tvc`")
                        st.dataframe(out_table.style.format(precision=2), use_container_width=True)
                    else:
                        st.error("Failed to solve estimation equations. Ensure target matrix limits fall within historical dataset scope bounds.")

# ================= TAB 2: DATA REPOSITORY ROOM =================
with tab2:
    st.subheader("Manage Component System Matrices")
    
    target_vol_model = st.text_input("Active Target Volume Identity Tag:", value=selected_volume)
    st.markdown("Upload a baseline CPT reference file (like **`CPT Report_365L_F-11708_R028323.xlsx`**) to calibrate model constraints.")
    
    cpt_matrix_file = st.file_uploader("Upload CPT Reference Profile Matrix (.xlsx)", type=["xlsx"], key="repository_cpt_upload")
    
    if st.button("Publish Profiles to System Memory Banks"):
        if cpt_matrix_file and target_vol_model:
            parsed_matrix = parse_cpt_file(cpt_matrix_file)
            if parsed_matrix:
                if target_vol_model not in st.session_state.db:
                    st.session_state.db[target_vol_model] = {}
                st.session_state.db[target_vol_model]['cpt_matrix'] = parsed_matrix
                save_memory(st.session_state.db)
                st.success(f"System metrics for model group [{target_vol_model}] compiled and successfully synchronized online.")
                st.rerun()
        else:
            st.error("Missing configuration actions: Please upload a valid sheet file component first.")

# ================= TAB 3: SECURED REVIEWER DASHBOARD =================
with tab3:
    st.subheader("🔒 Reviewer Audit Room")
    if not st.session_state.reviewer_logged_in:
        entered_password = st.text_input("Enter Reviewer Security Key:", type="password")
        if st.button("Unlock Dashboard Data"):
            if entered_password == REVIEWER_PASSWORD:
                st.session_state.reviewer_logged_in = True
                st.rerun()
            else:
                st.error("Incorrect credentials.")
    else:
        if st.button("🔒 Lock & Close Audit View"):
            st.session_state.reviewer_logged_in = False
            st.rerun()
            
        st.markdown(f"### Live Database Configuration Profiles: [{selected_volume}]")
        st.json(st.session_state.db.get(selected_volume, {}))
