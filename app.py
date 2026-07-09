import json
import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st

# Must be the absolute first Streamlit command in the script
st.set_page_config(page_title="Refrigerator Simulator Hub", layout="wide")

# =================================================================
# 1. HARD DISK PERSISTENCE LAYER
# =================================================================
STORAGE_FILE = "simulator_storage.json"
REVIEWER_PASSWORD = "Admin@Cooling2026"

def load_memory_from_disk():
    """Reads saved database matrix from local disk storage on startup."""
    if os.path.exists(STORAGE_FILE):
        try:
            with open(STORAGE_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            st.error(f"⚠️ Error loading backup database file: {str(e)}")
    return {}

def save_memory_to_disk(db_matrix):
    """Writes the current database matrix to local disk storage instantly."""
    try:
        with open(STORAGE_FILE, "w") as f:
            json.dump(db_matrix, f, indent=4)
    except Exception as e:
        st.error(f"⚠️ Failed to write backup data to disk: {str(e)}")

# Legacy compatibility wrapper in case it's explicitly called later in your file
def save_memory(data):
    save_memory_to_disk(data)

# =================================================================
# 2. STATE INITIALIZATION ON STARTUP
# =================================================================
if "db" not in st.session_state:
    st.session_state.db = load_memory_from_disk()

if 'reviewer_logged_in' not in st.session_state:
    st.session_state.reviewer_logged_in = False

# Global configuration constants
tc_features = ['tf-1', 'tf-2', 'tf-3', 'tf-4', 'tf-5', 'tc-1', 'tc-2', 'tc-3', 'tvc', 'S2']
metric_types = ['mean', 'min', 'max', '(max+min)/2']

# =================================================================
# 3. UTILITY HELPER FUNCTIONS
# =================================================================
def normalize_sensor_name(name):
    if not isinstance(name, str):
        return ""
    # Lowercase, remove hyphens, spaces, and underscores (e.g., "tf-1" -> "tf1")
    return name.lower().replace(" ", "").replace("-", "").replace("_", "").strip()

# Helper initialization to safeguard nested multi-arrangement structure
def verify_db_structure(vol, arr_name, p_amb, c_amb):
    if vol not in st.session_state.db or not isinstance(st.session_state.db[vol], dict):
        st.session_state.db[vol] = {}
    
    # Catch old layouts or non-existent arrangements
    if arr_name not in st.session_state.db[vol] or not isinstance(st.session_state.db[vol][arr_name], dict):
        st.session_state.db[vol][arr_name] = {}
        
    if p_amb not in st.session_state.db[vol][arr_name] or not isinstance(st.session_state.db[vol][arr_name][p_amb], dict):
        st.session_state.db[vol][arr_name][p_amb] = {}
        
    if c_amb not in st.session_state.db[vol][arr_name][p_amb]:
        st.session_state.db[vol][arr_name][p_amb][c_amb] = []

# =================================================================
# 4. ADVANCED INTERPOLATION ENGINE
# =================================================================
def run_automated_simulation(volume_records, new_pulldown, target_sensors):
    results_map = {}
    if not volume_records:
        return results_map
        
    for idx, target_s in enumerate(target_sensors):
        predicted_modes = []
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
                    
                    p_base = [record['pulldown_data'].get(f, 0.0) for f in tc_features] + [record['pulldown_baseline_sensor']]
                    outputs = record['cpt_data'][flag][metric]
                    y_vector = [outputs.get(f, 0.0) for f in tc_features] + [outputs.get('S2', 0.0), outputs.get('Sensor', 0.0)]
                    
                    X_train.append(p_base)
                    y_train.append(y_vector)
                
                if len(X_train) >= 1:
                    knn = KNeighborsRegressor(n_neighbors=min(2, len(X_train)), weights='distance')
                    knn.fit(X_train, y_train)
                    
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

# =================================================================
# 5. THE REST OF YOUR STREAMLIT UI CODE GOES BELOW HERE...
# =================================================================

# ================= SIDEBAR: PROFILE & ARRANGEMENT MANAGER =================
# Initialize unique tracking IDs for clearing inputs if they don't exist
if "model_form_id" not in st.session_state:
    st.session_state.model_form_id = 0
if "arr_form_id" not in st.session_state:
    st.session_state.arr_form_id = 1000

# 1. 📁 Volume Profile Manager Header & Active Model Dropdown (Sorted Ascending)
st.sidebar.header("📁 Volume Profile Manager")
existing_volumes = list(st.session_state.db.keys())

if existing_volumes:
    existing_volumes.sort(reverse=False)
selected_volume = st.sidebar.selectbox("Active Refrigerator Model:", existing_volumes if existing_volumes else ["None"])

# --- Delete Model Option (Fixed Overlap) ---
if existing_volumes and selected_volume != "None":
    st.sidebar.caption("⚠️ Permanently removes this model and all its arrangements")
    if st.sidebar.button("🗑️ Delete Selected Model"):
        del st.session_state.db[selected_volume]
        save_memory(st.session_state.db)
        st.sidebar.warning(f"Model '{selected_volume}' deleted.")
        st.rerun()

st.sidebar.markdown("---")

# 2. Select Arrangement of Selected Volume & Deletion (Sorted Ascending)
if selected_volume and selected_volume in st.session_state.db and isinstance(st.session_state.db[selected_volume], dict):
    existing_arrangements = list(st.session_state.db[selected_volume].keys())
else:
    existing_arrangements = []

if existing_arrangements:
    existing_arrangements.sort(reverse=False)
    # Renamed the label exactly as requested
    selected_arrangement = st.sidebar.selectbox("Select Arrangement of Selected Volume:", existing_arrangements)
    
    # --- Delete Arrangement Option (Fixed Overlap) ---
    st.sidebar.caption("⚠️ Removes this arrangement data only")
    if st.sidebar.button("🗑️ Delete Selected Arrangement"):
        if len(existing_arrangements) > 1:
            del st.session_state.db[selected_volume][selected_arrangement]
            save_memory(st.session_state.db)
            st.sidebar.warning(f"Arrangement '{selected_arrangement}' deleted.")
            st.rerun()
        else:
            st.sidebar.error("❌ Cannot delete the last remaining arrangement. A model must have at least one arrangement layout. Delete the entire model instead.")
else:
    selected_arrangement = "None"
    st.sidebar.caption("No arrangements found. Register one below.")

st.sidebar.markdown("---")

# 3. ➕ Create New Arrangement Inputs
st.sidebar.subheader("📐 Design Arrangements")
new_arr = st.sidebar.text_input("➕ Create New Arrangement:", placeholder="e.g., A2", key=f"input_arr_{st.session_state.arr_form_id}")

if st.sidebar.button("Register Arrangement"):
    if new_arr and selected_volume and selected_volume != "None":
        new_arr_clean = new_arr.strip()
        if selected_volume not in st.session_state.db or not isinstance(st.session_state.db[selected_volume], dict):
            st.session_state.db[selected_volume] = {}
        if new_arr_clean not in st.session_state.db[selected_volume]:
            st.session_state.db[selected_volume][new_arr_clean] = {}
            save_memory(st.session_state.db)
            st.sidebar.success(f"Arrangement '{new_arr_clean}' registered under {selected_volume}!")
            
            # FORCE CELL CLEAR: Increment the arrangement form ID to reset input box
            st.session_state.arr_form_id += 1
            st.rerun()

st.sidebar.markdown("---")

# 4. ➕ Register New Volume Model Form (Moved to the end of the sidebar)
st.sidebar.subheader("➕ Register New Volume Model")
new_vol = st.sidebar.text_input("New Model Name:", placeholder="e.g., 365L", key=f"input_vol_{st.session_state.model_form_id}")
initial_arr = st.sidebar.text_input("Initial Arrangement Name:", placeholder="e.g., A1", key=f"input_init_{st.session_state.model_form_id}")

if st.sidebar.button("Add Volume Segment"):
    if new_vol and initial_arr:
        new_vol_clean = new_vol.strip()
        initial_arr_clean = initial_arr.strip()
        
        if new_vol_clean not in st.session_state.db:
            st.session_state.db[new_vol_clean] = {initial_arr_clean: {}}
            save_memory(st.session_state.db)
            st.success(f"Model {new_vol_clean} initialized with arrangement {initial_arr_clean}!")
            
            # FORCE CELL CLEAR: Increment the form ID to reset input boxes
            st.session_state.model_form_id += 1
            st.rerun()
        else:
            st.sidebar.error("This model name already exists.")
    elif new_vol or initial_arr:
        st.sidebar.error("⚠️ You must provide both the Model Name AND the Initial Arrangement Name.")


# === RESTORED TAB DEFINITIONS ===
tab1, tab2, tab3 = st.tabs(["🎛️ Run Automated Simulator", "🛠️ Data Repository Room", "🔍 Reviewer Dashboard"])

# ================= TAB 1: RUN AUTOMATED SIMULATOR =================
with tab1:
    st.subheader(f"Predict Multilevel CPT Matrices for [{selected_volume}] ({selected_arrangement})")
    
    c1, c2 = st.columns(2)
    with c1:
        sim_p_ambient = st.selectbox("Select Target Pulldown Ambient:", ["32°C", "43°C"], key="sim_p_amb")
    with c2:
        sim_c_ambient = st.selectbox("Select Target Respected CPT Ambient:", ["16°C", "32°C", "43°C"], key="sim_c_amb")
        
    p_key = "32C" if "32" in sim_p_ambient else "43C"
    c_key = "16C" if "16" in sim_c_ambient else ("32C" if "32" in sim_c_ambient else "43C")
    
    verify_db_structure(selected_volume, selected_arrangement, p_key, c_key)
    vol_records = st.session_state.db[selected_volume][selected_arrangement][p_key][c_key]
    
    if not vol_records:
        st.warning(f"⚠️ No matching profiles found under arrangement **{selected_arrangement}** for **Pulldown: {sim_p_ambient}** linked to **CPT: {sim_c_ambient}**.")
    else:
        st.markdown("#### Step 1: Input Current Pulldown Telemetry Vector")
        sim_pulldown_file = st.file_uploader(
            f"Auto-fill fields from local Pulldown Report ({sim_p_ambient})", 
            type=["xlsx"], key=f"sim_file_upload_{p_key}_{c_key}"
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
                
                mapping_keys = {
                    'tf-1':'tf1', 'tf-2':'tf2', 'tf-3':'tf3', 'tf-4':'tf4', 'tf-5':'tf5', 
                    'tc-1':'tc1', 'tc-2':'tc2', 'tc-3':'tc3', 'tvc':'tvc', 'S2':'S2'
                }
                for feat, xl_label in mapping_keys.items():
                    if xl_label in df_sim_p.index:
                        st.session_state.active_pulldown_form[feat] = float(df_sim_p.loc[xl_label, 'avg'])
                st.session_state.last_uploaded_sim_file = sim_pulldown_file
                st.toast("🟢 Parsed values pulled from sheet!", icon="📊")
            except Exception as e:
                st.error(f"Error parsing configuration: {str(e)}")

        u_cols = st.columns(10)
        new_pulldown_input = []
        default_defaults = {'tf-1': -24.4, 'tf-2': -21.8, 'tf-3': -22.8, 'tf-4': -26.2, 'tf-5': -26.4, 'tc-1': 1.9, 'tc-2': 1.6, 'tc-3': 0.5, 'tvc': 8.1, 'S2': 41.1}
        
        for i, feat in enumerate(tc_features):
            curr_val = st.session_state.active_pulldown_form.get(feat, 0.0) or default_defaults.get(feat, 0.0)
            val = u_cols[i].number_input(f"{feat}:", value=curr_val, key=f"sim_inp_{p_key}_{c_key}_{feat}")
            new_pulldown_input.append(val)
            
        st.markdown("---")
        st.markdown("#### Step 2: Set Multi-Sensor Simulation Steps")
        num_targets = st.number_input("Number of target sensor points (1 to 7):", min_value=1, max_value=7, value=3, step=1)
        
        target_sensors = []
        s_cols = st.columns(int(num_targets))
        for idx in range(int(num_targets)):
            s_val = s_cols[idx].number_input(f"Sensor Query Point {idx+1} (°C):", value=-10.0 + (idx * 2.5), key=f"q_s_{p_key}_{c_key}_{idx}")
            target_sensors.append(s_val)
            
        if st.button("🚀 Generate Predictive CPT Dataset Matrices", type="primary"):
            with st.spinner("Interpolating variant mappings..."):
                simulation_outputs = run_automated_simulation(vol_records, new_pulldown_input, target_sensors)
                if not simulation_outputs:
                    st.error("Simulation engine execution error.")
                else:
                    for key_title, df_res in simulation_outputs.items():
                        st.markdown(f"### 📊 Predictions for {key_title}")
                        st.dataframe(df_res, use_container_width=True, hide_index=True)

# ================= INITIALIZE STATE ON STARTUP =================
# Run this before your tab containers to ensure historical models pull cleanly
if "db" not in st.session_state:
    st.session_state.db = load_memory_from_disk()

# =================================================================
# 1. INITIALIZE KEY COUNTERS FOR THE FILE UPLOADERS (Put this near your startup state logic)
# =================================================================
if "p_file_key" not in st.session_state:
    st.session_state.p_file_key = 0
if "cpt_file_key" not in st.session_state:
    st.session_state.cpt_file_key = 0

# ================= TAB 2: DATA REPOSITORY ROOM =================
with tab2:
    st.subheader(f"Onboard Lab Reports for [{selected_volume}] ({selected_arrangement})")
    
    repo_c1, repo_c2 = st.columns(2)
    with repo_c1:
        repo_p_ambient = st.selectbox("Source Pulldown File Ambient Layer:", ["32°C", "43°C"], key="repo_p_amb")
    with repo_c2:
        repo_c_ambient = st.selectbox("Source Connected CPT File Ambient Layer:", ["16°C", "32°C", "43°C"], key="repo_c_amb")
        
    p_repo_key = "32C" if "32" in repo_p_ambient else "43C"
    c_repo_key = "16C" if "16" in repo_c_ambient else ("32C" if "32" in repo_c_ambient else "43C")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        # Dynamically controlled key using a counter inside session state
        repo_pulldown_file = st.file_uploader(
            f"Upload Pulldown Excel ({repo_p_ambient})", 
            type=["xlsx", "xls"], 
            key=f"r_p_file_{p_repo_key}_{c_repo_key}_v_{st.session_state.p_file_key}"
        )
    with col_f2:
        # Dynamically controlled key using a counter inside session state
        repo_cpt_file = st.file_uploader(
            f"Upload Respected CPT Excel ({repo_c_ambient})", 
            type=["xlsx", "xls"], 
            key=f"r_cpt_file_{p_repo_key}_{c_repo_key}_v_{st.session_state.cpt_file_key}"
        )
        
    if repo_pulldown_file and repo_cpt_file:
        if st.button("💾 Process and Train Simulator Memory Buffer", type="primary"):
            try:
                # -------------------------------------------------------------
                # 1. PARSE PULLDOWN DATA
                # -------------------------------------------------------------
                df_p_sum = pd.read_excel(repo_pulldown_file, sheet_name=0, header=None)
                df_p_sum[0] = df_p_sum[0].astype(str).apply(normalize_sensor_name)
                
                sheet_data = {}
                for _, row in df_p_sum.dropna(subset=[0]).iterrows():
                    lbl = row[0]
                    if lbl not in sheet_data:
                        try: sheet_data[lbl] = float(row[1])
                        except (ValueError, TypeError): continue

                mapping_keys = {
                    'tf-1':'tf1', 'tf-2':'tf2', 'tf-3':'tf3', 'tf-4':'tf4', 'tf-5':'tf5', 
                    'tc-1':'tc1', 'tc-2':'tc2', 'tc-3':'tc3', 'S2':'s2'
                }
                
                p_extracted = {}
                for target_key, normalized_label in mapping_keys.items():
                    p_extracted[target_key] = sheet_data.get(normalized_label, 0.0)
                        
                tvc_values = [sheet_data[lbl] for lbl in ['tvc1', 'tvc2', 'tvc3'] if lbl in sheet_data]
                if tvc_values:
                    p_extracted['tvc'] = round(sum(tvc_values) / len(tvc_values), 4)
                else:
                    p_extracted['tvc'] = 0.0
                    
                resolved_sensor = sheet_data.get('sensor', 0.0)
                
                # -------------------------------------------------------------
                # 2. PARSE CPT DATA (5-WAY CASCADE MULTI-FORMAT FAILSAFE)
                # -------------------------------------------------------------
                cpt_structured = {}
                parsed_successfully = False
                
                # --- STRATEGY A: Original Standard Layout Sheet ---
                try:
                    df_cpt = pd.read_excel(repo_cpt_file, sheet_name='Report', skiprows=8)
                    df_cpt.dropna(subset=[df_cpt.columns[1], df_cpt.columns[2]], inplace=True)
                    
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
                    if cpt_structured: parsed_successfully = True
                except Exception: pass

                # --- STRATEGY B: 2nd Sheet Multi-Row Header Format ---
                if not parsed_successfully:
                    try:
                        df_cpt_alt = pd.read_excel(repo_cpt_file, sheet_name=1, header=None)
                        if df_cpt_alt.iloc[1].astype(str).str.contains("AVG-1").any() == False:
                            row_10 = df_cpt_alt.iloc[10].astype(str).str.strip().str.lower().fillna("")
                            row_11 = df_cpt_alt.iloc[11].astype(str).str.strip().str.lower().fillna("")
                            
                            combined_headers = []
                            for r10, r11 in zip(row_10, row_11):
                                lbl = r11 if r11 and r11 != "nan" else r10
                                lbl = lbl.replace(" ", "").replace("-", "").replace("_", "")
                                if "vc(" in lbl: lbl = "vc"
                                if "sensor(" in lbl: lbl = "sensor"
                                if "%rt" in lbl or "runtime%" in lbl or "runtime" == lbl: lbl = "runtime_pct"
                                combined_headers.append(lbl)
                                
                            df_cpt_alt.columns = combined_headers
                            df_data_rows = df_cpt_alt.iloc[12:].dropna(subset=["datacriteria"]).copy()
                            
                            current_flag = "Unknown"
                            for _, row in df_data_rows.iterrows():
                                val_f1 = str(row.iloc[0]).strip()
                                val_crit = str(row.get("datacriteria", "")).strip().lower()
                                
                                if val_f1 and val_f1 != "nan" and val_f1 != current_flag: current_flag = val_f1
                                if current_flag not in cpt_structured: cpt_structured[current_flag] = {}
                                    
                                try: rt_val = float(row.get("runtime_pct", 0.0))
                                except (ValueError, TypeError): rt_val = 0.0
                                    
                                if rt_val == 100: continue
                                    
                                if val_crit in ["mean", "avg", "average"]:
                                    cpt_structured[current_flag][val_crit] = {
                                        "tf-1": float(row.get("tf1", 0.0)), "tf-2": float(row.get("tf2", 0.0)),
                                        "tf-3": float(row.get("tf3", 0.0)), "tf-4": float(row.get("tf4", 0.0)),
                                        "tf-5": float(row.get("tf5", 0.0)), "tc-1": float(row.get("tc1", 0.0)),
                                        "tc-2": float(row.get("tc2", 0.0)), "tc-3": float(row.get("tc3", 0.0)),
                                        "tvc":  float(row.get("vc", 0.0)),  "S2":   float(row.get("s2", 0.0)),
                                        "Sensor": float(row.get("sensor", 0.0))
                                    }
                            if cpt_structured: parsed_successfully = True
                    except Exception: pass

                # --- STRATEGY C: Section Layout Matrix Format ---
                if not parsed_successfully:
                    try:
                        df_cpt_seg = pd.read_excel(repo_cpt_file, sheet_name=0, header=None)
                        current_flag = "Unknown"
                        for idx, r in df_cpt_seg.iterrows():
                            val_0 = str(r.iloc[0]).strip()
                            if pd.notna(r.iloc[0]) and ("level" in val_0.lower() or "boost" in val_0.lower()):
                                current_flag = val_0
                                if current_flag not in cpt_structured: cpt_structured[current_flag] = {"mean": {}}
                                continue
                            if val_0.lower() in ["section", "min", "nan", ""] or pd.isna(r.iloc[0]): continue
                            clean_tag = normalize_sensor_name(val_0)
                            
                            if current_flag != "Unknown":
                                if clean_tag == "sensor":
                                    try: cpt_structured[current_flag]["mean"]["Sensor"] = float(r.iloc[5])
                                    except (ValueError, TypeError, IndexError): cpt_structured[current_flag]["mean"]["Sensor"] = float(r.iloc[1])
                                else:
                                    mapping_dict = {
                                        "tf1": "tf-1", "tf2": "tf-2", "tf3": "tf-3", "tf4": "tf-4", "tf5": "tf-5",
                                        "tc1": "tc-1", "tc2": "tc-2", "tc3": "tc-3", "s2": "S2"
                                    }
                                    if clean_tag in mapping_dict: cpt_structured[current_flag]["mean"][mapping_dict[clean_tag]] = float(r.iloc[8])
                                    elif "tvc" in clean_tag:
                                        if "tvc_vals" not in cpt_structured[current_flag]: cpt_structured[current_flag]["tvc_vals"] = []
                                        cpt_structured[current_flag]["tvc_vals"].append(float(r.iloc[8]))

                        for flg in cpt_structured:
                            if "tvc_vals" in cpt_structured[flg] and cpt_structured[flg]["tvc_vals"]:
                                cpt_structured[flg]["mean"]["tvc"] = round(sum(cpt_structured[flg]["tvc_vals"]) / len(cpt_structured[flg]["tvc_vals"]), 4)
                                del cpt_structured[flg]["tvc_vals"]
                        parsed_successfully = True
                    except Exception: pass

                # --- STRATEGY D: Summary-Block Layout Matrix Format ---
                if not parsed_successfully:
                    try:
                        df_cpt_block = pd.read_excel(repo_cpt_file, sheet_name='Report', header=None)
                        current_flag = "Unknown"
                        flag_extracted = {}
                        for idx, row in df_cpt_block.iterrows():
                            if pd.notna(row.iloc[9]) and "Test Flag:" in str(row.iloc[9]):
                                if current_flag != "Unknown" and flag_extracted:
                                    if "tvc_sum" in flag_extracted: del flag_extracted["tvc_sum"]
                                    if current_flag not in cpt_structured: cpt_structured[current_flag] = {"mean": {}}
                                    cpt_structured[current_flag]["mean"] = flag_extracted.copy()
                                current_flag = str(row.iloc[9]).split(":")[-1].strip()
                                flag_extracted = {"tf-1": 0.0, "tf-2": 0.0, "tf-3": 0.0, "tf-4": 0.0, "tf-5": 0.0, "tc-1": 0.0, "tc-2": 0.0, "tc-3": 0.0, "tvc": 0.0, "S2": 0.0, "Sensor": 0.0}
                                continue
                            if current_flag != "Unknown":
                                for col_offset in [0, 5, 10]:
                                    if col_offset < len(row):
                                        tag = str(row.iloc[col_offset]).strip()
                                        if tag in ["tF1", "tF2", "tF3", "tF4", "tF5"]:
                                            try: flag_extracted[f"tf-{tag[-1]}"] = float(row.iloc[col_offset + 1])
                                            except (ValueError, TypeError): pass
                                        elif tag in ["tc1", "tc2", "tc3"]:
                                            try: flag_extracted[f"tc-{tag[-1]}"] = float(row.iloc[col_offset + 1])
                                            except (ValueError, TypeError): pass
                                        elif tag in ["tVC1", "tVC2", "tVC3"]:
                                            if "tvc_sum" not in flag_extracted: flag_extracted["tvc_sum"] = []
                                            try:
                                                flag_extracted["tvc_sum"].append(float(row.iloc[col_offset + 1]))
                                                flag_extracted["tvc"] = round(sum(flag_extracted["tvc_sum"]) / len(flag_extracted["tvc_sum"]), 4)
                                            except (ValueError, TypeError): pass
                                        elif tag == "tS21":
                                            try: flag_extracted["S2"] = float(row.iloc[col_offset + 1])
                                            except (ValueError, TypeError): pass
                                        elif tag == "tSensor1":
                                            try: flag_extracted["Sensor"] = float(row.iloc[col_offset + 2])
                                            except (ValueError, TypeError): pass
                        if current_flag != "Unknown" and flag_extracted:
                            if "tvc_sum" in flag_extracted: del flag_extracted["tvc_sum"]
                            if current_flag not in cpt_structured: cpt_structured[current_flag] = {"mean": {}}
                            cpt_structured[current_flag]["mean"] = flag_extracted
                        if cpt_structured: parsed_successfully = True
                    except Exception: pass

                # --- STRATEGY E: Horizontal Wide Matrix Block Format ---
                if not parsed_successfully:
                    try:
                        df_cpt_horiz = pd.read_excel(repo_cpt_file, sheet_name=0, header=None)
                        for r_idx, row in df_cpt_horiz.iterrows():
                            val_0 = str(row.iloc[0]).strip()
                            if pd.notna(row.iloc[0]) and any(x in val_0.lower() for x in ["level", "boost"]):
                                current_flag = val_0
                                flag_extracted = {
                                    "tf-1": 0.0, "tf-2": 0.0, "tf-3": 0.0, "tf-4": 0.0, "tf-5": 0.0,
                                    "tc-1": 0.0, "tc-2": 0.0, "tc-3": 0.0, "tvc": 0.0, "S2": 0.0, "Sensor": 0.0
                                }
                                tvc_accumulator = []
                                
                                for target_offset in range(1, 11):
                                    if r_idx + target_offset < len(df_cpt_horiz):
                                        sub_row = df_cpt_horiz.iloc[r_idx + target_offset]
                                        for c_idx in range(len(sub_row)):
                                            tag = str(sub_row.iloc[c_idx]).strip().lower().replace("-", "")
                                            if tag in ["tf1", "tf2", "tf3", "tf4", "tf5"]:
                                                try: flag_extracted[f"tf-{tag[-1]}"] = float(sub_row.iloc[c_idx + 1])
                                                except (ValueError, TypeError): pass
                                            elif tag in ["tc1", "tc2", "tc3"]:
                                                try: flag_extracted[f"tc-{tag[-1]}"] = float(sub_row.iloc[c_idx + 1])
                                                except (ValueError, TypeError): pass
                                            elif "tvc" in tag:
                                                try: tvc_accumulator.append(float(sub_row.iloc[c_idx + 1]))
                                                except (ValueError, TypeError): pass
                                            elif tag == "s2":
                                                try: flag_extracted["S2"] = float(sub_row.iloc[c_idx + 1])
                                                except (ValueError, TypeError): pass
                                            elif tag == "sensor":
                                                try: flag_extracted["Sensor"] = float(sub_row.iloc[c_idx + 2])
                                                except (ValueError, TypeError): pass
                                
                                if tvc_accumulator:
                                    flag_extracted["tvc"] = round(sum(tvc_accumulator) / len(tvc_accumulator), 4)
                                    
                                if current_flag not in cpt_structured:
                                    cpt_structured[current_flag] = {}
                                cpt_structured[current_flag]["mean"] = flag_extracted
                        
                        if cpt_structured: parsed_successfully = True
                    except Exception: pass

                # -------------------------------------------------------------
                # 3. SAVE DATA MATRIX AND FORCE RETENTION TO HARD DISK
                # -------------------------------------------------------------
                if not parsed_successfully or not cpt_structured:
                    raise ValueError("CPT processing pipeline failed. Spreadsheet structural pattern unknown.")

                new_block = {"pulldown_data": p_extracted, "pulldown_baseline_sensor": resolved_sensor, "cpt_data": cpt_structured}
                
                verify_db_structure(selected_volume, selected_arrangement, p_repo_key, c_repo_key)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key].append(new_block)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key] = st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key][-10:]
                
                save_memory_to_disk(st.session_state.db)
                
                # -------------------------------------------------------------
                # 4. RESET FILE UPLOADERS BY INCREMENTING WIDGET KEYS
                # -------------------------------------------------------------
                st.session_state.p_file_key += 1
                st.session_state.cpt_file_key += 1
                
                st.success(f"🚀 Model Simulator Trained successfully! Hard-Backup saved to storage file context.")
                st.rerun()
            except Exception as e:
                st.error(f"Compilation error parsing dataset rows: {str(e)}")

# ================= TAB 3: REVIEWER DASHBOARD =================
with tab3:
    # 1. Secure Authentication Shield Check
    if not st.session_state.reviewer_logged_in:
        st.subheader("🔒 Secure Reviewer Administration Access")
        pass_input = st.text_input("Enter Laboratory Administrative Password:", type="password")
        if st.button("Unlock Admin Dashboard Space", type="primary"):
            if pass_input == REVIEWER_PASSWORD:
                st.session_state.reviewer_logged_in = True
                st.success("Access Granted. Re-routing to matrix editor...")
                st.rerun()
            else:
                st.error("Invalid Administrative Credentials. Access Denied.")
    else:
        # 2. Main Dashboard Layout Once Unlocked
        st.write("### 🔓 Repository Memory Inspection & Manipulation")
        
        c_header1, c_header2 = st.columns([4, 1])
        with c_header1:
            st.write(f"### 🛠️ Data Editor: {selected_volume} | Arrangement: {selected_arrangement}")
        with c_header2:
            if st.button("Close Secure Lock", use_container_width=True):
                st.session_state.reviewer_logged_in = False
                st.rerun()
                
        st.write("---")
        
        # 3. Setup Layout Columns for Inspection Selection
        insp_c1, insp_c2 = st.columns(2)
        
        with insp_c1:
            inspect_p_amb = st.selectbox(
                "Inspect Pulldown Ambient Space:", 
                ["32°C", "43°C"], 
                key="rev_inspect_p_amb"
            )
        with insp_c2:
            inspect_c_amb = st.selectbox(
                "Inspect CPT Ambient Space:", 
                ["16°C", "32°C", "43°C"], 
                index=1,  # Sets default to 32°C so both dropdowns align on load!
                key="rev_inspect_c_amb"
            )
            
        # 4. Dynamic Live Filtering Subtext Status (Fixes your string tracking mismatch error)
        st.caption(f"Currently filtering Pulldown: {inspect_p_amb} / CPT: {inspect_c_amb}")
        
        # 5. Extract Correct Normalized Mapping Dictionary Keys
        p_inspect_key = "32C" if "32" in inspect_p_amb else "43C"
        c_inspect_key = "16C" if "16" in inspect_c_amb else ("32C" if "32" in inspect_c_amb else "43C")
        
        # 6. Safety Verify DB Matrix Sub-Structure Exists
        verify_db_structure(selected_volume, selected_arrangement, p_inspect_key, c_inspect_key)
        records = st.session_state.db[selected_volume][selected_arrangement][p_inspect_key][c_inspect_key]
        
        # 7. Render Active Memory Pool Records Table
        if not records:
            st.info("No paired records matched for this specific arrangement selection.")
        else:
            st.success(f"Found {len(records)} trained datasets stored in hard backup matrix memory loop.")
            
            # (Your existing table plotting loops, data deletion widgets, or row-pop blocks can safely execute here using the synchronized 'records' variable)
