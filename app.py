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
            try:
                return json.load(f)
            except:
                return {}
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

# ================= SIDEBAR: PROFILE & ARRANGEMENT MANAGER =================
# 1. 📁 Volume Profile Manager Header & Active Model Dropdown
st.sidebar.header("📁 Volume Profile Manager")
existing_volumes = list(st.session_state.db.keys())
selected_volume = st.sidebar.selectbox("Active Refrigerator Model:", existing_volumes if existing_volumes else ["None"])

# --- Delete Model Option ---
if existing_volumes:
    if st.sidebar.button("🗑️ Delete Selected Model", help="Permanently removes this model and all its arrangements"):
        del st.session_state.db[selected_volume]
        save_memory(st.session_state.db)
        st.sidebar.warning(f"Model '{selected_volume}' deleted.")
        st.rerun()

st.sidebar.markdown("---")

# 2. ➕ Register New Volume Model Form (Forces an initial Arrangement Name)
st.sidebar.subheader("➕ Register New Volume Model")
new_vol = st.sidebar.text_input("New Model Name:", placeholder="e.g., 365L")
initial_arr = st.sidebar.text_input("Initial Arrangement Name:", placeholder="e.g., A1")

if st.sidebar.button("Add Volume Segment"):
    if new_vol and initial_arr:
        if new_vol not in st.session_state.db:
            # Create the model using your custom arrangement name instead of a generic default
            st.session_state.db[new_vol] = {initial_arr: {}}
            save_memory(st.session_state.db)
            st.success(f"Model {new_vol} initialized with arrangement {initial_arr}!")
            st.rerun()
        else:
            st.sidebar.error("This model name already exists.")
    elif new_vol or initial_arr:
        st.sidebar.error("⚠️ You must provide both the Model Name AND the Initial Arrangement Name.")

st.sidebar.markdown("---")

# 3. ➕ Create New Arrangement Inputs
st.sidebar.subheader("📐 Design Arrangements")
new_arr = st.sidebar.text_input("➕ Create New Arrangement:", placeholder="e.g., A2")
if st.sidebar.button("Register Arrangement"):
    if new_arr and selected_volume and selected_volume != "None":
        if selected_volume not in st.session_state.db or not isinstance(st.session_state.db[selected_volume], dict):
            st.session_state.db[selected_volume] = {}
        if new_arr not in st.session_state.db[selected_volume]:
            st.session_state.db[selected_volume][new_arr] = {}
            save_memory(st.session_state.db)
            st.sidebar.success(f"Arrangement '{new_arr}' registered under {selected_volume}!")
            st.rerun()

# 4. Ambient Arrangement Dropdown & Deletion
if selected_volume and selected_volume in st.session_state.db and isinstance(st.session_state.db[selected_volume], dict):
    existing_arrangements = list(st.session_state.db[selected_volume].keys())
else:
    existing_arrangements = []

if existing_arrangements:
    selected_arrangement = st.sidebar.selectbox("Ambient Arrangement:", existing_arrangements)
    
    # --- Delete Arrangement Option ---
    if st.sidebar.button("🗑️ Delete Selected Arrangement", help="Removes this arrangement data only"):
        if len(existing_arrangements) > 1:
            del st.session_state.db[selected_volume][selected_arrangement]
            save_memory(st.session_state.db)
            st.sidebar.warning(f"Arrangement '{selected_arrangement}' deleted.")
            st.rerun()
        else:
            st.sidebar.error("❌ Cannot delete the last remaining arrangement. A model must have at least one arrangement layout. Delete the entire model instead.")
else:
    selected_arrangement = "None"
    st.sidebar.caption("No arrangements found. Register one above.")

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
        repo_pulldown_file = st.file_uploader(f"Upload Pulldown Excel ({repo_p_ambient})", type=["xlsx"], key=f"r_p_file_{p_repo_key}_{c_repo_key}")
    with col_f2:
        repo_cpt_file = st.file_uploader(f"Upload Respected CPT Excel ({repo_c_ambient})", type=["xlsx"], key=f"r_cpt_file_{p_repo_key}_{c_repo_key}")
        
    if repo_pulldown_file and repo_cpt_file:
        if st.button("💾 Process and Save Report Pair to Memory Buffer", type="primary"):
            try:
                df_p_sum = pd.read_excel(repo_pulldown_file, sheet_name='Summary', header=1)
                df_p_sum.columns = [str(c).strip() for c in df_p_sum.columns]
                df_p_sum.iloc[:, 0] = df_p_sum.iloc[:, 0].astype(str).str.strip()
                df_p_sum.set_index(df_p_sum.columns[0], inplace=True)
                
                mapping_keys = {
                    'tf-1':'tf1', 'tf-2':'tf2', 'tf-3':'tf3', 'tf-4':'tf4', 'tf-5':'tf5', 
                    'tc-1':'tc1', 'tc-2':'tc2', 'tc-3':'tc3', 'tvc':'tvc', 'S2':'S2'
                }
                p_extracted = {f: float(df_p_sum.loc[xl, 'avg']) if xl in df_p_sum.index else 0.0 for f, xl in mapping_keys.items()}
                resolved_sensor = float(df_p_sum.loc['Sensor', 'avg']) if 'Sensor' in df_p_sum.index else 0.0
                
                df_cpt = pd.read_excel(repo_cpt_file, sheet_name='Report', skiprows=8)
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

                new_block = {"pulldown_data": p_extracted, "pulldown_baseline_sensor": resolved_sensor, "cpt_data": cpt_structured}
                
                verify_db_structure(selected_volume, selected_arrangement, p_repo_key, c_repo_key)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key].append(new_block)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key] = st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key][-5:]
                
                save_memory(st.session_state.db)
                st.success(f"📦 Paired Set saved into: {selected_volume} → {selected_arrangement}")
                st.rerun()
            except Exception as e:
                st.error(f"Compilation error parsing dataset rows: {str(e)}")

# ================= TAB 3: REVIEWER DASHBOARD =================
with tab3:
    st.subheader("🔒 Repository Memory Inspection & Manipulation")
    
    rd_c1, rd_c2 = st.columns(2)
    with rd_c1:
        rev_p_ambient = st.selectbox("Inspect Pulldown Ambient Space:", ["32°C", "43°C"], key="rev_p_amb")
    with rd_c2:
        rev_c_ambient = st.selectbox("Inspect CPT Ambient Space:", ["16°C", "32°C", "43°C"], key="rev_c_amb")
        
    p_rev_key = "32C" if "32" in rev_p_ambient else "43C"
    c_rev_key = "16C" if "16" in rev_c_ambient else ("32C" if "32" in rev_c_ambient else "43C")
    
    if not st.session_state.reviewer_logged_in:
        pwd = st.text_input("Enter Reviewer Code Key:", type="password", key=f"rev_pwd_{selected_volume}_{p_rev_key}_{c_rev_key}")
        if st.button("Unlock Database", key=f"rev_unl_{selected_volume}_{p_rev_key}_{c_rev_key}"):
            if pwd == REVIEWER_PASSWORD:
                st.session_state.reviewer_logged_in = True
                st.rerun()
    else:
        col_header, col_lock = st.columns([4, 1])
        with col_header:
            st.markdown(f"### 🛠️ Data Editor: **{selected_volume}** | Arrangement: **{selected_arrangement}**")
            st.caption(f"Currently filtering Pulldown: {rev_p_ambient} / CPT: {rev_c_ambient}")
        with col_lock:
            if st.button("Close Secure Lock", type="secondary", use_container_width=True, key=f"c_lck_{selected_volume}_{p_rev_key}_{c_rev_key}"):
                st.session_state.reviewer_logged_in = False
                st.rerun()
                
        verify_db_structure(selected_volume, selected_arrangement, p_rev_key, c_rev_key)
        records_list = st.session_state.db[selected_volume][selected_arrangement][p_rev_key][c_rev_key]
        
        if not records_list:
            st.info("No paired records matched for this specific arrangement selection.")
        else:
            st.markdown("#### 📐 Part A: Manipulate Pulldown Telemetry Base Vectors")
            pulldown_rows = []
            for idx, r in enumerate(records_list):
                row_dict = {"Record ID": idx}
                row_dict.update(r.get("pulldown_data", {}))
                row_dict["Baseline Sensor"] = r.get("pulldown_baseline_sensor", 0.0)
                pulldown_rows.append(row_dict)
                
            df_pulldown_edit = pd.DataFrame(pulldown_rows)
            edited_pulldown_df = st.data_editor(df_pulldown_edit, hide_index=True, use_container_width=True, key=f"ed_p_{selected_volume}_{p_rev_key}_{c_rev_key}")
            
            st.markdown("#### 📊 Part B: View/Verify Structural CPT Target Maps")
            cpt_tabs = st.tabs([f"Record Entry #{i}" for i in range(len(records_list))])
            for idx, c_tab in enumerate(cpt_tabs):
                with c_tab:
                    raw_cpt = records_list[idx].get("cpt_data", {})
                    cpt_rows = []
                    for flag, modes in raw_cpt.items():
                        for mode, metrics in modes.items():
                            meta = {"TestFlag": flag, "Criteria": mode}
                            meta.update(metrics)
                            cpt_rows.append(meta)
                    if cpt_rows:
                        st.dataframe(pd.DataFrame(cpt_rows), hide_index=True, use_container_width=True)
                    else:
                        st.write("No connected CPT matrix rows found.")

            st.markdown("---")
            st.markdown("#### ⚙️ Database Modification Panel")
            del_col, save_col = st.columns([2, 3])
            
            with del_col:
                record_to_delete = st.selectbox("❌ Delete Record ID instance:", options=list(range(len(records_list))), format_func=lambda x: f"Record Entry #{x}", key=f"sel_del_{selected_volume}_{p_rev_key}_{c_rev_key}")
                if st.button("🗑️ Drop Selected Record Pair", type="secondary", use_container_width=True, key=f"b_del_{selected_volume}_{p_rev_key}_{c_rev_key}"):
                    records_list.pop(record_to_delete)
                    st.session_state.db[selected_volume][selected_arrangement][p_rev_key][c_rev_key] = records_list
                    save_memory(st.session_state.db)
                    st.toast(f"Dropped Entry #{record_to_delete}.", icon="🗑️")
                    st.rerun()
                    
            with save_col:
                st.write("\n\n")
                if st.button("💾 Commit Manipulated Changes permanently to Disk", type="primary", use_container_width=True, key=f"b_cmt_{selected_volume}_{p_rev_key}_{c_rev_key}"):
                    try:
                        for index, row in edited_pulldown_df.iterrows():
                            if index < len(records_list):
                                updated_p_data = {f: float(row.get(f, 0.0)) for f in tc_features}
                                records_list[index]["pulldown_data"] = updated_p_data
                                records_list[index]["pulldown_baseline_sensor"] = float(row.get("Baseline Sensor", 0.0))
                        
                        st.session_state.db[selected_volume][selected_arrangement][p_rev_key][c_rev_key] = records_list
                        save_memory(st.session_state.db)
                        st.success("🎉 Changes permanently saved for this model arrangement layout!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error preserving configuration: {str(e)}")
