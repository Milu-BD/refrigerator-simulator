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

# --- SIDEBAR WORKSPACE MANAGER ---
st.sidebar.header("📁 Volume Profile Manager")
existing_volumes = list(st.session_state.db.keys()) or ["365L"]
selected_volume = st.sidebar.selectbox("Active Refrigerator Volume Model:", existing_volumes)

new_vol = st.sidebar.text_input("➕ Register New Volume Model:")
if st.sidebar.button("Add Volume Segment"):
    if new_vol and new_vol not in st.session_state.db:
        st.session_state.db[new_vol] = {"32C": [], "43C": []}
        save_memory(st.session_state.db)
        st.success(f"Model {new_vol} registered.")
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🌡️ Universal Ambient Selection")
selected_ambient = st.sidebar.radio("Select Target Test Ambient conditions:", ["32°C", "43°C"], index=0)
ambient_key = "32C" if "32" in selected_ambient else "43C"

tab1, tab2, tab3 = st.tabs(["🎛️ Run Automated Simulator", "🛠️ Data Repository Room", "🔍 Reviewer Dashboard"])

# ================= TAB 1: RUN AUTOMATED SIMULATOR =================
with tab1:
    st.subheader(f"Predict Multilevel CPT Matrices ({selected_volume} @ {selected_ambient})")
    
    # Safely handle old data shapes or initialize missing layers
    if selected_volume not in st.session_state.db:
        st.session_state.db[selected_volume] = {"32C": [], "43C": []}
    elif isinstance(st.session_state.db[selected_volume], list):
        # Migrating flat records legacy setup safely to structured ambient profile
        old_data = st.session_state.db[selected_volume]
        st.session_state.db[selected_volume] = {"32C": old_data, "43C": []}
        
    vol_records = st.session_state.db[selected_volume].get(ambient_key, [])
    
    if not vol_records:
        st.warning(f"⚠️ No profile instances saved for **[{selected_volume}]** under **{selected_ambient}** yet. Please upload them in the Data Repository Room.")
    else:
        st.markdown("#### Step 1: Input Current Pulldown Telemetry Vector")
        
        sim_pulldown_file = st.file_uploader(
            f"Optional: Auto-fill fields from local {selected_ambient} Pulldown Report (.xlsx)", 
            type=["xlsx"], 
            key=f"sim_tab_p_uploader_{ambient_key}"
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
                st.toast("🟢 Telemetry values successfully pulled from spreadsheet!", icon="📊")
            except Exception as e:
                st.error(f"Error parsing local test profile: {str(e)}")

        if not sim_pulldown_file and st.session_state.last_uploaded_sim_file is not None:
            st.session_state.active_pulldown_form = {feat: 0.0 for feat in tc_features}
            st.session_state.last_uploaded_sim_file = None

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
                key=f"automated_sim_input_{ambient_key}_{feat}"
            )
            new_pulldown_input.append(val)
            
        st.markdown("---")
        st.markdown("#### Step 2: Set Multi-Sensor Simulation Steps")
        num_targets = st.number_input("Number of target sensor variables to predict (1 to 7):", min_value=1, max_value=7, value=3, step=1)
        
        target_sensors = []
        s_cols = st.columns(int(num_targets))
        for idx in range(int(num_targets)):
            s_val = s_cols[idx].number_input(f"Sensor Query Point {idx+1} (°C):", value=-10.0 + (idx * 2.5), key=f"query_s_{ambient_key}_{idx}")
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
with tab2:
    st.subheader(f"Onboard Lab Reports for [{selected_volume}] under {selected_ambient}")
    st.info(f"ℹ️ Note: File targets will commit to the specific **{selected_ambient} Ambient** memory repository block.")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        repo_pulldown_file = st.file_uploader("Upload Pulldown Excel Report File (.xlsx)", type=["xlsx"], key=f"repo_p_file_{ambient_key}")
    with col_f2:
        repo_cpt_file = st.file_uploader("Upload CPT Reference Excel File (.xlsx)", type=["xlsx"], key=f"repo_cpt_file_{ambient_key}")
        
    st.markdown("---")
    st.markdown("#### 🔍 Repository File Upload Verification Status")
    vs1, vs2 = st.columns(2)
    vs1.markdown(f"**Pulldown Uploaded:** {'🟢 File Ready' if repo_pulldown_file else '🔴 Missing File Input'}")
    vs2.markdown(f"**CPT Uploaded:** {'🟢 File Ready' if repo_cpt_file else '🔴 Missing File Input'}")
    
    if repo_pulldown_file and repo_cpt_file:
        if st.button("💾 Process and Save Report Pair to Memory Buffer"):
            try:
                # FIXED: Corrected precise file binding links
                df_p_sum = pd.read_excel(repo_pulldown_file, sheet_name='Summary', header=1)
                df_p_sum.columns = [str(c).strip() for c in df_p_sum.columns]
                df_p_sum.iloc[:, 0] = df_p_sum.iloc[:, 0].astype(str).str.strip()
                df_p_sum.set_index(df_p_sum.columns[0], inplace=True)
                df_p_sum.index.name = 'Metric'
                
                mapping_keys = {
                    'tf-1':'tf1', 'tf-2':'tf2', 'tf-3':'tf3', 'tf-4':'tf4', 'tf-5':'tf5', 
                    'tc-1':'tc1', 'tc-2':'tc2', 'tc-3':'tc3', 'tvc':'tvc', 'S2':'S2'
                }
                p_extracted = {f: float(df_p_sum.loc[xl, 'avg']) if xl in df_p_sum.index else 0.0 for f, xl in mapping_keys.items()}
                
                if 'Sensor' in df_p_sum.index and pd.notna(df_p_sum.loc['Sensor', 'avg']):
                    resolved_sensor = float(df_p_sum.loc['Sensor', 'avg'])
                elif 'Eva Out100' in df_p_sum.index and 'Eva Out' in df_p_sum.index:
                    resolved_sensor = (float(df_p_sum.loc['Eva Out100', 'avg']) + float(df_p_sum.loc['Eva Out', 'avg'])) / 2.0
                elif 'Eva Out' in df_p_sum.index:
                    resolved_sensor = float(df_p_sum.loc['Eva Out', 'avg'])
                else:
                    resolved_sensor = 0.0
                
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

                new_block = {
                    "pulldown_data": p_extracted,
                    "pulldown_baseline_sensor": resolved_sensor,
                    "cpt_data": cpt_structured
                }
                
                if selected_volume not in st.session_state.db or not isinstance(st.session_state.db[selected_volume], dict):
                    st.session_state.db[selected_volume] = {"32C": [], "43C": []}
                    
                st.session_state.db[selected_volume][ambient_key].append(new_block)
                st.session_state.db[selected_volume][ambient_key] = st.session_state.db[selected_volume][ambient_key][-5:]
                
                save_memory(st.session_state.db)
                st.success(f"📦 Pair saved to {selected_ambient} repository! Buffer contains {len(st.session_state.db[selected_volume][ambient_key])}/5 profile slots.")
                st.rerun()
                
            except Exception as e:
                st.error(f"Error compiling automation metrics layout: {str(e)}")

# ================= TAB 3: REVIEWER DASHBOARD =================
with tab3:
    st.subheader("🔒 Repository Memory Inspection & Manipulation")
    
    if not st.session_state.reviewer_logged_in:
        pwd = st.text_input("Enter Reviewer Code Key:", type="password", key="reviewer_pwd_input")
        if st.button("Unlock Database", key="reviewer_unlock_btn"):
            if pwd == REVIEWER_PASSWORD:
                st.session_state.reviewer_logged_in = True
                st.rerun()
    else:
        col_header, col_lock = st.columns([4, 1])
        with col_header:
            st.markdown(f"### 🛠️ Data Editor for Profile: **{selected_volume}** ({selected_ambient})")
        with col_lock:
            if st.button("Close Secure Lock", type="secondary", use_container_width=True):
                st.session_state.reviewer_logged_in = False
                st.rerun()
                
        # Safely fetch the current active array of records
        if selected_volume not in st.session_state.db:
            st.session_state.db[selected_volume] = {"32C": [], "43C": []}
        records_list = st.session_state.db[selected_volume].get(ambient_key, [])
        
        if not records_list:
            st.info(f"No records found to manipulate for {selected_volume} under the {selected_ambient} ambient.")
        else:
            st.markdown("#### 📐 Part A: Manipulate Pulldown Telemetry Base Vectors")
            st.caption("Double-click any cell to edit numbers directly. Select a row and hit 'Delete' on your keyboard to remove a dataset record.")
            
            # Flatten the pulldown list into a structured DataFrame for editing
            pulldown_rows = []
            for idx, r in enumerate(records_list):
                row_dict = {"Record ID": idx}
                row_dict.update(r.get("pulldown_data", {}))
                row_dict["Baseline Sensor"] = r.get("pulldown_baseline_sensor", 0.0)
                pulldown_rows.append(row_dict)
                
            df_pulldown_edit = pd.DataFrame(pulldown_rows)
            
            # Render the interactive editor for Pulldown Data
            edited_pulldown_df = st.data_editor(
                df_pulldown_edit, 
                hide_index=True, 
                num_rows="dynamic", # Allows row deletion
                use_container_width=True,
                key=f"editor_p_{selected_volume}_{ambient_key}"
            )
            
            st.markdown("#### 📊 Part B: View/Verify Structural CPT Target Maps")
            st.caption("Reviewing parsed CPT step matrices linked to your records.")
            
            # Render CPT view profiles as sub-tabs so it doesn't clutter the space
            cpt_tabs = st.tabs([f"Record Entry #{i}" for i in range(len(records_list))])
            for idx, c_tab in enumerate(cpt_tabs):
                with c_tab:
                    raw_cpt = records_list[idx].get("cpt_data", {})
                    # Reconstruct readable layout for inspection
                    cpt_rows = []
                    for flag, modes in raw_cpt.items():
                        for mode, metrics in modes.items():
                            meta = {"TestFlag": flag, "Criteria": mode}
                            meta.update(metrics)
                            cpt_rows.append(meta)
                    if cpt_rows:
                        st.dataframe(pd.DataFrame(cpt_rows), hide_index=True, use_container_width=True)
                    else:
                        st.write("No CPT sub-matrix linked to this index slot.")
            
            # ================= TAB 3: REVIEWER DASHBOARD =================
with tab3:
    st.subheader("🔒 Repository Memory Inspection & Manipulation")
    
    if not st.session_state.reviewer_logged_in:
        pwd = st.text_input("Enter Reviewer Code Key:", type="password", key="reviewer_pwd_input")
        if st.button("Unlock Database", key="reviewer_unlock_btn"):
            if pwd == REVIEWER_PASSWORD:
                st.session_state.reviewer_logged_in = True
                st.rerun()
    else:
        col_header, col_lock = st.columns([4, 1])
        with col_header:
            st.markdown(f"### 🛠️ Dedicated Data Editor: **{selected_volume}** ({selected_ambient})")
        with col_lock:
            if st.button("Close Secure Lock", type="secondary", use_container_width=True):
                st.session_state.reviewer_logged_in = False
                st.rerun()
                
        # Safely fetch the current active array of paired records
        if selected_volume not in st.session_state.db:
            st.session_state.db[selected_volume] = {"32C": [], "43C": []}
        records_list = st.session_state.db[selected_volume].get(ambient_key, [])
        
        if not records_list:
            st.info(f"No paired records found to manipulate for {selected_volume} under the {selected_ambient} ambient.")
        else:
            st.markdown("#### 📐 Combined Data Matrix (Pulldown & Respected CPT Targets)")
            st.caption("Each row represents one independent 2-file pair simulation profile. Double-click cells to edit. Select a row and hit 'Delete' on your keyboard to clear that entire dataset from memory entirely.")
            
            # Re-architect data into a flat spreadsheet structure where each row is an isolated 2-file slot
            flattened_rows = []
            for idx, r in enumerate(records_list):
                row_dict = {"Slot ID": idx}
                
                # Prefix Pulldown features
                p_data = r.get("pulldown_data", {})
                for f in tc_features:
                    row_dict[f"Pulldown_{f}"] = p_data.get(f, 0.0)
                row_dict["Pulldown_Baseline_Sensor"] = r.get("pulldown_baseline_sensor", 0.0)
                
                # Extract connected CPT metric flags (focusing on standard reference flag headers parsed)
                cpt_data = r.get("cpt_data", {})
                first_flag = list(cpt_data.keys())[0] if cpt_data.keys() else "Unknown_Flag"
                row_dict["CPT_Linked_Flag"] = first_flag
                
                # Flatten the 'mean' criteria metrics as editable samples in the row matrix
                mean_cpt = cpt_data.get(first_flag, {}).get("mean", {})
                for f in tc_features:
                    row_dict[f"CPT_Mean_{f}"] = mean_cpt.get(f, 0.0)
                row_dict["CPT_Mean_Sensor"] = mean_cpt.get("Sensor", 0.0)
                
                flattened_rows.append(row_dict)
                
            df_master_edit = pd.DataFrame(flattened_rows)
            
            # Render the unified matrix editor
            edited_master_df = st.data_editor(
                df_master_edit, 
                hide_index=True, 
                num_rows="dynamic", # Enables full row deletion capabilities
                use_container_width=True,
                key=f"master_editor_{selected_volume}_{ambient_key}"
            )
            
            # --- DATABASE STRUCTURAL RE-COMMIT SAVE ENGINE ---
            st.markdown("---")
            if st.button("💾 Commit Manipulated Changes permanently to Disk", type="primary"):
                try:
                    remaining_slots = edited_master_df["Slot ID"].tolist() if "Slot ID" in edited_master_df.columns else []
                    
                    updated_records = []
                    for current_slot in remaining_slots:
                        if current_slot < len(records_list):
                            # Retain original structure frame but override data values from edited UI row matrix
                            orig_block = records_list[current_slot]
                            ui_row = edited_master_df[edited_master_df["Slot ID"] == current_slot].iloc[0]
                            
                            # 1. Update Pulldown block metrics
                            updated_p_data = {f: float(ui_row.get(f"Pulldown_{f}", 0.0)) for f in tc_features}
                            orig_block["pulldown_data"] = updated_p_data
                            orig_block["pulldown_baseline_sensor"] = float(ui_row.get("Pulldown_Baseline_Sensor", 0.0))
                            
                            # 2. Update Respected CPT block metrics
                            flag_key = str(ui_row.get("CPT_Linked_Flag", "Unknown_Flag"))
                            
                            # Rebuild all 4 key metrics profiles consistently with user changes
                            updated_cpt_metrics = {f: float(ui_row.get(f"CPT_Mean_{f}", 0.0)) for f in tc_features}
                            updated_sensor_val = float(ui_row.get("CPT_Mean_Sensor", 0.0))
                            
                            # Re-map criteria sub-objects safely back to database frame schema
                            orig_block["cpt_data"] = {
                                flag_key: {
                                    metric: {
                                        **updated_cpt_metrics,
                                        "Sensor": updated_sensor_val
                                    } for metric in metric_types
                                }
                            }
                            
                            updated_records.append(orig_block)
                    
                    # Overwrite state buffer cache and save changes to disk
                    st.session_state.db[selected_volume][ambient_key] = updated_records
                    save_memory(st.session_state.db)
                    st.success("🎉 Combined dataset memory maps updated successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to synchronize combined database adjustments: {str(e)}")
