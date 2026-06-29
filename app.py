import streamlit as st
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import json
import os

# Page layout configurations
st.set_page_config(page_title="Refrigerator Simulator Hub", layout="wide")

MEMORY_FILE = "simulator_memory.json"
REVIEWER_PASSWORD = "Admin@Cooling2026"  # Modify password credentials here if required

# --- SYSTEM MEMORY MANAGEMENT ---
def load_memory():
    """Loads stored training data sets from local JSON storage."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_memory(data):
    """Saves training data sets to local JSON storage."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Instantiate memory state session tracker
if 'db' not in st.session_state:
    st.session_state.db = load_memory()

if 'reviewer_logged_in' not in st.session_state:
    st.session_state.reviewer_logged_in = False

tc_features = ['tf-1', 'tf-2', 'tf-3', 'tf-4', 'tf-5', 'tc-1', 'tc-2', 'tc-3', 'tvc']

# UPDATED: Added 'Respected' as the 5th structural metric row
metric_types = ['Mean', 'Min', 'Max', '(Max+Min)/2', 'Respected']

# --- CORE INTERPOLATION SIMULATION ENGINE ---
def run_manual_simulation(volume_records, pulldown_input, target_sensor):
    """Executes multi-output inverse-distance regression using manual training maps."""
    simulated_results = []
    
    for metric in metric_types:
        X_train = []
        y_train = []
        
        # Loop through each manually entered training block for this volume
        for record_id, data in volume_records.items():
            metric_rows = [r for r in data['outputs'] if r['Metric_Type'] == metric]
            if not metric_rows:
                continue
            
            # Locate the output row closest to the user's targeted sensor value
            sensor_diffs = [abs(r['Sensor'] - target_sensor) for r in metric_rows]
            closest_idx = np.argmin(sensor_diffs)
            matched_row = metric_rows[closest_idx]
            
            p_vals = [data['pulldown'][feat] for feat in tc_features]
            m_vals = [matched_row[feat] for feat in tc_features]
            
            X_train.append(p_vals)
            y_train.append(m_vals)
            
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        if len(X_train) == 0:
            return None
            
        # Machine learning dynamic curve tracking based on geometric distance
        knn = KNeighborsRegressor(n_neighbors=min(3, len(X_train)), weights='distance')
        knn.fit(X_train, y_train)
        
        user_vector = np.array(pulldown_input).reshape(1, -1)
        predicted_values = knn.predict(user_vector)[0]
        
        simulated_results.append([metric] + list(np.round(predicted_values, 2)))
        
    return pd.DataFrame(simulated_results, columns=['Metric_Type'] + tc_features)


# --- USER INTERFACE SIDEBAR ---
st.sidebar.header("📁 Volume Profile Manager")
existing_volumes = list(st.session_state.db.keys())
if not existing_volumes:
    existing_volumes = ["365L"]

selected_volume = st.sidebar.selectbox("Active Refrigerator Volume Model:", existing_volumes)

new_vol = st.sidebar.text_input("➕ Register New Volume Model:")
if st.sidebar.button("Add Volume Segment"):
    if new_vol and new_vol not in st.session_state.db:
        st.session_state.db[new_vol] = {}
        save_memory(st.session_state.db)
        st.success(f"Model {new_vol} registered.")
        st.rerun()

st.sidebar.markdown("---")

# --- THREE-TAB NAVIGATION CONTROL ---
tab1, tab2, tab3 = st.tabs(["🎛️ Run Simulator", "🛠️ Manual Training Room", "🔍 Reviewer Dashboard"])

# ================= TAB 1: RUN SIMULATOR =================
with tab1:
    st.subheader(f"Execute Live Multi-Level Predictions ({selected_volume})")
    vol_records = st.session_state.db.get(selected_volume, {})
    
    if not vol_records:
        st.warning(f"⚠️ No training data has been entered for **[{selected_volume}]** yet. Please head to the 'Manual Training Room' tab to submit reference records first.")
    else:
        st.markdown("#### Step 1: Input Current Pulldown Data Vector")
        u_cols = st.columns(9)
        user_pulldown = []
        for i, feat in enumerate(tc_features):
            default_val = -29.2 if feat=='tf-1' else (-27.0 if feat=='tf-2' else (-28.4 if feat=='tf-3' else (-29.8 if feat=='tf-4' else (-30.5 if feat=='tf-5' else (-3.0 if feat=='tc-1' else (-4.1 if feat=='tc-2' else (-5.4 if feat=='tc-3' else 1.4)))))))
            val = u_cols[i].number_input(f"{feat}:", value=default_val, key=f"sim_run_{feat}")
            user_pulldown.append(val)
            
        st.markdown("---")
        st.markdown("#### Step 2: Set Target Levels and Values")
        num_set_levels = st.number_input("Number of Set Levels required:", min_value=1, max_value=10, value=2, step=1)
        
        sensor_levels = []
        level_cols = st.columns(int(num_set_levels))
        
        for i in range(int(num_set_levels), 0, -1):
            col_idx = int(num_set_levels) - i
            s_val = level_cols[col_idx].number_input(f"Sensor Value (Level-{i}):", value=-21.5 + (col_idx * 2), key=f"sim_lvl_{i}")
            sensor_levels.append((f"Level-{i}", s_val))
            
        st.markdown("---")
        st.markdown("#### 📊 Predicted Level Configurations Result")
        
        for lvl_name, sensor_target in sensor_levels:
            st.markdown(f"##### 📍 Outputs for **{lvl_name}** (Target Sensor Temp: `{sensor_target}°C`)")
            output_table = run_manual_simulation(vol_records, user_pulldown, sensor_target)
            if output_table is not None:
                st.dataframe(output_table.style.format(precision=2), use_container_width=True)
            else:
                st.error(f"Could not compute matrix for {lvl_name}. Ensure adequate data is saved in training room.")

# ================= TAB 2: MANUAL TRAINING ROOM =================
with tab2:
    st.subheader(f"Train Dataset Arrays for [{selected_volume}]")
    num_sets = st.number_input("Number of training blocks to register:", min_value=1, max_value=10, value=1, step=1, key="train_num_sets")
    
    with st.form("manual_training_form"):
        submitted_data_sets = {}
        
        for s in range(int(num_sets)):
            st.markdown(f"### 📊 Dataset Block #{s+1}")
            st.markdown("**Pulldown Initialization Values:**")
            p_cols = st.columns(10)
            p_data = {}
            for idx, feat in enumerate(tc_features):
                p_data[feat] = p_cols[idx].number_input(f"{feat}", value=-25.0, key=f"p_{s}_{feat}")
            p_data['Sensor'] = p_cols[9].number_input("Sensor", value=-25.0, key=f"p_{s}_sens")
            
            # UPDATED: This dynamic code segment now loops 5 times instead of 4, adding the 'Respected' tracking fields
            st.markdown("**Respected Outputs (Mean, Min, Max, (Max+Min)/2, Respected):**")
            out_rows = []
            for metric in metric_types:
                m_cols = st.columns(11)
                m_cols[0].markdown(f"**{metric}**")
                m_data = {"Metric_Type": metric}
                for idx, feat in enumerate(tc_features):
                    m_data[feat] = m_cols[idx+1].number_input(f"{feat}", value=0.0, label_visibility="collapsed", key=f"m_{s}_{metric}_{feat}")
                m_data['Sensor'] = m_cols[10].number_input("Sensor Target", value=0.0, label_visibility="collapsed", key=f"m_{s}_{metric}_sens")
                out_rows.append(m_data)
                
            submitted_data_sets[f"set_{len(st.session_state.db.get(selected_volume, {})) + s}"] = {
                "pulldown": p_data,
                "outputs": out_rows
            }
            st.markdown("---")
            
        if st.form_submit_button("Commit Data Blocks to Memory Banks"):
            if selected_volume not in st.session_state.db:
                st.session_state.db[selected_volume] = {}
            for k, v in submitted_data_sets.items():
                st.session_state.db[selected_volume][k] = v
            save_memory(st.session_state.db)
            st.success(f"Retrained matrix for [{selected_volume}] successfully!")
            st.rerun()

# ================= TAB 3: SECURED REVIEWER DASHBOARD =================
with tab3:
    st.subheader("🔒 Reviewer Audit Room")
    
    if not st.session_state.reviewer_logged_in:
        col_sec1, col_sec2 = st.columns([1, 2])
        with col_sec1:
            entered_password = st.text_input("Enter Reviewer Security Key:", type="password")
            if st.button("Unlock Dashboard Data"):
                if entered_password == REVIEWER_PASSWORD:
                    st.session_state.reviewer_logged_in = True
                    st.success("Access Granted.")
                    st.rerun()
                else:
                    st.error("Incorrect credentials. Access denied.")
    else:
        if st.button("🔒 Lock & Close Audit View"):
            st.session_state.reviewer_logged_in = False
            st.rerun()
            
        st.markdown(f"### 📋 Reviewing Dataset Profiles Saved for Model: **[{selected_volume}]**")
        vol_records = st.session_state.db.get(selected_volume, {})
        
        if not vol_records:
            st.info("No records available to review for this configuration model.")
        else:
            pulldown_rows_list = []
            output_rows_list = []
            
            for set_id, content in vol_records.items():
                p_row = {"Set_ID": set_id}
                p_row.update(content["pulldown"])
                pulldown_rows_list.append(p_row)
                
                for out_row in content["outputs"]:
                    o_row = {"Set_ID": set_id}
                    o_row.update(out_row)
                    output_rows_list.append(o_row)
            
            df_review_pulldown = pd.DataFrame(pulldown_rows_list)
            df_review_outputs = pd.DataFrame(output_rows_list)
            
            st.markdown("#### 🔍 Apply Dynamic Thermocouple Range Filters")
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                filter_target = st.selectbox("Select Thermocouple to inspect:", tc_features)
            with fc2:
                filter_operator = st.selectbox("Logical Operator Condition:", ["Show All Data", "Less than or equal (<=)", "Greater than or equal (>=)"])
            with fc3:
                filter_value = st.number_input("Threshold Temperature Value (°C):", value=0.0, step=0.5)
                
            if filter_operator != "Show All Data":
                if filter_operator == "Less than or equal (<=)":
                    df_review_pulldown = df_review_pulldown[df_review_pulldown[filter_target] <= filter_value]
                    df_review_outputs = df_review_outputs[df_review_outputs[filter_target] <= filter_value]
                elif filter_operator == "Greater than or equal (>=)":
                    df_review_pulldown = df_review_pulldown[df_review_pulldown[filter_target] >= filter_value]
                    df_review_outputs = df_review_outputs[df_review_outputs[filter_target] >= filter_value]
            
            st.markdown("---")
            st.markdown(f"**Historical Pulldown Baseline Data Matrix ({len(df_review_pulldown)} entries matches):**")
            st.dataframe(df_review_pulldown.style.format(precision=2), use_container_width=True)
            
            st.markdown(f"**Historical Stabilization Segment Data Matrix ({len(df_review_outputs)} cycle rows matches):**")
            st.dataframe(df_review_outputs.style.format(precision=2), use_container_width=True)
