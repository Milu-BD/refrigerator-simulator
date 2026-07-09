# ==============================================================================
# BLOCK 1: CORE INITIALIZATION, SIMULATOR CACHING & HARDWARE CONSTANTS
# Function: Sets up global parameters, default dictionaries, and file serialization.
# ==============================================================================
import os
import pickle
import pandas as pd
import streamlit as st

# Configure page metadata
st.set_page_config(page_title="OQC Lab Thermal Simulator Engine", layout="wide")

# Lab Environment and Target Appliance Configurations
VOLUME_OPTIONS = ["175L Single Door", "200L Single Door", "220L Double Door"]
ARRANGEMENT_OPTIONS = ["Standard Chiller", "Modified Thermostat Capillary (Post-2021)"]
REVIEWER_PASSWORD = "OQC_LAB_ADMIN_SECURE"

metric_types = ["mean", "avg", "average"]

# Default schema structure for the database memory matrix
def get_empty_db_schema():
    return {
        vol: {
            arr: {
                p_amb: {c_amb: [] for c_amb in ["16C", "32C", "43C"]}
                for p_amb in ["32C", "43C"]
            } for arr in ARRANGEMENT_OPTIONS
        } for vol in VOLUME_OPTIONS
    }

DB_FILE_PATH = "simulator_memory_buffer.pkl"

def load_memory_from_disk():
    if os.path.exists(DB_FILE_PATH):
        try:
            with open(DB_FILE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            return get_empty_db_schema()
    return get_empty_db_schema()

def save_memory_to_disk(db_matrix):
    try:
        with open(DB_FILE_PATH, "wb") as f:
            pickle.dump(db_matrix, f)
    except Exception as e:
        st.error(f"Critical System Error: Hard-disk backup serialization failed: {str(e)}")

# Ensure session states exist
if "db" not in st.session_state:
    st.session_state.db = load_memory_from_disk()
if "p_file_key" not in st.session_state:
    st.session_state.p_file_key = 100
if "cpt_file_key" not in st.session_state:
    st.session_state.cpt_file_key = 500
if "reviewer_logged_in" not in st.session_state:
    st.session_state.reviewer_logged_in = False

def verify_db_structure(vol, arr, p_amb, c_amb):
    if vol not in st.session_state.db:
        st.session_state.db[vol] = {}
    if arr not in st.session_state.db[vol]:
        st.session_state.db[vol][arr] = {}
    if p_amb not in st.session_state.db[vol][arr]:
        st.session_state.db[vol][arr][p_amb] = {}
    if c_amb not in st.session_state.db[vol][arr][p_amb]:
        st.session_state.db[vol][arr][p_amb][c_amb] = []

# ==============================================================================
# BLOCK 2: UTILITY TEXT-PARSERS & INTERPOLATION NUMERICAL MATHEMATICS
# Function: Data cleaning strings and execution logic for continuous regression loops.
# ==============================================================================
def normalize_sensor_name(name):
    """Normalizes variation in sensor notation from automated loggers."""
    if pd.isna(name):
        return ""
    s = str(name).strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if "tf" in s: return s
    if "tc" in s: return s
    if "s2" in s: return "s2"
    if "sensor" in s: return "sensor"
    if "tvc" in s: return s
    return s

def to_float(val):
    """Safely converts Excel cell variants to python floats."""
    try:
        if pd.isna(val) or str(val).strip().lower() in ['nan', '']:
            return 0.0
        return float(str(val).strip())
    except (ValueError, TypeError):
        return 0.0

def interpolate_prediction(x_target, x1, x2, y1_dict, y2_dict):
    """Applies a strict linear interpolation across dictionary structures."""
    if abs(x1 - x2) < 1e-5:
        return y1_dict
    
    factor = (x_target - x1) / (x2 - x1)
    interpolated = {}
    for flag_key, metrics in y1_dict.items():
        interpolated[flag_key] = {}
        for metric_name, channels in metrics.items():
            interpolated[flag_key][metric_name] = {}
            for channel_name, val1 in channels.items():
                val2 = y2_dict.get(flag_key, {}).get(metric_name, {}).get(channel_name, val1)
                interpolated[flag_key][metric_name][channel_name] = round(val1 + factor * (val2 - val1), 4)
    return interpolated

# ==============================================================================
# BLOCK 3: USER DASHBOARD - SIDEBAR COMPONENT INTERFACE
# Function: Renders structural arrangement criteria controllers and volume contexts.
# ==============================================================================
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/thermomenter.png", width=70)
    st.title("OQC Simulator Hub")
    st.write("Performance Test Judgment Engine")
    st.write("---")
    
    selected_volume = st.selectbox("Target Appliance Volume Profile:", VOLUME_OPTIONS)
    selected_arrangement = st.selectbox("Physical Hardware Arrangement Matrix:", ARRANGEMENT_OPTIONS)
    
    st.write("---")
    st.caption("Designed for OQC Laboratory automation compliance standards. Supports multi-format telemetry logs.")

# ==============================================================================
# BLOCK 4: TAB 1 - AUTOMATED SIMULATOR ENGINE (PREDICTION & OVERRIDES)
# Function: Computes predicted operational parameters from multi-dimensional tables.
# ==============================================================================
tab1, tab2, tab3 = st.tabs([
    "📈 Auto-Interpolation Simulator Engine", 
    "📥 Telemetry Lab Repository Room", 
    "🛠️ Secure Reviewer Dashboard Space"
])

with tab1:
    st.subheader(f"Predictive Matrix Output for [{selected_volume}]")
    st.info(f"Current Target Configuration Layer: {selected_arrangement}")
    
    # User Input Parameters
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        sim_p_ambient = st.selectbox("Target Pulldown Ambient:", ["32°C", "43°C"], key="sim_p_amb")
    with sim_c2:
        sim_c_ambient = st.selectbox("Target Connected CPT Ambient:", ["16°C", "32°C", "43°C"], key="sim_c_amb")
    with sim_c3:
        input_sensor_target = st.number_input("Desired Target Sensor Value (°C):", value=4.0, step=0.1)
        
    p_sim_key = "32C" if "32" in sim_p_ambient else "43C"
    c_sim_key = "16C" if "16" in sim_c_ambient else ("32C" if "32" in sim_c_ambient else "43C")
    
    # Query database from cache layers
    verify_db_structure(selected_volume, selected_arrangement, p_sim_key, c_sim_key)
    available_records = st.session_state.db[selected_volume][selected_arrangement][p_sim_key][c_sim_key]
    
    if len(available_records) < 2:
        st.warning("⚠️ Insufficient trained memory blocks inside the data repository. Please upload at least **2 companion reference files** in Tab 2 to calibrate the linear interpolation matrices.")
    else:
        # Sort history records based on their recorded baseline hardware sensors
        sorted_records = sorted(available_records, key=lambda k: k.get("pulldown_baseline_sensor", 0.0))
        sensor_baselines = [r.get("pulldown_baseline_sensor", 0.0) for r in sorted_records]
        
        # Locate bracket anchors
        idx1, idx2 = None, None
        if input_sensor_target <= sensor_baselines[0]:
            idx1, idx2 = 0, 1
        elif input_sensor_target >= sensor_baselines[-1]:
            idx1, idx2 = len(sorted_records) - 2, len(sorted_records) - 1
        else:
            for i in range(len(sensor_baselines) - 1):
                if sensor_baselines[i] <= input_sensor_target <= sensor_baselines[i+1]:
                    idx1, idx2 = i, i+1
                    break
        
        r1, r2 = sorted_records[idx1], sorted_records[idx2]
        x1, x2 = r1["pulldown_baseline_sensor"], r2["pulldown_baseline_sensor"]
        
        # Run matrix loop calculation
        simulated_cpt = interpolate_prediction(input_sensor_target, x1, x2, r1["cpt_data"], r2["cpt_data"])
        
        st.write("### 🚀 Generated Simulation Matrix Output")
        st.success(f"Mathematical Interpolation verified successfully bounded between reference points {x1}°C and {x2}°C.")
        
        # Construct displayable structures
        output_rows = []
        for flg, metrics in simulated_cpt.items():
            for crit, channels in metrics.items():
                row_item = {"Test Knob Flag": flg, "Data Criteria": crit}
                row_item.update(channels)
                output_rows.append(row_item)
                
        if output_rows:
            st.dataframe(pd.DataFrame(output_rows), use_container_width=True, hide_index=True)

# ==============================================================================
# BLOCK 5: TAB 2 - DATA REPOSITORY ROOM & 5-WAY MULTI-FORMAT FAILSAFE
# Function: Handles openpyxl visibility checking and parsing strategies A, B, C, D, E.
# ==============================================================================
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
        repo_pulldown_file = st.file_uploader(
            f"Upload Pulldown Excel ({repo_p_ambient})", 
            type=["xlsx", "xls"], 
            key=f"r_p_file_{p_repo_key}_{c_repo_key}_v_{st.session_state.p_file_key}"
        )
    with col_f2:
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
                p_extracted['tvc'] = round(sum(tvc_values) / len(tvc_values), 4) if tvc_values else 0.0
                resolved_sensor = sheet_data.get('sensor', 0.0)
                
                # -------------------------------------------------------------
                # 2. PARSE CPT DATA (5-WAY CASCADE MULTI-FORMAT FAILSAFE)
                # -------------------------------------------------------------
                cpt_structured = {}
                parsed_successfully = False
                
                import openpyxl
import pandas as pd

# ---------------------------------------------------------
# Strategy A : Visible Ambient Parser
# ---------------------------------------------------------

def to_float(v):
    try:
        if pd.isna(v):
            return 0.0
        return float(v)
    except:
        return 0.0


def parse_visible_cpt(file):

    wb = openpyxl.load_workbook(file, data_only=True)

    sheet_name = (
        "ANALYSIS REPORT"
        if "ANALYSIS REPORT" in wb.sheetnames
        else wb.sheetnames[0]
    )

    ws = wb[sheet_name]

    # --------------------------------------------
    # Read ONLY visible rows
    # --------------------------------------------
    visible_rows = []

    for r in range(1, ws.max_row + 1):

        if ws.row_dimensions[r].hidden:
            continue

        visible_rows.append(
            [ws.cell(r, c).value for c in range(1, ws.max_column + 1)]
        )

    df = pd.DataFrame(visible_rows)

    # --------------------------------------------
    # Locate the table header automatically
    # --------------------------------------------
    header_row = None

    for i in range(len(df)):

        row = (
            df.iloc[i]
            .astype(str)
            .str.lower()
            .str.replace(" ", "")
        )

        if (
            row.str.contains("tf1").any()
            and row.str.contains("tc1").any()
        ):
            header_row = i
            break

    if header_row is None:
        raise Exception("Unable to locate CPT table.")

    headers = (
        df.iloc[header_row]
        .astype(str)
        .str.strip()
        .tolist()
    )

    df = df.iloc[header_row + 1:].reset_index(drop=True)
    df.columns = headers

    # --------------------------------------------
    # Normalize column names
    # --------------------------------------------

    rename = {}

    for c in df.columns:

        x = str(c).lower()

        x = (
            x.replace("-", "")
             .replace(" ", "")
             .replace("_", "")
        )

        if x == "tf1":
            rename[c] = "tf1"

        elif x == "tf2":
            rename[c] = "tf2"

        elif x == "tf3":
            rename[c] = "tf3"

        elif x == "tf4":
            rename[c] = "tf4"

        elif x == "tf5":
            rename[c] = "tf5"

        elif x == "tc1":
            rename[c] = "tc1"

        elif x == "tc2":
            rename[c] = "tc2"

        elif x == "tc3":
            rename[c] = "tc3"

        elif "vc" in x:
            rename[c] = "tvc"

        elif "chilleravg" in x or x == "s2":
            rename[c] = "S2"

        elif "sensor" in x:
            rename[c] = "Sensor"

        elif "criteria" in x:
            rename[c] = "Criteria"

        elif "flag" in x or "knob" in x:
            rename[c] = "Flag"

    df.rename(columns=rename, inplace=True)

    # --------------------------------------------
    # Forward fill Test Flag
    # --------------------------------------------

    df["Flag"] = df["Flag"].ffill()

    # --------------------------------------------
    # Extract required values
    # --------------------------------------------

    cpt_structured = {}

    for flag in df["Flag"].dropna().unique():

        block = df[df["Flag"] == flag]

        mean_row = block[
            block["Criteria"]
            .astype(str)
            .str.lower()
            .eq("mean")
        ]

        min_row = block[
            block["Criteria"]
            .astype(str)
            .str.lower()
            .eq("min")
        ]

        if mean_row.empty:
            continue

        mean_row = mean_row.iloc[0]

        sensor_value = (
            to_float(min_row.iloc[0]["Sensor"])
            if not min_row.empty
            else 0.0
        )

        cpt_structured[flag] = {

            "mean": {

                "tf-1": to_float(mean_row["tf1"]),
                "tf-2": to_float(mean_row["tf2"]),
                "tf-3": to_float(mean_row["tf3"]),
                "tf-4": to_float(mean_row["tf4"]),
                "tf-5": to_float(mean_row["tf5"]),

                "tc-1": to_float(mean_row["tc1"]),
                "tc-2": to_float(mean_row["tc2"]),
                "tc-3": to_float(mean_row["tc3"]),

                "tvc": to_float(mean_row["tvc"]),

                "S2": to_float(mean_row["S2"]),

                "Sensor": sensor_value,
            }

        }

    return cpt_structured

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
                # DB STORAGE REFRESH & STATE RERUN
                # -------------------------------------------------------------
                if not parsed_successfully or not cpt_structured:
                    raise ValueError("CPT processing pipeline failed. Spreadsheet structural pattern unknown.")

                new_block = {"pulldown_data": p_extracted, "pulldown_baseline_sensor": resolved_sensor, "cpt_data": cpt_structured}
                verify_db_structure(selected_volume, selected_arrangement, p_repo_key, c_repo_key)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key].append(new_block)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key] = st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key][-10:]
                
                save_memory_to_disk(st.session_state.db)
                
                st.session_state.p_file_key += 1
                st.session_state.cpt_file_key += 1
                st.success(f"🚀 Model Simulator Trained successfully! Hard-Backup saved to storage file context.")
                st.rerun()
            except Exception as e:
                st.error(f"Compilation error parsing dataset rows: {str(e)}")

# ==============================================================================
# BLOCK 6: TAB 3 - SECURE ADMINISTRATIVE REVIEWER DASHBOARD
# Function: Secure password gate, data record inspections, and memory purging.
# ==============================================================================
with tab3:
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
        st.write("### 🔓 Repository Memory Inspection & Manipulation")
        
        c_header1, c_header2 = st.columns([4, 1])
        with c_header1:
            st.write(f"### 🛠️ Data Editor: {selected_volume} | Arrangement: {selected_arrangement}")
        with c_header2:
            if st.button("Close Secure Lock", use_container_width=True):
                st.session_state.reviewer_logged_in = False
                st.rerun()
                
        st.write("---")
        
        insp_c1, insp_c2 = st.columns(2)
        with insp_c1:
            inspect_p_amb = st.selectbox("Inspect Pulldown Ambient Space:", ["32°C", "43°C"], key="rev_inspect_p_amb")
        with insp_c2:
            inspect_c_amb = st.selectbox("Inspect CPT Ambient Space:", ["16°C", "32°C", "43°C"], index=1, key="rev_inspect_c_amb")
            
        st.caption(f"Currently filtering Pulldown: {inspect_p_amb} / CPT: {inspect_c_amb}")
        
        p_inspect_key = "32C" if "32" in inspect_p_amb else "43C"
        c_inspect_key = "16C" if "16" in inspect_c_amb else ("32C" if "32" in inspect_c_amb else "43C")
        
        verify_db_structure(selected_volume, selected_arrangement, p_inspect_key, c_inspect_key)
        records = st.session_state.db[selected_volume][selected_arrangement][p_inspect_key][c_inspect_key]
        
        if not records:
            st.info("No paired records matched for this specific arrangement selection.")
        else:
            st.success(f"Found {len(records)} trained datasets stored in hard backup matrix memory loop.")
            
            for run_idx, record in enumerate(records):
                with st.expander(f"📦 Trained Dataset Record #{run_idx + 1}", expanded=(run_idx == 0)):
                    c_btn1, c_btn2 = st.columns([4, 1])
                    with c_btn1:
                        st.markdown(f"**Baseline Sensor Target Setting:** `{record.get('pulldown_baseline_sensor', 0.0)}°C`")
                    with c_btn2:
                        if st.button("🗑️ Delete Dataset", key=f"del_ds_{p_inspect_key}_{c_inspect_key}_{run_idx}"):
                            records.pop(run_idx)
                            save_memory_to_disk(st.session_state.db)
                            st.success("Dataset purged successfully!")
                            st.rerun()

                    st.markdown("#### 🔹 Pulldown Baseline Layer Matrix")
                    p_df = pd.DataFrame([record['pulldown_data']])
                    st.dataframe(p_df, use_container_width=True, hide_index=True)
                    
                    st.markdown("#### 🔹 Connected CPT Multi-Format Condition Flags")
                    cpt_rows = []
                    for flag_name, flag_metrics in record['cpt_data'].items():
                        for metric_name, sensor_values in flag_metrics.items():
                            row_entry = {"Test Flag": flag_name, "Data Criteria": metric_name}
                            row_entry.update(sensor_values)
                            cpt_rows.append(row_entry)
                    
                    if cpt_rows:
                        st.dataframe(pd.DataFrame(cpt_rows), use_container_width=True, hide_index=True)
                    else:
                        st.warning("No metric matrix entries found inside this specific record block.")
