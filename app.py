import json
import os
import re
import base64
import numpy as np
import pandas as pd
import requests
import openpyxl
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import copy

# Initialize session state variables
if "p_file_key" not in st.session_state:
    st.session_state.p_file_key = 0

if "cpt_file_key" not in st.session_state:
    st.session_state.cpt_file_key = 0

# Must be the absolute first Streamlit command in the script
st.set_page_config(page_title="Refrigerator Simulator Hub", layout="wide")

# =================================================================
# 1. GITHUB-BACKED PERSISTENCE LAYER
# =================================================================
# Local disk on Streamlit Cloud is wiped on every reboot/redeploy, so the
# database matrix is instead stored as a JSON file committed directly to
# your GitHub repo via the Contents API. Requires a [github] block in
# Streamlit secrets: token, repo ("user/repo"), and optionally branch.
REVIEWER_PASSWORD = "Admin@Cooling2026"

GITHUB_FILE_PATH = "simulator_storage.json"

def _github_config_ok():
    return "github" in st.secrets and "token" in st.secrets["github"] and "repo" in st.secrets["github"]

def _github_api_url():
    repo = st.secrets["github"]["repo"]
    return f"https://api.github.com/repos/{repo}/contents/{GITHUB_FILE_PATH}"

def _github_headers():
    token = st.secrets["github"]["token"]
    return {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

def load_memory_from_disk():
    """Reads the saved database matrix from the GitHub repo on startup."""
    if not _github_config_ok():
        st.error("⚠️ GitHub storage isn't configured. Add a [github] block (token, repo) to Streamlit secrets.")
        return {}
    try:
        branch = st.secrets["github"].get("branch", "main")
        resp = requests.get(_github_api_url(), headers=_github_headers(), params={"ref": branch}, timeout=15)
        if resp.status_code == 200:
            content = resp.json()
            decoded = base64.b64decode(content["content"]).decode("utf-8")
            return json.loads(decoded) if decoded.strip() else {}
        elif resp.status_code == 404:
            # No file yet — first run. It will be created on first save.
            return {}
        else:
            st.error(f"⚠️ GitHub load error ({resp.status_code}): {resp.text}")
            return {}
    except Exception as e:
        st.error(f"⚠️ Error loading backup database file from GitHub: {str(e)}")
        return {}

def save_memory_to_disk(db_matrix):
    """Commits the current database matrix to the GitHub repo instantly."""
    if not _github_config_ok():
        st.error("⚠️ GitHub storage isn't configured. Add a [github] block (token, repo) to Streamlit secrets.")
        return
    try:
        branch = st.secrets["github"].get("branch", "main")
        headers = _github_headers()

        # Need the current file's SHA to update it (GitHub requires this for existing files)
        get_resp = requests.get(_github_api_url(), headers=headers, params={"ref": branch}, timeout=15)
        sha = get_resp.json().get("sha") if get_resp.status_code == 200 else None

        content_str = json.dumps(db_matrix, indent=4)
        encoded_content = base64.b64encode(content_str.encode("utf-8")).decode("utf-8")

        payload = {
            "message": "Update simulator storage data",
            "content": encoded_content,
            "branch": branch,
        }
        if sha:
            payload["sha"] = sha

        put_resp = requests.put(_github_api_url(), headers=headers, json=payload, timeout=15)
        if put_resp.status_code not in (200, 201):
            st.error(f"⚠️ Failed to write backup data to GitHub ({put_resp.status_code}): {put_resp.text}")
    except Exception as e:
        st.error(f"⚠️ Failed to write backup data to GitHub: {str(e)}")

# Legacy compatibility wrapper in case it's explicitly called later in your file
def save_memory(data):
    save_memory_to_disk(data)

# =================================================================
# 2. STATE INITIALIZATION ON STARTUP
# =================================================================
# --- Initialize Session States ---
if "db" not in st.session_state:
    st.session_state.db = load_memory_from_disk()

if 'reviewer_logged_in' not in st.session_state:
    st.session_state.reviewer_logged_in = False

if "arr_form_id" not in st.session_state:
    st.session_state.arr_form_id = 0

if "model_form_id" not in st.session_state:
    st.session_state.model_form_id = 0

if "cabinet_form_id" not in st.session_state:
    st.session_state.cabinet_form_id = 0

if "sensor_form_id" not in st.session_state:
    st.session_state.sensor_form_id = 0

# Global configuration constants
tc_features = ['tf-1', 'tf-2', 'tf-3', 'tf-4', 'tf-5', 'tc-1', 'tc-2', 'tc-3', 'tvc', 'S2']
metric_types = ['mean', 'min', 'max', '(max+min)/2']
metric_labels = {'mean': 'Mean', 'min': 'Min', 'max': 'Max', '(max+min)/2': '(Max+Min)/2'}

# =================================================================
# 3. UTILITY HELPER FUNCTIONS
# =================================================================
def normalize_sensor_name(name):
    if not isinstance(name, str):
        return ""
    # Lowercase, strip whitespace/punctuation commonly found around labels
    # (e.g. "Sensor (°C)" / "Sensor:" / "Th. Sensor" -> "sensor" / "thsensor")
    cleaned = name.lower().strip()
    for ch in [" ", "-", "_", "(", ")", ":", ".", "°", "'", '"']:
        cleaned = cleaned.replace(ch, "")
    return cleaned

# Finds the Sensor reading in a parsed sheet_data dict. Tries an exact normalized
# match first ("sensor"), then falls back to any key containing "sensor" as a
# substring (covers labels like "Sensor(oC)" -> "sensoroc", "ThSensor", etc.)
# so real files aren't mistakenly treated as missing due to label wording.
def find_sensor_reading(sheet_data):
    if "sensor" in sheet_data:
        return sheet_data["sensor"]
    for lbl, val in sheet_data.items():
        if "sensor" in lbl:
            return val
    return None

# Returns every "sensor"-containing row in a parsed pulldown sheet_data dict, keyed by
# its original (pre-normalization) label when available — e.g. {"Sensor FC": -23.3,
# "Sensor PC": 5.0, "Defrost Sensor": -29.5}. Unlike find_sensor_reading (which only
# ever returns one value, for backward compatibility with the single-sensor Pulldown
# "Sensor (°C)" field), this collects all of them, for cross-matching against whatever
# sensor-named columns a CPT file has.
def sensor_names_in_cpt_data(cpt_data_dict):
    """Every distinct sensor name found within one record's cpt_data (e.g.
    ["Sensor FC", "Sensor PC"]), in first-seen order. Empty for records saved
    before the dynamic multi-sensor structure existed (legacy Sensor/SensorMax only)."""
    names = []
    seen = set()
    for flag, block in cpt_data_dict.items():
        sensors_dict = block.get("sensors") if isinstance(block, dict) else None
        if sensors_dict:
            for name in sensors_dict.keys():
                if name not in seen:
                    seen.add(name)
                    names.append(name)
    return names

def cpt_has_sensor_data(cpt_structured):
    """True if any level in a parsed CPT has a real (non-zero) Sensor reading, either in
    the dynamic per-sensor structure or the legacy Sensor/SensorMax fields. Exactly 0.0
    everywhere is how a missing column parses, so it is treated as "not found"."""
    for block in cpt_structured.values():
        if not isinstance(block, dict):
            continue
        if block.get("Sensor") or block.get("SensorMax"):
            return True
        for sdata in (block.get("sensors") or {}).values():
            if sdata.get("min") or sdata.get("max"):
                return True
    return False

def cpt_has_s2_data(cpt_structured):
    """True if any level in a parsed CPT has a real (non-zero) S2 reading."""
    return any(isinstance(b, dict) and b.get("S2") for b in cpt_structured.values())

def discover_sensor_names(volume_records):
    """Returns every distinct sensor name found in historical CPT training data for
    this Volume+Arrangement, across all records. Falls back to a single generic
    "Sensor" entry if no record has the dynamic multi-sensor structure."""
    names = []
    seen = set()
    for record in volume_records:
        for name in sensor_names_in_cpt_data(record.get("cpt_data", {})):
            if name not in seen:
                seen.add(name)
                names.append(name)
    return names if names else ["Sensor"]

def find_all_sensor_readings(sheet_data, sheet_labels=None):
    sheet_labels = sheet_labels or {}
    found = {}
    for norm_key, val in sheet_data.items():
        if "sensor" in norm_key:
            label = sheet_labels.get(norm_key, norm_key)
            found[label] = val
    return found

# Scans an openpyxl worksheet for a cell whose text matches one of label_variants
# (case-insensitive, whitespace-trimmed — e.g. "Entry Code" / "entry code" / "ENTRY CODE"
# all match), then looks nearby for a value starting with value_prefix — first scanning
# a few cells to the right in the same row, then a few rows below in the same column.
# The small scan window (rather than only the single adjacent cell) makes this tolerant
# of report layouts that leave a blank gap cell between a label and its value.
def find_labeled_value(ws, label_variants, value_prefix, max_right=6, max_down=4, max_scan_rows=60, max_scan_cols=30):
    targets = [v.strip().lower() for v in label_variants]
    prefix = value_prefix.strip().upper()

    for r in range(1, max_scan_rows + 1):
        for c in range(1, max_scan_cols + 1):
            cell_val = ws.cell(r, c).value
            if cell_val is None or not isinstance(cell_val, str):
                continue
            if cell_val.strip().lower() not in targets:
                continue

            # Look right, in the same row, within a small window
            for dc in range(1, max_right + 1):
                right_val = ws.cell(r, c + dc).value
                if right_val is not None and str(right_val).strip().upper().startswith(prefix):
                    return str(right_val).strip()

            # Look below, in the same column, within a small window
            for dr in range(1, max_down + 1):
                below_val = ws.cell(r + dr, c).value
                if below_val is not None and str(below_val).strip().upper().startswith(prefix):
                    return str(below_val).strip()

    return None

# Finds a token starting with `prefix` inside a piece of free text (typically a filename
# like "PCT-2_Report_307L_F-12043_R029809.xlsx"), splitting on common separators.
# Strips a trailing file extension if the matched token happens to carry one.
def find_prefixed_token_in_text(text, prefix):
    if not text:
        return None
    prefix_upper = prefix.strip().upper()
    tokens = re.split(r'[_\s]+', text)
    for tok in tokens:
        tok_clean = tok.strip()
        if tok_clean.upper().startswith(prefix_upper):
            tok_clean = re.sub(r'\.(xlsx|xlsm|xls)$', '', tok_clean, flags=re.IGNORECASE)
            return tok_clean
    return None

# Fallback: scans every cell of a worksheet for any value starting with `prefix`,
# with no label required nearby. Used when a value isn't findable via the filename.
def find_prefixed_cell_value(ws, prefix, max_scan_rows=80, max_scan_cols=30):
    prefix_upper = prefix.strip().upper()
    for r in range(1, max_scan_rows + 1):
        for c in range(1, max_scan_cols + 1):
            v = ws.cell(r, c).value
            if v is not None and str(v).strip().upper().startswith(prefix_upper):
                return str(v).strip()
    return None

# Combined extractor for a Pulldown file: tries the filename first (most reliable for
# this report style), then falls back to scanning the first sheet's cells.
def extract_pulldown_entry_test_id(uploaded_file):
    entry_code = find_prefixed_token_in_text(getattr(uploaded_file, "name", None), "F-")
    test_id = find_prefixed_token_in_text(getattr(uploaded_file, "name", None), "R0")
    if entry_code is None or test_id is None:
        try:
            uploaded_file.seek(0)
            _wb = openpyxl.load_workbook(uploaded_file, data_only=True)
            _ws = _wb[_wb.sheetnames[0]]
            if entry_code is None:
                entry_code = find_prefixed_cell_value(_ws, "F-")
            if test_id is None:
                test_id = find_prefixed_cell_value(_ws, "R0")
        except Exception:
            pass
        finally:
            uploaded_file.seek(0)
    return entry_code, test_id

# Scans an elaborated Pulldown report (channel names in column A, a header row with
# Avg/Min/Max plus any number of named checkpoint columns like "tfa: -6.0", "tca: 8.0",
# or "Hr: 0.5") and returns {checkpoint_name: {channel_key: value}}. Returns {} for
# simple 2-column or Summary-style files with no such header — those keep working
# through the existing Avg-only baseline path, unaffected.
#
# A checkpoint column is any header cell containing a colon, other than the bare
# "avg"/"min"/"max" labels — this makes the checkpoint set fully dynamic. Files can
# have more, fewer, or differently-named checkpoints (e.g. a future "tvca: 5.0")
# with no code change needed here.
def extract_pulldown_checkpoints(uploaded_file):
    try:
        uploaded_file.seek(0)
        wb = openpyxl.load_workbook(uploaded_file, data_only=True)
        ws = wb[wb.sheetnames[0]]
    except Exception:
        return {}
    finally:
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

    # Find the header row: the one row containing both a literal "avg" and "min" cell
    header_row = None
    checkpoint_cols = {}
    for r in range(1, min(ws.max_row, 60) + 1):
        row_texts = {}
        for c in range(1, min(ws.max_column, 40) + 1):
            v = ws.cell(r, c).value
            if isinstance(v, str) and v.strip():
                row_texts[c] = v.strip()
        lowered = {c: t.lower() for c, t in row_texts.items()}
        if any(t == "avg" for t in lowered.values()) and any(t == "min" for t in lowered.values()):
            header_row = r
            for c, t in row_texts.items():
                if ":" in t:
                    checkpoint_cols[c] = t
            break

    if header_row is None or not checkpoint_cols:
        return {}

    # Collect each channel's value under every checkpoint column, using the same
    # channel-name normalization the rest of the app already uses
    raw_checkpoints = {}
    for r in range(header_row + 1, min(ws.max_row, header_row + 60) + 1):
        chan_raw = ws.cell(r, 1).value
        if chan_raw is None or not isinstance(chan_raw, str) or not chan_raw.strip():
            continue
        chan_norm = normalize_sensor_name(chan_raw)
        for col_idx, cp_name in checkpoint_cols.items():
            val = ws.cell(r, col_idx).value
            if val is None:
                continue
            try:
                val_f = round(float(val), 1)
            except (ValueError, TypeError):
                continue
            raw_checkpoints.setdefault(cp_name, {})[chan_norm] = val_f

    # Translate normalized channel names (tf1, tc1, tvc1...) into the app's standard
    # keys (tf-1, tc-1, tvc), same mapping used for the baseline Avg extraction
    mapping_keys = {
        'tf-1': 'tf1', 'tf-2': 'tf2', 'tf-3': 'tf3', 'tf-4': 'tf4', 'tf-5': 'tf5',
        'tc-1': 'tc1', 'tc-2': 'tc2', 'tc-3': 'tc3', 'S2': 's2'
    }
    result = {}
    for cp_name, chan_vals in raw_checkpoints.items():
        entry = {}
        for target_key, norm_label in mapping_keys.items():
            if norm_label in chan_vals:
                entry[target_key] = chan_vals[norm_label]
        tvc_vals = [chan_vals[l] for l in ['tvc1', 'tvc2', 'tvc3'] if l in chan_vals]
        if tvc_vals:
            entry['tvc'] = round(sum(tvc_vals) / len(tvc_vals), 1)
        if entry:  # only keep checkpoints that yielded at least one recognized channel
            result[cp_name] = entry
    return result

def to_float(v):
    try:
        if pd.isna(v) or str(v).strip() == "":
            return 0.0
        return round(float(v), 1)
    except:
        return 0.0

# Rounds every numeric column in a dataframe to 1 decimal place for display/export/save
def round_df(df, decimals=1):
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        df[numeric_cols] = df[numeric_cols].round(decimals)
    return df

# Builds a column_config dict: numeric columns get 1-decimal NumberColumns, text_cols
# get TextColumns, and any column named in disabled_cols is locked from editing.
def build_column_config(df, disabled_cols=None, text_cols=None):
    disabled_cols = set(disabled_cols or [])
    text_cols = set(text_cols or [])
    config = {}
    for col in df.columns:
        if col in text_cols:
            config[col] = st.column_config.TextColumn(col, disabled=(col in disabled_cols))
        else:
            config[col] = st.column_config.NumberColumn(col, format="%.1f", disabled=(col in disabled_cols))
    return config

# Wraps st.data_editor with a snapshot-based Undo/Redo stack. NOTE: this is whole-table
# undo/redo (each button press reverts/reapplies the entire table to a prior snapshot),
# not per-cell — Streamlit's data_editor has no API for tracking individual cell edits,
# so there's no way to undo "just the last cell" from the Python side.
# base_key must be unique per editor instance (e.g. tied to record + edit-session version).
def undoable_data_editor(base_key, initial_df, column_config, key_suffix=""):
    undo_key = f"{base_key}_undo_stack"
    redo_key = f"{base_key}_redo_stack"
    base_df_key = f"{base_key}_base_df"
    ver_key = f"{base_key}_ver"

    if undo_key not in st.session_state:
        st.session_state[undo_key] = []
    if redo_key not in st.session_state:
        st.session_state[redo_key] = []
    if base_df_key not in st.session_state:
        st.session_state[base_df_key] = initial_df.copy()
    if ver_key not in st.session_state:
        st.session_state[ver_key] = 0

    editor_widget_key = f"{base_key}_editor{key_suffix}_v{st.session_state[ver_key]}"

    edited_raw = st.data_editor(
        st.session_state[base_df_key],
        use_container_width=True,
        hide_index=True,
        num_rows="fixed",
        column_config=column_config,
        key=editor_widget_key,
    )
    edited = round_df(add_avg_columns(edited_raw.drop(columns=avg_column_names(edited_raw), errors="ignore")))

    # A real edit happened this rerun (not an undo/redo click) — snapshot the prior state
    if not edited.equals(st.session_state[base_df_key]):
        st.session_state[undo_key].append(st.session_state[base_df_key].copy())
        st.session_state[base_df_key] = edited.copy()
        st.session_state[redo_key] = []  # new edit invalidates any redo history

    undo_col, redo_col, _spacer = st.columns([1, 1, 6])
    with undo_col:
        if st.button("↶ Undo", key=f"{base_key}_undo_btn", disabled=(len(st.session_state[undo_key]) == 0)):
            st.session_state[redo_key].append(st.session_state[base_df_key].copy())
            st.session_state[base_df_key] = st.session_state[undo_key].pop()
            st.session_state[ver_key] += 1
            st.rerun()
    with redo_col:
        if st.button("↷ Redo", key=f"{base_key}_redo_btn", disabled=(len(st.session_state[redo_key]) == 0)):
            st.session_state[undo_key].append(st.session_state[base_df_key].copy())
            st.session_state[base_df_key] = st.session_state[redo_key].pop()
            st.session_state[ver_key] += 1
            st.rerun()

    return st.session_state[base_df_key]

# Clears all Undo/Redo state for a given editor instance — call this whenever leaving
# edit mode (Cancel or Save) so the next edit session starts with empty undo/redo stacks.
def reset_undo_state(base_key):
    for suffix in ("_undo_stack", "_redo_stack", "_base_df", "_ver"):
        st.session_state.pop(f"{base_key}{suffix}", None)

# Blanks the Test Flag on continuation rows of a group (rows 2..N of each level),
# so the editable grid visually reads like the Test Flag cell is "merged" downward.
def blank_repeated_flag(df):
    df = df.copy()
    prev = object()  # sentinel that can't equal any real flag value
    for idx in df.index:
        cur = df.at[idx, "Test Flag"]
        if cur == prev:
            df.at[idx, "Test Flag"] = ""
        else:
            prev = cur
    return df

# Reverses blank_repeated_flag: fills blanked Test Flag cells with the nearest
# non-blank value above them, so downstream save/comparison logic sees full values.
def forward_fill_flag(df):
    df = df.copy()
    last = None
    filled = []
    for v in df["Test Flag"]:
        if v is None or (isinstance(v, str) and v.strip() == ""):
            filled.append(last)
        else:
            last = v
            filled.append(v)
    df["Test Flag"] = filled
    return df

# Generic merged-cell (rowspan) HTML table renderer. Groups consecutive rows that
# share the same value in `group_col`, and merges any column named in `merge_cols`
# (in addition to group_col itself) into one spanning cell per group, showing the
# first non-null value found in that group. Used for read-only display only —
# st.data_editor can't render merged cells, but a plain HTML table can.
def render_merged_table(df, group_col, merge_cols=None):
    if df.empty:
        return "<p><em>No data</em></p>"

    merge_cols = set(merge_cols or [])
    merge_cols.add(group_col)

    df = df.reset_index(drop=True)
    cols = list(df.columns)
    border = "border:1px solid rgba(120,120,120,0.5);"

    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, (int, float, np.floating, np.integer)):
            return f"{float(v):.1f}"
        return str(v)

    html = ["<div style='overflow-x:auto;'>", "<table style='width:100%; border-collapse:collapse; font-size:14px;'>", "<thead><tr>"]
    for c in cols:
        html.append(f"<th style='{border} padding:6px 10px; text-align:center; font-weight:600;'>{c}</th>")
    html.append("</tr></thead><tbody>")

    n = len(df)
    i = 0
    while i < n:
        group_val = df.loc[i, group_col]
        j = i
        while j < n and df.loc[j, group_col] == group_val:
            j += 1
        group_size = j - i

        # For each merged column, find the one row in this group that actually carries a value
        merged_vals = {}
        for mcol in merge_cols:
            if mcol == group_col or mcol not in df.columns:
                continue
            val = None
            for k in range(i, j):
                if pd.notna(df.loc[k, mcol]):
                    val = df.loc[k, mcol]
            merged_vals[mcol] = val

        for local_idx, row_idx in enumerate(range(i, j)):
            html.append("<tr>")
            for c in cols:
                if c == group_col:
                    if local_idx == 0:
                        html.append(f"<td rowspan='{group_size}' style='{border} padding:6px 10px; text-align:center; vertical-align:middle; font-weight:600;'>{fmt(group_val)}</td>")
                elif c in merge_cols:
                    if local_idx == 0:
                        html.append(f"<td rowspan='{group_size}' style='{border} padding:6px 10px; text-align:center; vertical-align:middle;'>{fmt(merged_vals.get(c))}</td>")
                else:
                    html.append(f"<td style='{border} padding:6px 10px; text-align:center;'>{fmt(df.loc[row_idx, c])}</td>")
            html.append("</tr>")
        i = j

    html.append("</tbody></table></div>")
    return "".join(html)

# CPT tables: merges Test Flag, S2 (Mean-only), and Sensor (Min-only) across each level's group
def render_merged_cpt_table(df):
    return render_merged_table(df, group_col="Test Flag", merge_cols=["S2"])

# Predictions table: merges Sensor Value and S2 (Mean-only) across each query point's group of 4 metric rows
def render_merged_predictions_table(df):
    return render_merged_table(df, group_col="Sensor Value", merge_cols=["S2"])

# Renders a dataframe as a plain styled HTML table (same visual style as
# render_merged_cpt_table) with no merged cells — used for tables with no
# grouping to merge across, like the single-row Pulldown Matrix.
# highlight_missing_cols: column names whose cell gets a red-tinted background
# when the value is missing (None/NaN), to flag data gaps like an absent Sensor reading.
def render_simple_html_table(df, highlight_missing_cols=None):
    if df.empty:
        return "<p><em>No data</em></p>"

    highlight_missing_cols = set(highlight_missing_cols or [])
    df = df.reset_index(drop=True)
    cols = list(df.columns)
    border = "border:1px solid rgba(120,120,120,0.5);"
    highlight_style = "border:2px solid #ff4b4b; background-color:rgba(255,75,75,0.12);"

    def fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return ""
        if isinstance(v, (int, float, np.floating, np.integer)):
            return f"{float(v):.1f}"
        return str(v)

    def is_missing(v):
        return v is None or (isinstance(v, float) and pd.isna(v))

    html = ["<div style='overflow-x:auto;'>", "<table style='width:100%; border-collapse:collapse; font-size:14px;'>", "<thead><tr>"]
    for c in cols:
        html.append(f"<th style='{border} padding:6px 10px; text-align:center; font-weight:600;'>{c}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        html.append("<tr>")
        for c in cols:
            if c in highlight_missing_cols and is_missing(row[c]):
                cell_style = f"{highlight_style} padding:6px 10px; text-align:center;"
            else:
                cell_style = f"{border} padding:6px 10px; text-align:center;"
            html.append(f"<td style='{cell_style}'>{fmt(row[c])}</td>")
        html.append("</tr>")

    html.append("</tbody></table></div>")
    return "".join(html)

# Builds a single-row Pulldown dataframe with a fixed, consistent column order
# (tf-1..5, tc-1..3, tvc, S2, Sensor), regardless of the underlying dict's key order.
# sensor_value may be None if no Sensor reading was ever found for this record.
def build_pulldown_df(pulldown_data_dict, sensor_value):
    # Group every "{prefix}-{number}" channel key present, in first-seen prefix order
    # and numeric order within each group — works for any cabinet, any count, not
    # just the original fixed tf-1..5/tc-1..3.
    groups = {}
    prefix_order = []
    for k in pulldown_data_dict.keys():
        m = re.match(r'^(.+)-(\d+)$', str(k))
        if m:
            prefix = m.group(1)
            if prefix not in groups:
                groups[prefix] = []
                prefix_order.append(prefix)
            groups[prefix].append((int(m.group(2)), k))

    row = {}
    for prefix in prefix_order:
        for _, k in sorted(groups[prefix]):
            row[k] = pulldown_data_dict.get(k, 0.0)

    # Legacy fallback: an old record's bare cabinet value (e.g. "tvc") that was never
    # split into individual numbered channels — show it as-is rather than losing it
    for k, v in pulldown_data_dict.items():
        if k == "S2" or re.match(r'^.+-(\d+)$', str(k)) or str(k).endswith('-a'):
            continue
        row[k] = v

    row["S2"] = pulldown_data_dict.get("S2", 0.0)
    row["Sensor"] = sensor_value if sensor_value is not None else np.nan
    return pd.DataFrame([row])

# Computes tf-a (avg of tf-1..tf-5) and tc-a (avg of tc-1..tc-3) for a dict of sensor values
def compute_avg_fields(values_dict):
    tf_vals = [to_float(values_dict.get(f"tf-{i}", 0.0)) for i in range(1, 6)]
    tc_vals = [to_float(values_dict.get(f"tc-{i}", 0.0)) for i in range(1, 4)]
    tf_a = round(sum(tf_vals) / len(tf_vals), 1) if tf_vals else 0.0
    tc_a = round(sum(tc_vals) / len(tc_vals), 1) if tc_vals else 0.0
    return tf_a, tc_a

# Adds a "{prefix}-a" average column for every group of "{prefix}-{number}" columns
# found in the dataframe (e.g. tf-1..tf-6 -> tf-a, tc-1..tc-3 -> tc-a, tvc-1..tvc-3 ->
# tvc-a) — fully generic, so this keeps working unchanged for the old fixed CPT/
# prediction dataframes (which only ever have tf-1..5/tc-1..3) AND for the new
# dynamic Pulldown dataframes (any cabinet, any channel count), with no separate
# code path needed for either. Each average is positioned right after its group's
# highest-numbered member.
def add_avg_columns(df):
    df = df.copy()
    groups = {}
    for c in df.columns:
        m = re.match(r'^(.+)-(\d+)$', str(c))
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), c))

    last_member_of = {}
    for prefix, numbered_cols in groups.items():
        numbered_cols.sort(key=lambda t: t[0])  # sort by numeric suffix, not column order
        cols = [c for _, c in numbered_cols]
        avg_col = f"{prefix}-a"
        if avg_col not in df.columns:
            df[avg_col] = df[cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).round(1)
            last_member_of[avg_col] = cols[-1]

    # Reorder so each "{prefix}-a" sits right after its group's last numbered member
    cols_order = list(df.columns)
    for avg_col, after_col in last_member_of.items():
        cols_order.remove(avg_col)
        cols_order.insert(cols_order.index(after_col) + 1, avg_col)
    return df[cols_order]

# Returns the names of every computed average column (e.g. "tf-a", "tvc-a") present
# in a dataframe — used instead of a hardcoded ["tf-a", "tc-a"] list so this keeps
# working for any cabinet's average column, not just the original two.
def avg_column_names(df):
    return [c for c in df.columns if re.match(r'^.+-a$', str(c))]

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
# 3B. CABINET & SENSOR CONFIGURATION (per Volume Model + Arrangement)
# =================================================================
# Built-in cabinets keep their established, historical key prefixes (tf/tc were already
# used everywhere before this system existed) — CC and VC happen to already follow the
# "t" + lowercase-name pattern, which custom cabinets also use going forward.
BUILTIN_CABINET_PREFIXES = {"FC": "tf", "PC": "tc", "CC": "tcc", "VC": "tvc"}

def derive_cabinet_prefix(name):
    key = name.strip().upper()
    if key in BUILTIN_CABINET_PREFIXES:
        return BUILTIN_CABINET_PREFIXES[key]
    clean = normalize_sensor_name(name)
    return f"t{clean}" if clean else "tx"

def default_cabinet_sensor_config():
    return {
        "cabinets": [
            {"name": "FC", "prefix": "tf", "count": 5},
            {"name": "CC", "prefix": "tcc", "count": 3},
            {"name": "PC", "prefix": "tc", "count": 3},
            {"name": "VC", "prefix": "tvc", "count": 3},
        ],
        "sensors": ["Sensor-1"],
        # Which sensors control which CABINET (keyed by cabinet name, e.g. "FC"); a sensor
        # assigned to a cabinet controls every thermocouple in it. A cabinet absent from
        # this map, or mapped to an empty list, is treated as controlled by every sensor —
        # same rule as S2, which never appears here at all.
        "cabinet_sensor_map": {},
    }

def get_cabinet_sensor_config(vol, arr_name):
    """Returns (and lazily initializes) the cabinet/sensor config for a Volume+Arrangement."""
    if vol not in st.session_state.db or not isinstance(st.session_state.db[vol], dict):
        st.session_state.db[vol] = {}
    if arr_name not in st.session_state.db[vol] or not isinstance(st.session_state.db[vol][arr_name], dict):
        st.session_state.db[vol][arr_name] = {}
    if "_config" not in st.session_state.db[vol][arr_name] or not isinstance(st.session_state.db[vol][arr_name]["_config"], dict):
        st.session_state.db[vol][arr_name]["_config"] = default_cabinet_sensor_config()
    cfg = st.session_state.db[vol][arr_name]["_config"]
    # Backfill any missing keys for configs saved by an older version of this app
    cfg.setdefault("cabinets", default_cabinet_sensor_config()["cabinets"])
    cfg.setdefault("sensors", ["Sensor-1"])
    cfg.setdefault("cabinet_sensor_map", {})
    # Migrate the old per-thermocouple assignments (channel_sensor_map) to per-cabinet:
    # a cabinet keeps its selection only if every thermocouple in it that had one agreed
    # on the same sensors. Mixed selections can't be represented per cabinet, so those
    # cabinets go back to "all sensors". The old key is then dropped.
    old_map = cfg.pop("channel_sensor_map", None)
    if old_map:
        for cab in cfg["cabinets"]:
            chan_sets = [
                frozenset(old_map[f"{cab['prefix']}-{i}"])
                for i in range(1, int(cab.get("count", 0)) + 1)
                if old_map.get(f"{cab['prefix']}-{i}")
            ]
            if chan_sets and len(set(chan_sets)) == 1 and cab["name"] not in cfg["cabinet_sensor_map"]:
                cfg["cabinet_sensor_map"][cab["name"]] = sorted(chan_sets[0])
    # The per-cabinet on/off tick was removed: every listed cabinet is always active.
    # Drop any stale "enabled" flag saved by an older version, so a cabinet that was
    # once unticked doesn't stay silently inactive with no way to turn it back on.
    for cab in cfg["cabinets"]:
        cab.pop("enabled", None)
    return cfg

def all_channel_keys(cabinet_config):
    """Flat list of every channel key (e.g. 'tf-1', 'tf-2', ...) across all cabinets,
    across every listed cabinet — used to build the sensor-assignment checklist."""
    keys = []
    for cab in cabinet_config["cabinets"]:
        for i in range(1, int(cab.get("count", 0)) + 1):
            keys.append(f"{cab['prefix']}-{i}")
    return keys

# Extracts every cabinet's channel values from a parsed {normalized_label: value}
# pulldown dict, driven entirely by Cabinet Configuration (not a fixed field list).
# For each cabinet (prefix P, count N):
#   1. Fill slots P-1..P-N using exact numeric matches ("p1", "p2", ...) in sheet_data.
#   2. Any slots still unfilled are filled, in file order, from OTHER sheet_data keys
#      that start with prefix P but weren't already claimed — this is what lets a
#      non-numeric channel name like "tf Box" (normalized "tfbox") count as an extra
#      FC thermocouple when the configured count exceeds the numbered ones found.
# sheet_labels (optional) maps normalized_label -> original file text (e.g.
# "tfbox" -> "tf Box"), used to report which physical channel a fallback slot came
# from — without this, assigning two extras (e.g. "tf Box" and "tf Pocket") to tf-6
# and tf-7 would leave no way to tell which number means which sensor.
# Returns (channel_values, notifications, slot_labels):
#   channel_values — {"tf-1": ..., ...} for every matched slot across every cabinet
#   notifications  — human-readable warnings for any partial/total match failure
#   slot_labels    — {"tf-6": "tf Box", ...} original text for fallback-matched slots
#                     only (numeric matches like tf-1..tf-5 are unambiguous already)
def extract_cabinet_channels(sheet_data, cabinet_config, sheet_labels=None):
    sheet_labels = sheet_labels or {}
    channel_values = {}
    notifications = []
    slot_labels = {}
    sheet_keys_in_order = list(sheet_data.keys())  # dict preserves file row order

    active_cabs = [c for c in cabinet_config["cabinets"] if int(c.get("count", 0)) > 0]

    # Claimed keys are GLOBAL across all cabinets in this call, so the same file key
    # (e.g. "tcc1") can never be double-counted toward two different cabinets.
    claimed_keys = set()
    # Results keyed by cabinet name, filled in a first pass ordered by descending
    # prefix length — so a longer, more specific prefix (e.g. "tcc" for Chiller
    # Cabinet) claims its matches before a shorter one (e.g. "tc" for PC) gets a
    # chance to mistake "tcc1" for one of its own leftover candidates.
    results_by_cab = {}
    for cab in sorted(active_cabs, key=lambda c: -len(c["prefix"])):
        prefix = cab["prefix"]
        count = int(cab.get("count", 0))

        filled = {}
        filled_from_fallback = {}

        # Step 1: exact numeric matches (tf1, tf2, ...)
        for i in range(1, count + 1):
            numeric_key = f"{prefix}{i}"
            if numeric_key in sheet_data and numeric_key not in claimed_keys:
                filled[i] = sheet_data[numeric_key]
                claimed_keys.add(numeric_key)

        # Step 2: fill remaining slots from other prefix-matching, unclaimed keys,
        # in the order they appear in the file (e.g. "tf box" -> "tfbox"). A
        # candidate is skipped if it starts with another cabinet's LONGER
        # prefix (e.g. "tcc1" belongs to "tcc", not the shorter "tc"), on top of
        # the global claimed_keys check.
        other_longer_prefixes = [c["prefix"] for c in active_cabs if c["prefix"] != prefix and len(c["prefix"]) > len(prefix)]
        missing_slots = [i for i in range(1, count + 1) if i not in filled]
        if missing_slots:
            leftover_candidates = [
                k for k in sheet_keys_in_order
                if k.startswith(prefix)
                and k not in claimed_keys
                and not any(k.startswith(p) for p in other_longer_prefixes)
            ]
            for slot, cand_key in zip(missing_slots, leftover_candidates):
                filled[slot] = sheet_data[cand_key]
                filled_from_fallback[slot] = cand_key
                claimed_keys.add(cand_key)

        results_by_cab[cab["name"]] = (filled, filled_from_fallback)

    # Second pass, in the cabinet's original configured order, just for readable output
    for cab in cabinet_config["cabinets"]:
        if cab["name"] not in results_by_cab:
            continue
        prefix = cab["prefix"]
        count = int(cab.get("count", 0))
        filled, filled_from_fallback = results_by_cab[cab["name"]]

        matched_count = len(filled)
        fallback_descriptions = []
        for i in range(1, count + 1):
            if i in filled:
                slot_key = f"{prefix}-{i}"
                channel_values[slot_key] = filled[i]
                if i in filled_from_fallback:
                    original_text = sheet_labels.get(filled_from_fallback[i], filled_from_fallback[i])
                    slot_labels[slot_key] = original_text
                    fallback_descriptions.append(f"{slot_key} = \"{original_text}\"")

        if matched_count == 0:
            notifications.append(f"⚠️ No thermocouples found for cabinet '{cab['name']}' — this cabinet's data was not collected.")
        else:
            if matched_count < count:
                notifications.append(f"⚠️ Only {matched_count} of {count} thermocouples found for cabinet '{cab['name']}' — using the {matched_count} found.")
            if fallback_descriptions:
                notifications.append(f"ℹ️ Cabinet '{cab['name']}' extra channel mapping: {', '.join(fallback_descriptions)}")

    return channel_values, notifications, slot_labels

# -----------------------------------------------------------------
# CPT-side channel detection (new report format).
# Reads the two header rows of the CPT table: the upper row holds cabinet titles
# ("Freezer Cabinet", "Refrigerator Cabinet", ...) and the lower row holds the
# thermocouple names ("tf1", "tf Box", "tvc1", "Avg. C", ...). Every named
# thermocouple column is returned, so the CPT file decides what channels exist —
# nothing is fixed to 5/3/1. "Avg." columns are skipped (averages are always
# computed by this app). Returns [(column, original_label, cabinet_title), ...].
# -----------------------------------------------------------------
def detect_cpt_channel_columns(ws, header_row, first_col, last_col):
    out = []
    current_title = ""
    for c in range(first_col, last_col + 1):
        top = ws.cell(header_row, c).value
        if isinstance(top, str) and top.strip():
            current_title = top.strip()
        label = ws.cell(header_row + 1, c).value
        if not isinstance(label, str) or not label.strip():
            continue
        if label.strip().lower().startswith("avg"):
            continue
        out.append((c, label.strip(), current_title))
    return out

# Builds {slot_key: value} for one CPT data row (one Mean/Min/Max/(Max+Min)/2 row),
# using the same Cabinet Configuration matching as the Pulldown file (so "tf Box"
# becomes the next free FC slot, exactly like in the Pulldown). Blank cells are skipped.
def extract_cpt_row_channels(ws, data_row, channel_cols, cabinet_config):
    row_data, row_labels = {}, {}
    for col, label, _title in channel_cols:
        v = ws.cell(data_row, col).value
        try:
            if isinstance(v, str):
                v = v.replace("°C", "").replace("̊C", "").strip()
            fv = float(v)
        except (ValueError, TypeError):
            continue
        if fv != fv:  # NaN
            continue
        key = normalize_sensor_name(label)
        if key and key not in row_data:
            row_data[key] = round(fv, 1)
            row_labels[key] = label
    return extract_cabinet_channels(row_data, cabinet_config, row_labels)

# Compares the thermocouple slots found in the CPT file with those found in the
# Pulldown file. Returns (matched_keys, notes). Only a thermocouple present in BOTH
# files can be used together later; everything else is reported in plain English.
def match_pulldown_cpt_channels(pulldown_channels, cpt_channels, cpt_slot_labels=None):
    cpt_slot_labels = cpt_slot_labels or {}
    p_keys = [k for k in pulldown_channels if k != "S2"]
    c_keys = list(cpt_channels.keys())
    matched = [k for k in c_keys if k in set(p_keys)]
    notes = []

    def nice(k):
        return f"{k} ({cpt_slot_labels[k]})" if k in cpt_slot_labels else k

    only_cpt = [nice(k) for k in c_keys if k not in set(p_keys)]
    only_pull = [k for k in p_keys if k not in set(c_keys)]
    if not matched and (p_keys or c_keys):
        notes.append("⚠️ No thermocouple is common to the Pulldown and CPT files — nothing can be matched.")
    else:
        if only_cpt:
            notes.append(f"⚠️ In the CPT file only (not in Pulldown, so not matched): {', '.join(only_cpt)}")
        if only_pull:
            notes.append(f"⚠️ In the Pulldown file only (not in CPT, so not matched): {', '.join(only_pull)}")
        if matched and not only_cpt and not only_pull:
            notes.append(f"✅ All {len(matched)} thermocouples match between Pulldown and CPT.")
    return matched, notes

# Computes each cabinet's own average (e.g. "tf-a", "tc-a", "tvc-a") from
# whichever channel_values slots actually got filled — never a fixed 5/3 count.
# Returns {"tf-a": ..., ...} for every cabinet that had at least one matched channel.
def compute_cabinet_averages(channel_values, cabinet_config):
    averages = {}
    for cab in cabinet_config["cabinets"]:
        prefix = cab["prefix"]
        count = int(cab.get("count", 0))
        vals = [channel_values[f"{prefix}-{i}"] for i in range(1, count + 1) if f"{prefix}-{i}" in channel_values]
        if vals:
            averages[f"{prefix}-a"] = round(sum(vals) / len(vals), 1)
    return averages


# =================================================================
def run_automated_simulation(volume_records, new_pulldown, pulldown_sensor, target_sensor_pairs, pulldown_feature_keys,
                             sensor_names=None, target_sensor_points=None, cabinet_config=None):
    """
    Builds the consolidated prediction table. Every query point produces 4 rows —
    Mean, Min, Max, (Max+Min)/2.

    How it predicts:
      * Every thermocouple (tf-1.., tc-1.., tvc-1.., and extras like tf-6 = "tf Box")
        gets its OWN nearest-neighbour model, trained only on the stored runs that
        actually have that thermocouple.
      * Each Sensor gets its own prediction run, using that sensor's Min/Max as the
        driving feature (Min row -> Min, Max row -> Max, Mean and (Max+Min)/2 -> the
        average of Min and Max).
      * The per-sensor predictions are then averaged. A cabinet with sensors chosen in
        the Cabinet & Sensor Configuration uses only those sensors; a cabinet with none
        chosen (and S2) uses every sensor.

    `sensor_names` + `target_sensor_points` ([{sensor: (min, max)}, ...] — one dict per
    query point) switch on the multi-sensor mode. Without them the old single-sensor
    call style (`target_sensor_pairs`) still works unchanged.
    `pulldown_feature_keys` is the ordered list of pulldown channel keys used as the
    context features for both the live query and every stored run; an older record
    that only has a bare "tvc" reuses it for every tvc-N slot.
    """
    if not volume_records:
        return pd.DataFrame()

    def clean_val(v):
        if v is None or pd.isna(v):
            return 0.0
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    def get_pulldown_feature_value(pulldown_data, key):
        if key in pulldown_data:
            return pulldown_data[key]
        if '-' in key:
            legacy_key = key.rsplit('-', 1)[0]
            if legacy_key in pulldown_data:
                return pulldown_data[legacy_key]
        return 0.0

    def fallback_sensor(flag, offset=0.0):
        if "1" in flag: base = -27.5
        elif "2" in flag: base = -27.0
        elif "3" in flag: base = -25.5
        elif "4" in flag: base = -24.0
        elif "5" in flag: base = -21.0
        else: base = 0.0
        return base + offset

    def feature_for_metric(metric_key, sensor_min, sensor_max):
        if metric_key == "min":
            return sensor_min
        elif metric_key == "max":
            return sensor_max
        return (sensor_min + sensor_max) / 2.0

    # ---- Which sensors / query points are we working with? ----
    if sensor_names and target_sensor_points:
        sensors = list(sensor_names)
        points = [{s: pt.get(s, (0.0, 0.0)) for s in sensors} for pt in target_sensor_points]
    else:
        sensors = ["__single__"]
        points = [{"__single__": pair} for pair in target_sensor_pairs]
    primary_sensor = sensors[0]

    def hist_sensor_pair(level_data, flag, sensor):
        """Min/Max of `sensor` for one stored level, or None if this run has none."""
        sdict = level_data.get("sensors") if isinstance(level_data, dict) else None
        if sensor == "__single__":
            pair = (level_data.get("Sensor", 0.0), level_data.get("SensorMax", 0.0))
        elif sdict:
            sd = sdict.get(sensor)
            if not sd:
                return None
            pair = (sd.get("min", 0.0), sd.get("max", 0.0))
        elif sensor == primary_sensor:      # old single-sensor record
            pair = (level_data.get("Sensor", 0.0), level_data.get("SensorMax", 0.0))
        else:
            return None
        mn, mx = clean_val(pair[0]), clean_val(pair[1])
        if mn == 0.0: mn = fallback_sensor(flag, 0.0)
        if mx == 0.0: mx = fallback_sensor(flag, 1.0)
        return mn, mx

    # ---- Which thermocouples will be predicted? ----
    fixed_fields = ["tf-1", "tf-2", "tf-3", "tf-4", "tf-5", "tc-1", "tc-2", "tc-3", "tvc"]
    extra_fields = set()
    for record in volume_records:
        for level_data in (record.get("cpt_data") or {}).values():
            if not isinstance(level_data, dict):
                continue
            for mk in metric_types:
                md = level_data.get(mk)
                if isinstance(md, dict):
                    for k in md:
                        if re.match(r'^.+-\d+$', str(k)) and k not in fixed_fields:
                            extra_fields.add(k)
    target_fields = list(fixed_fields) + sorted(extra_fields)
    # If individual tvc-1..N exist, the single "tvc" average column is redundant
    if any(k.startswith("tvc-") for k in extra_fields) and "tvc" in target_fields:
        target_fields.remove("tvc")

    prefix_rank = {}
    for i, p in enumerate((c["prefix"] for c in (cabinet_config or {}).get("cabinets", []))):
        prefix_rank[p] = i
    for p in ["tf", "tcc", "tc", "tvc"]:
        prefix_rank.setdefault(p, len(prefix_rank))

    def field_sort_key(k):
        m = re.match(r'^(.+)-(\d+)$', k)
        prefix, num = (m.group(1), int(m.group(2))) if m else (k, 0)
        return (prefix_rank.get(prefix, 999), prefix, num)
    target_fields.sort(key=field_sort_key)

    def sensors_for_field(field):
        """Sensors whose predictions are averaged for this thermocouple."""
        if field == "S2" or not cabinet_config:
            return None
        prefix = field.rsplit('-', 1)[0] if '-' in field else field
        for cab in cabinet_config.get("cabinets", []):
            if cab.get("prefix") == prefix:
                chosen = (cabinet_config.get("cabinet_sensor_map") or {}).get(cab["name"]) or []
                chosen = [s for s in chosen if s in sensors]
                return chosen or None
        return None

    def predict_one(metric_key, sensor, q_min, q_max):
        """{field: predicted value} for one sensor, or {} if no usable stored runs."""
        X, Y = [], []
        for record in volume_records:
            if not record.get('cpt_data'):
                continue
            p_features = [clean_val(get_pulldown_feature_value(record["pulldown_data"], f)) for f in pulldown_feature_keys]
            p_baseline = clean_val(record.get("pulldown_baseline_sensor", 0.0))
            for flag, level_data in record["cpt_data"].items():
                metric_data = level_data.get(metric_key) if isinstance(level_data, dict) else None
                if not metric_data:
                    continue
                pair = hist_sensor_pair(level_data, flag, sensor)
                if pair is None:
                    continue
                X.append(p_features + [p_baseline, feature_for_metric(metric_key, pair[0], pair[1])])
                targets = {f: clean_val(metric_data.get(f, 0.0)) for f in fixed_fields}
                targets.update({k: clean_val(v) for k, v in metric_data.items() if k in extra_fields})
                if metric_key == "mean":
                    targets["S2"] = clean_val(level_data.get("S2", 0.0))
                Y.append(targets)
        if not X:
            return {}
        X_arr = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0)
        q = np.nan_to_num(np.array(
            [clean_val(v) for v in new_pulldown] + [clean_val(pulldown_sensor), feature_for_metric(metric_key, q_min, q_max)],
            dtype=np.float64).reshape(1, -1), nan=0.0)
        fields = target_fields + (["S2"] if metric_key == "mean" else [])
        out = {}
        for field in fields:
            rows = [i for i, t in enumerate(Y) if field in t]
            if not rows:
                continue
            y = np.array([Y[i][field] for i in rows], dtype=np.float64)
            knn = KNeighborsRegressor(n_neighbors=min(3, len(rows)), weights='distance')
            knn.fit(X_arr[rows], y)
            out[field] = float(knn.predict(q)[0])
        return out

    predicted_rows = []
    for point in points:
        point = {s: (clean_val(pt[0]), clean_val(pt[1])) for s, pt in point.items()}
        if sensors == ["__single__"]:
            label = f"Min {point['__single__'][0]:.1f}°C / Max {point['__single__'][1]:.1f}°C"
        else:
            label = " | ".join(f"{s}: Min {mn:.1f}°C / Max {mx:.1f}°C" for s, (mn, mx) in point.items())

        for metric_key in metric_types:
            per_sensor = {}
            for s in sensors:
                res = predict_one(metric_key, s, point[s][0], point[s][1])
                if res:
                    per_sensor[s] = res
            if not per_sensor:
                continue

            row = {"Sensor Value": label, "Metric": metric_labels[metric_key]}
            for field in target_fields:
                use = sensors_for_field(field)
                vals = [per_sensor[s][field] for s in per_sensor
                        if (use is None or s in use) and field in per_sensor[s]]
                if not vals:  # chosen sensors have no data -> fall back to every sensor
                    vals = [per_sensor[s][field] for s in per_sensor if field in per_sensor[s]]
                if vals:
                    row[field] = round(sum(vals) / len(vals), 1)
            if metric_key == "mean":
                s2_vals = [per_sensor[s]["S2"] for s in per_sensor if "S2" in per_sensor[s]]
                row["S2"] = round(sum(s2_vals) / len(s2_vals), 1) if s2_vals else np.nan
            else:
                row["S2"] = np.nan
            predicted_rows.append(row)

    if predicted_rows:
        return pd.DataFrame(predicted_rows)
    return pd.DataFrame()


# =================================================================
# Checkpoint-refined predictions.
# Elaborated Pulldown reports contain named checkpoint columns ("Hr: 1.0",
# "tfa: -6.0", "tca: 8.0", ...) = the thermocouple readings at that moment of the
# pulldown. For every checkpoint that exists in BOTH the new Pulldown file and at
# least one stored run, this repeats the normal prediction using that checkpoint's
# readings (instead of the average readings) as the pulldown features, trained only
# on the stored runs that also have that checkpoint.
# "Support" = how many stored runs have the checkpoint; it decides the confidence
# label. Results come back sorted by support (then by number of thermocouples used).
# The normal (average-based) prediction is unchanged and stays the baseline.
# =================================================================
def run_checkpoint_refined_simulations(volume_records, new_checkpoints, pulldown_sensor, target_sensor_pairs,
                                       sensor_names=None, target_sensor_points=None, cabinet_config=None):
    results = []
    for cp_name, new_cp in (new_checkpoints or {}).items():
        supporters = [r for r in volume_records
                      if r.get("cpt_data") and isinstance(r.get("pulldown_checkpoints"), dict)
                      and cp_name in r["pulldown_checkpoints"]]
        if not supporters:
            continue
        common = [k for k in new_cp if all(k in r["pulldown_checkpoints"][cp_name] for r in supporters)]
        if not common:
            continue
        pseudo_records = [{
            "pulldown_data": {k: r["pulldown_checkpoints"][cp_name][k] for k in common},
            "pulldown_baseline_sensor": r.get("pulldown_baseline_sensor", 0.0),
            "cpt_data": r["cpt_data"],
        } for r in supporters]
        df = run_automated_simulation(
            pseudo_records, [new_cp[k] for k in common], pulldown_sensor, target_sensor_pairs, common,
            sensor_names=sensor_names, target_sensor_points=target_sensor_points, cabinet_config=cabinet_config)
        if df.empty:
            continue
        n = len(supporters)
        confidence = "High" if n >= 3 else ("Medium" if n == 2 else "Low")
        results.append({"name": cp_name, "support": n, "confidence": confidence, "channels": len(common), "df": df})
    results.sort(key=lambda r: (-r["support"], -r["channels"], r["name"]))
    return results


# =================================================================
# 5. STREAMLIT USER INTERFACE (TABS) & REPOSITORY LOGIC
# =================================================================

# ================= SIDEBAR: PROFILE & ARRANGEMENT MANAGER =================
st.sidebar.header("📁 Volume Profile Manager")

# Clean up empty strings or accidental "None" literal string keys if they exist in memory
if "None" in st.session_state.db:
    del st.session_state.db["None"]
if "" in st.session_state.db:
    del st.session_state.db[""]

existing_volumes = list(st.session_state.db.keys())
if existing_volumes:
    existing_volumes.sort(reverse=False)

# Render selectbox with actual available volumes
selected_volume = st.sidebar.selectbox(
    "Active Refrigerator Model:", 
    existing_volumes if existing_volumes else ["None"]
)

# --- Delete Model Option ---
if existing_volumes:
    st.sidebar.caption("⚠️ Permanently removes this model and all its arrangements")
    if st.sidebar.button("🗑️ Delete Selected Model"):
        # Allow deletion if there is at least one other valid model left
        if len(existing_volumes) > 1:
            if selected_volume in st.session_state.db:
                del st.session_state.db[selected_volume]
                save_memory(st.session_state.db)
                st.sidebar.success(f"Model '{selected_volume}' deleted successfully.")
                st.rerun()
        else:
            st.sidebar.error("❌ Cannot delete the last remaining model. Create another model before deleting this one.")

st.sidebar.markdown("---")

# 2. Select Arrangement of Selected Volume & Deletion
existing_arrangements = []
if selected_volume and selected_volume in st.session_state.db and isinstance(st.session_state.db[selected_volume], dict):
    existing_arrangements = list(st.session_state.db[selected_volume].keys())

if existing_arrangements:
    existing_arrangements.sort(reverse=False)
    selected_arrangement = st.sidebar.selectbox("Select Arrangement of Selected Volume:", existing_arrangements)
    
    # --- Delete Arrangement Option ---
    st.sidebar.caption("⚠️ Removes this arrangement data only")
    if st.sidebar.button("🗑️ Delete Selected Arrangement"):
        if len(existing_arrangements) > 1:
            del st.session_state.db[selected_volume][selected_arrangement]
            save_memory(st.session_state.db)
            st.sidebar.success(f"Arrangement '{selected_arrangement}' deleted.")
            st.rerun()
        else:
            st.sidebar.error("❌ Cannot delete the last remaining arrangement. Create another arrangement before deleting it.")
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
            st.session_state.arr_form_id += 1
            st.rerun()

st.sidebar.markdown("---")

# 4. ➕ Register New Volume Model Form
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
        sim_c_ambient = st.selectbox("Select Target Respected CPT Ambient:", ["16°C", "32°C", "43°C"], index=1, key="sim_c_amb")
        
    p_key = "32C" if "32" in sim_p_ambient else "43C"
    c_key = "16C" if "16" in sim_c_ambient else ("32C" if "32" in sim_c_ambient else "43C")
    
    verify_db_structure(selected_volume, selected_arrangement, p_key, c_key)
    vol_records = st.session_state.db[selected_volume][selected_arrangement][p_key][c_key]
    
    # Initialize version counters to force-refresh fields upon upload
    if "sim_ver" not in st.session_state:
        st.session_state.sim_ver = 0
    if 'active_pulldown_form' not in st.session_state:
        st.session_state.active_pulldown_form = {}
        st.session_state.last_uploaded_sim_file = None

    if not vol_records:
        st.warning(f"⚠️ No matching profiles found under arrangement **{selected_arrangement}** for **Pulldown: {sim_p_ambient}** linked to **CPT: {sim_c_ambient}**.")
    else:
        # ================= STEP 1: AUTOMATED SIMULATOR PARSER =================
        st.markdown("#### Step 1: Input Current Pulldown Telemetry Vector")
        sim_pulldown_file = st.file_uploader(
            f"Auto-fill fields from local Pulldown Report ({sim_p_ambient})", 
            type=["xlsx", "xls"], key=f"sim_file_upload_{p_key}_{c_key}"
        )

        if sim_pulldown_file and sim_pulldown_file != st.session_state.last_uploaded_sim_file:
            try:
                # Read first available sheet dynamically
                df_sim_p = pd.read_excel(sim_pulldown_file, sheet_name=0, header=None)
                raw_labels_sim = df_sim_p[0].astype(str)
                df_sim_p[0] = raw_labels_sim.apply(normalize_sensor_name)
                
                sheet_data = {}
                sheet_labels = {}  # normalized key -> original text (e.g. "tfbox" -> "tf Box")
                for idx, row in df_sim_p.dropna(subset=[0]).iterrows():
                    lbl = row[0]
                    if lbl not in sheet_data:
                        try:
                            parsed_val = float(row[1])
                            if pd.isna(parsed_val):
                                continue  # blank cell — don't treat as a found value
                            sheet_data[lbl] = round(parsed_val, 1)
                            sheet_labels[lbl] = raw_labels_sim.loc[idx].strip()
                        except (ValueError, TypeError):
                            continue

                # Extract every cabinet's channels, driven by Cabinet Configuration
                # (not a fixed field list) — this is what lets a non-numeric channel like
                # "tf Box" count as an extra thermocouple when a cabinet's configured count
                # exceeds the plain numbered ones found in the file.
                _sim_cab_cfg = get_cabinet_sensor_config(selected_volume, selected_arrangement)
                _sim_channel_values, _sim_cab_notifications, _sim_slot_labels = extract_cabinet_channels(sheet_data, _sim_cab_cfg, sheet_labels)
                st.session_state.active_pulldown_form.update(_sim_channel_values)
                st.session_state.active_pulldown_form['_cabinet_notifications'] = _sim_cab_notifications
                st.session_state.active_pulldown_form['_slot_labels'] = _sim_slot_labels

                # S2 is not part of the cabinet system. If the new file has no S2, drop any
                # value left over from a previous upload so the field shows empty/required.
                if 's2' in sheet_data:
                    st.session_state.active_pulldown_form['S2'] = sheet_data['s2']
                else:
                    st.session_state.active_pulldown_form.pop('S2', None)

                # Sensor is tracked separately from the cabinet channels — only set if the
                # uploaded file actually has a "Sensor" reading; otherwise leave it absent
                # so the field shows empty/highlighted rather than a fabricated number
                found_sensor = find_sensor_reading(sheet_data)
                if found_sensor is not None:
                    st.session_state.active_pulldown_form['Sensor'] = found_sensor
                else:
                    st.session_state.active_pulldown_form.pop('Sensor', None)

                # Entry Code / Test ID — purely informational here (Tab 1 doesn't persist
                # records), so the person can visually confirm they uploaded the right file.
                _sim_entry_code, _sim_test_id = extract_pulldown_entry_test_id(sim_pulldown_file)
                st.session_state.active_pulldown_form['_entry_code'] = _sim_entry_code
                st.session_state.active_pulldown_form['_test_id'] = _sim_test_id
                st.session_state.active_pulldown_form['_checkpoints'] = extract_pulldown_checkpoints(sim_pulldown_file)

                st.session_state.last_uploaded_sim_file = sim_pulldown_file
                # Increment key version to instantly clear old component cache and force update UI inputs
                st.session_state.sim_ver += 1
                st.toast("🟢 Parsed values pulled from sheet!", icon="📊")
                st.rerun()
            except Exception as e:
                st.error(f"Error parsing configuration: {str(e)}")

        if st.session_state.active_pulldown_form.get('_entry_code') or st.session_state.active_pulldown_form.get('_test_id'):
            st.caption(
                f"📄 Entry Code: `{st.session_state.active_pulldown_form.get('_entry_code') or 'Not found'}`  |  "
                f"Test ID: `{st.session_state.active_pulldown_form.get('_test_id') or 'Not found'}`"
            )

        # Dynamic channel list, driven entirely by Cabinet Configuration — this is both
        # the set of input boxes shown below AND the exact order run_automated_simulation
        # will use to build the live query vector, so the two always stay in sync.
        cab_sensor_cfg = get_cabinet_sensor_config(selected_volume, selected_arrangement)
        pulldown_feature_keys = all_channel_keys(cab_sensor_cfg) + ['S2']

        _cab_notifications = st.session_state.active_pulldown_form.get('_cabinet_notifications', [])
        for _note in _cab_notifications:
            st.caption(_note)

        _slot_labels = st.session_state.active_pulldown_form.get('_slot_labels', {})
        default_defaults = {'tf-1': -24.4, 'tf-2': -21.8, 'tf-3': -22.8, 'tf-4': -26.2, 'tf-5': -26.4, 'tc-1': 1.9, 'tc-2': 1.6, 'tc-3': 0.5}

        new_pulldown_input = []
        CHUNK_SIZE = 8
        for chunk_start in range(0, len(pulldown_feature_keys), CHUNK_SIZE):
            chunk = pulldown_feature_keys[chunk_start:chunk_start + CHUNK_SIZE]
            u_cols = st.columns(len(chunk))
            for col, feat in zip(u_cols, chunk):
                # Prioritize extracted file data if available, otherwise use defaults
                if feat in st.session_state.active_pulldown_form:
                    curr_val = round(float(st.session_state.active_pulldown_form[feat]), 1)
                elif feat == 'S2':
                    curr_val = None  # required: no fabricated default, must come from the file or be typed
                else:
                    curr_val = default_defaults.get(feat, 0.0)

                # Show which physical channel a fallback-matched slot (e.g. "tf Box") came from
                label = f"{feat} ({_slot_labels[feat]}):" if feat in _slot_labels else f"{feat}:"

                # Bound dynamic widget version to key parameters to force a redraw when new files parse
                val = col.number_input(
                    label,
                    value=curr_val,
                    step=0.1,
                    format="%.1f",
                    placeholder="Required" if feat == 'S2' else None,
                    key=f"sim_inp_{p_key}_{c_key}_{feat}_v{st.session_state.sim_ver}"
                )
                new_pulldown_input.append(val)

        if 'S2' in pulldown_feature_keys and new_pulldown_input[pulldown_feature_keys.index('S2')] is None:
            st.caption(":red[⚠️ S2 was not found in the uploaded pulldown file — enter it above, or upload a corrected file. Predictions are blocked until S2 is provided.]")

        # Sensor is a genuinely distinct reading (from the pulldown file's own "Sensor" row),
        # kept separate from the 10 fields above. Same behavior as tf-1..S2: auto-filled and
        # editable when found in the file; empty, highlighted, and editable when not found.
        sensor_missing_in_file = 'Sensor' not in st.session_state.active_pulldown_form
        sensor_widget_key = f"sim_inp_{p_key}_{c_key}_Sensor_v{st.session_state.sim_ver}"

        sensor_col, caption_col = st.columns([1, 3])
        if sensor_missing_in_file:
            with sensor_col:
                new_pulldown_sensor = st.number_input(
                    "Sensor (°C):",
                    value=None,
                    step=0.1,
                    format="%.1f",
                    placeholder="No data",
                    key=sensor_widget_key
                )
            with caption_col:
                if new_pulldown_sensor is None:
                    # Best-effort visual highlight — targets the input by its accessible label.
                    # If a future Streamlit version renders this differently, the highlight simply
                    # won't apply; the empty field and caption below still make the gap clear.
                    st.markdown(
                        "<style>input[aria-label='Sensor (°C):']{border:2px solid #ff4b4b !important; "
                        "background-color:rgba(255,75,75,0.12) !important;}</style>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        ":red[⚠️ No Sensor reading found in the uploaded pulldown file — please enter it manually.]"
                    )
                else:
                    st.markdown(":blue[✏️ Sensor value entered manually.]")
        else:
            _file_sensor_val = round(float(st.session_state.active_pulldown_form['Sensor']), 1)
            with sensor_col:
                new_pulldown_sensor = st.number_input(
                    "Sensor (°C):",
                    value=_file_sensor_val,
                    step=0.1,
                    format="%.1f",
                    key=sensor_widget_key
                )
            with caption_col:
                if new_pulldown_sensor == _file_sensor_val:
                    st.markdown(":green[✅ Sensor reading found in the uploaded file.]")
                else:
                    st.markdown(":blue[✏️ Sensor value entered manually.]")
            
        st.markdown("---")
        
        # ================= STEP 2: SET MULTI-SENSOR SIMULATION STEPS =================
        st.markdown("#### Step 2: Set Multi-Sensor Simulation Steps")

        sensor_names_for_tab1 = discover_sensor_names(vol_records)
        if len(sensor_names_for_tab1) > 1:
            st.caption(
                f"Training data has {len(sensor_names_for_tab1)} sensors: {', '.join(sensor_names_for_tab1)}. "
                f"Each query point needs a Min and Max reading per sensor. Each sensor is predicted separately "
                f"and the results are averaged."
            )
        else:
            st.caption("Each query point needs a Sensor Min and Sensor Max reading.")
        
        num_targets = st.number_input("Number of target sensor points:", min_value=1, max_value=None, value=5, step=1)
        
        target_sensor_pairs = []       # primary sensor only — feeds the existing prediction call
        target_sensor_pairs_all = []   # {sensor_name: (min, max)} per point — all sensors, for later use
        s_cols = st.columns(int(num_targets))
        for idx in range(int(num_targets)):
            # Set up default sensor values for the points (adjust the start value or step as needed)
            default_min_val = -27.5 + (idx * 1.5)
            default_max_val = 4.0

            with s_cols[idx]:
                st.markdown(f"**Point {idx+1}**")
                point_pairs = {}
                for sensor_name in sensor_names_for_tab1:
                    min_val = st.number_input(
                        f"{sensor_name} Min (°C):",
                        value=default_min_val,
                        step=0.1,
                        format="%.1f",
                        key=f"q_s_min_{p_key}_{c_key}_{idx}_{sensor_name}"
                    )
                    max_val = st.number_input(
                        f"{sensor_name} Max (°C):",
                        value=default_max_val,
                        step=0.1,
                        format="%.1f",
                        key=f"q_s_max_{p_key}_{c_key}_{idx}_{sensor_name}"
                    )
                    point_pairs[sensor_name] = (min_val, max_val)
            target_sensor_pairs.append(point_pairs[sensor_names_for_tab1[0]])
            target_sensor_pairs_all.append(point_pairs)
            
        if st.button("🚀 Generate Predictive CPT Dataset Matrices", type="primary"):
            _missing_inputs = []
            if new_pulldown_sensor is None:
                _missing_inputs.append("Sensor")
            if 'S2' in pulldown_feature_keys and new_pulldown_input[pulldown_feature_keys.index('S2')] is None:
                _missing_inputs.append("S2")

            if _missing_inputs:
                st.error(
                    f"❌ Cannot predict: {' and '.join(_missing_inputs)} value is missing. "
                    f"Enter it in the highlighted field above, or rework the pulldown file so it contains it, then try again."
                )
            else:
                with st.spinner("Processing automated interpolation runs..."):
                    df_final_predictions = run_automated_simulation(
                        vol_records, new_pulldown_input, new_pulldown_sensor, target_sensor_pairs, pulldown_feature_keys,
                        sensor_names=sensor_names_for_tab1, target_sensor_points=target_sensor_pairs_all,
                        cabinet_config=get_cabinet_sensor_config(selected_volume, selected_arrangement)
                    )

                    if df_final_predictions.empty:
                        st.error("Simulation engine run failed. Make sure dataset memory contains recorded instances.")
                    else:
                        st.markdown("### 📊 Consolidated Predictive Simulation Output Matrix")
                        df_final_predictions = round_df(add_avg_columns(df_final_predictions))
                        st.markdown(render_merged_predictions_table(df_final_predictions), unsafe_allow_html=True)

                        # ---- Checkpoint-refined predictions (optional extra, baseline above is unchanged) ----
                        _new_cps = st.session_state.active_pulldown_form.get('_checkpoints') or {}
                        _stored_have_cps = any(r.get("pulldown_checkpoints") for r in vol_records)
                        if _new_cps and _stored_have_cps:
                            with st.spinner("Refining with Pulldown checkpoints..."):
                                _cp_results = run_checkpoint_refined_simulations(
                                    vol_records, _new_cps, new_pulldown_sensor, target_sensor_pairs,
                                    sensor_names=sensor_names_for_tab1, target_sensor_points=target_sensor_pairs_all,
                                    cabinet_config=get_cabinet_sensor_config(selected_volume, selected_arrangement))
                            if _cp_results:
                                st.markdown("### 📍 Checkpoint-Refined Predictions")
                                st.caption(
                                    "Each block repeats the prediction using the readings at one Pulldown checkpoint. "
                                    "Confidence depends on how many stored runs contain that same checkpoint "
                                    "(3 or more = High, 2 = Medium, 1 = Low). Sorted with the best-supported first."
                                )
                                for _res in _cp_results:
                                    _icon = {"High": "🟢", "Medium": "🟡", "Low": "🟠"}[_res["confidence"]]
                                    with st.expander(
                                        f"{_icon} {_res['name']} — {_res['confidence']} confidence "
                                        f"({_res['support']} stored run(s), {_res['channels']} thermocouples used)"
                                    ):
                                        _cp_df = round_df(add_avg_columns(_res["df"]))
                                        st.markdown(render_merged_predictions_table(_cp_df), unsafe_allow_html=True)
                            else:
                                st.info("📍 No checkpoint in this Pulldown file matches a checkpoint stored in your training runs, so no refined prediction was made.")
                        elif _new_cps and not _stored_have_cps:
                            st.info("📍 This Pulldown file has checkpoints, but none of your stored training runs do — only the normal prediction is shown.")

# ================= TAB 2: DATA REPOSITORY ROOM =================
with tab2:
    st.subheader(f"Onboard Lab Reports for [{selected_volume}] ({selected_arrangement})")

    # ============ CABINET & SENSOR CONFIGURATION ============
    cab_sensor_cfg = get_cabinet_sensor_config(selected_volume, selected_arrangement)

    with st.expander(f"🗄️ Cabinet & Sensor Configuration for [{selected_volume}] ({selected_arrangement})", expanded=False):
        st.markdown("##### Cabinets")
        st.caption("Every cabinet listed here is active. Delete a cabinet to stop collecting/predicting it. Set the thermocouple count per cabinet — there's no upper limit; if an uploaded file is missing a channel, you'll be told which ones weren't found.")

        cabinets_changed = False
        cab_to_delete = None
        for cab_idx, cab in enumerate(cab_sensor_cfg["cabinets"]):
            c_name, c_count, c_del = st.columns([3, 2, 1])
            with c_name:
                st.markdown(f"**{cab['name']}** `({cab['prefix']}-N)`")
            with c_count:
                new_count = st.number_input(
                    "Thermocouple count",
                    min_value=0, max_value=None, step=1, value=int(cab.get("count", 0)),
                    key=f"cab_count_{selected_volume}_{selected_arrangement}_{cab_idx}",
                    label_visibility="collapsed"
                )
                if int(new_count) != cab.get("count", 0):
                    cab["count"] = int(new_count)
                    cabinets_changed = True
            with c_del:
                if st.button("🗑️", key=f"cab_del_{selected_volume}_{selected_arrangement}_{cab_idx}"):
                    cab_to_delete = cab_idx

        if cab_to_delete is not None:
            removed_name = cab_sensor_cfg["cabinets"][cab_to_delete]["name"]
            cab_sensor_cfg["cabinets"].pop(cab_to_delete)
            cab_sensor_cfg["cabinet_sensor_map"].pop(removed_name, None)
            save_memory_to_disk(st.session_state.db)
            st.success(f"Cabinet '{removed_name}' removed.")
            st.rerun()
        elif cabinets_changed:
            save_memory_to_disk(st.session_state.db)

        new_cab_name = st.text_input("➕ Add Cabinet:", placeholder="e.g., DC", key=f"input_cabinet_{st.session_state.cabinet_form_id}")
        if st.button("Register Cabinet"):
            if new_cab_name and new_cab_name.strip():
                clean_name = new_cab_name.strip()
                existing_names = [c["name"].strip().upper() for c in cab_sensor_cfg["cabinets"]]
                if clean_name.upper() in existing_names:
                    st.error(f"A cabinet named '{clean_name}' already exists.")
                else:
                    cab_sensor_cfg["cabinets"].append({
                        "name": clean_name,
                        "prefix": derive_cabinet_prefix(clean_name),
                        "count": 3,
                    })
                    save_memory_to_disk(st.session_state.db)
                    st.session_state.cabinet_form_id += 1
                    st.success(f"Cabinet '{clean_name}' added.")
                    st.rerun()

        st.markdown("---")
        st.markdown("##### Sensors")
        st.caption("With only 1 sensor, everything works as a single shared Sensor Min/Max. With 2+ sensors, assign which sensor(s) control each thermocouple below — a channel left unassigned to any sensor is treated as controlled by all active sensors.")

        sensor_to_delete = None
        for s_idx, s_name in enumerate(cab_sensor_cfg["sensors"]):
            s_col, s_del_col = st.columns([4, 1])
            with s_col:
                st.markdown(f"**{s_name}**")
            with s_del_col:
                if len(cab_sensor_cfg["sensors"]) > 1 and st.button("🗑️", key=f"sensor_del_{selected_volume}_{selected_arrangement}_{s_idx}"):
                    sensor_to_delete = s_idx

        if sensor_to_delete is not None:
            removed_sensor = cab_sensor_cfg["sensors"][sensor_to_delete]
            cab_sensor_cfg["sensors"].pop(sensor_to_delete)
            # Drop the removed sensor from any per-cabinet assignments
            for cab_name, assigned in cab_sensor_cfg["cabinet_sensor_map"].items():
                if removed_sensor in assigned:
                    assigned.remove(removed_sensor)
            save_memory_to_disk(st.session_state.db)
            st.success(f"'{removed_sensor}' removed.")
            st.rerun()

        new_sensor_name = st.text_input("➕ Add Sensor:", placeholder="e.g., Sensor-2", key=f"input_sensor_{st.session_state.sensor_form_id}")
        if st.button("Register Sensor"):
            if new_sensor_name and new_sensor_name.strip():
                clean_sensor = new_sensor_name.strip()
                if clean_sensor in cab_sensor_cfg["sensors"]:
                    st.error(f"A sensor named '{clean_sensor}' already exists.")
                else:
                    cab_sensor_cfg["sensors"].append(clean_sensor)
                    save_memory_to_disk(st.session_state.db)
                    st.session_state.sensor_form_id += 1
                    st.success(f"Sensor '{clean_sensor}' added.")
                    st.rerun()

        # Per-cabinet sensor assignment — only meaningful with 2+ sensors. A sensor picked
        # here controls every thermocouple in that cabinet.
        if len(cab_sensor_cfg["sensors"]) >= 2:
            st.markdown("###### Which sensor(s) control each cabinet")
            st.caption("A sensor picked for a cabinet controls all thermocouples in it. Leave a cabinet empty to have every sensor control it. S2 is always controlled by every sensor.")
            if not cab_sensor_cfg["cabinets"]:
                st.info("No cabinets configured yet — add a cabinet above.")
            else:
                assignment_changed = False
                # The sensor list is part of each widget key, so the selector resets cleanly
                # whenever a sensor is added or removed instead of holding a stale choice.
                sensors_key_part = "_".join(cab_sensor_cfg["sensors"])
                for cab in cab_sensor_cfg["cabinets"]:
                    current_assignment = cab_sensor_cfg["cabinet_sensor_map"].get(cab["name"], [])
                    current_assignment = [x for x in current_assignment if x in cab_sensor_cfg["sensors"]]
                    new_assignment = st.multiselect(
                        f"{cab['name']} — {int(cab.get('count', 0))} thermocouple(s)",
                        options=cab_sensor_cfg["sensors"],
                        default=current_assignment,
                        key=f"cab_sensor_{selected_volume}_{selected_arrangement}_{cab['name']}_{sensors_key_part}",
                        help="Leave empty to have every sensor control this cabinet."
                    )
                    if set(new_assignment) != set(current_assignment):
                        cab_sensor_cfg["cabinet_sensor_map"][cab["name"]] = new_assignment
                        assignment_changed = True
                if assignment_changed:
                    save_memory_to_disk(st.session_state.db)

    st.markdown("---")

    repo_c1, repo_c2 = st.columns(2)
    with repo_c1:
        repo_p_ambient = st.selectbox("Source Pulldown File Ambient Layer:", ["32°C", "43°C"], key="repo_p_amb")
    with repo_c2:
        repo_c_ambient = st.selectbox("Source Connected CPT File Ambient Layer:", ["16°C", "32°C", "43°C"], index=1, key="repo_c_amb")
        
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
                # 1. PARSE PULLDOWN DATA
                df_p_sum = pd.read_excel(repo_pulldown_file, sheet_name=0, header=None)
                raw_labels_repo = df_p_sum[0].astype(str)
                df_p_sum[0] = raw_labels_repo.apply(normalize_sensor_name)
                
                sheet_data = {}
                sheet_labels = {}  # normalized key -> original text (e.g. "tfbox" -> "tf Box")
                for idx, row in df_p_sum.dropna(subset=[0]).iterrows():
                    lbl = row[0]
                    if lbl not in sheet_data:
                        try:
                            parsed_val = float(row[1])
                            if pd.isna(parsed_val):
                                continue  # blank cell — don't treat as a found value
                            sheet_data[lbl] = round(parsed_val, 1)
                            sheet_labels[lbl] = raw_labels_repo.loc[idx].strip()
                        except (ValueError, TypeError):
                            continue

                # Extract every cabinet's channels, driven by Cabinet Configuration
                # (not a fixed field list) — same mechanism used in Tab 1, so a non-numeric
                # channel like "tf Box" counts as an extra thermocouple here too.
                _repo_cab_cfg = get_cabinet_sensor_config(selected_volume, selected_arrangement)
                p_extracted, cabinet_notifications, repo_slot_labels = extract_cabinet_channels(sheet_data, _repo_cab_cfg, sheet_labels)

                # S2 is not part of the cabinet system — same handling as always
                if 's2' in sheet_data:
                    p_extracted['S2'] = sheet_data['s2']

                # None (not 0.0) when the file has no "Sensor" row — 0.0 could be a real reading
                resolved_sensor = find_sensor_reading(sheet_data)

                # Optional richer data: named checkpoint columns (e.g. "tfa: -6.0",
                # "tca: 8.0") found in elaborated Pulldown report formats. Empty {} for
                # simple/Summary-style files — the baseline Avg-only path above is
                # unaffected either way.
                pulldown_checkpoints = extract_pulldown_checkpoints(repo_pulldown_file)

                # Entry Code / Test ID from the Pulldown file — typically embedded in its
                # filename (e.g. "PCT-2_Report_307L_F-12043_R029809.xlsx"), with a cell-scan
                # fallback for report styles that might embed it in a cell instead.
                pulldown_entry_code, pulldown_test_id = extract_pulldown_entry_test_id(repo_pulldown_file)

                # Entry Code / Test ID from the CPT file — extracted independently of which
                # CPT row-parsing strategy below ends up succeeding, since this only needs
                # the raw header cells, not the data table. Best-effort: failures here never
                # block the rest of the upload.
                entry_code = None
                test_id = None
                try:
                    repo_cpt_file.seek(0)
                    _wb_meta = openpyxl.load_workbook(repo_cpt_file, data_only=True)
                    _ws_meta = _wb_meta[_wb_meta.sheetnames[0]]
                    entry_code = find_labeled_value(_ws_meta, ["entry code"], "F-")
                    test_id = find_labeled_value(_ws_meta, ["test id"], "R0")
                except Exception:
                    pass
                finally:
                    repo_cpt_file.seek(0)  # rewind so the strategies below read from the start

                # Cross-check: the Pulldown and CPT files should describe the same test run.
                # Prefer the CPT-side value when both are found and agree; warn if they disagree
                # (likely mismatched files uploaded together by mistake), and fall back to
                # whichever single source found a value if the other didn't find one at all.
                if entry_code and pulldown_entry_code and entry_code.upper() != pulldown_entry_code.upper():
                    st.warning(f"⚠️ Entry Code mismatch: CPT file says '{entry_code}', Pulldown file says '{pulldown_entry_code}'. Double-check you uploaded matching files.")
                elif not entry_code:
                    entry_code = pulldown_entry_code

                if test_id and pulldown_test_id and test_id.upper() != pulldown_test_id.upper():
                    st.warning(f"⚠️ Test ID mismatch: CPT file says '{test_id}', Pulldown file says '{pulldown_test_id}'. Double-check you uploaded matching files.")
                elif not test_id:
                    test_id = pulldown_test_id
                
                # 2. PARSE CPT DATA
                cpt_match_notes = []   # Pulldown <-> CPT thermocouple matching messages
                cpt_structured = {}
                parsed_successfully = False

                # -------------------------------------------------------------
                # STRATEGY A : Visible Ambient Parser (FIXED VARIABLES)
                # -------------------------------------------------------------
                try:
                    import openpyxl
                    from openpyxl.utils import get_column_letter

                    wb = openpyxl.load_workbook(repo_cpt_file, data_only=True)
                    
                    if "ANALYSIS REPORT" in wb.sheetnames:
                        sheet_name = "ANALYSIS REPORT"
                    elif "CPT CALCULATION REPORT" in wb.sheetnames:
                        sheet_name = "CPT CALCULATION REPORT"
                    else:
                        sheet_name = wb.sheetnames[0]
                        
                    ws = wb[sheet_name]
                    cpt_structured = {}

                    # Safe numeric extraction, rounded to 1 decimal at the source
                    def safe_float(val):
                        if val is None:
                            return 0.0
                        try:
                            if isinstance(val, str):
                                val = val.replace("°C", "").replace("̊C", "").strip()
                            return round(float(val), 1)
                        except (ValueError, TypeError):
                            return 0.0

                    # Unhidden cell detection validation
                    def get_visible_value(row_idx, col_idx):
                        if ws.row_dimensions[row_idx].hidden:
                            return None
                        col_letter = get_column_letter(col_idx)
                        if ws.column_dimensions[col_letter].hidden:
                            return None
                        return ws.cell(row_idx, col_idx).value

                    start_row = None
                    for check_row in range(1, ws.max_row + 1):
                        if ws.row_dimensions[check_row].hidden:
                            continue
                        
                        raw_a = get_visible_value(check_row, 1)
                        raw_b = get_visible_value(check_row, 2)

                        a_str = str(raw_a).strip().lower() if raw_a is not None else ""
                        b_str = str(raw_b).strip().lower() if raw_b is not None else ""

                        if ("th. knob" in a_str or "th knob" in a_str) and "data criteria" in b_str:
                            start_row = check_row + 2
                            break

                    if start_row is None:
                        raise Exception("Visible CPT table header coordinates not found.")

                    current_flag = None
                    for data_row in range(start_row, ws.max_row + 1):
                        if ws.row_dimensions[data_row].hidden:
                            continue

                        colA = get_visible_value(data_row, 1)
                        colB = get_visible_value(data_row, 2)

                        colA = "" if colA is None else str(colA).strip()
                        colB = "" if colB is None else str(colB).strip().lower()

                        if "level" in colA.lower() or "boost" in colA.lower():
                            current_flag = colA
                            if current_flag not in cpt_structured:
                                cpt_structured[current_flag] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0}

                        if current_flag is None:
                            continue
                        if current_flag not in cpt_structured:
                            cpt_structured[current_flag] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0}

                        # Read strictly from mapped unhidden cells, one sub-block per criteria row
                        if colB in metric_types:
                            cpt_structured[current_flag][colB] = {
                                "tf-1": safe_float(get_visible_value(data_row, 3)),
                                "tf-2": safe_float(get_visible_value(data_row, 4)),
                                "tf-3": safe_float(get_visible_value(data_row, 5)),
                                "tf-4": safe_float(get_visible_value(data_row, 6)),
                                "tf-5": safe_float(get_visible_value(data_row, 7)),
                                "tc-1": safe_float(get_visible_value(data_row, 13)),
                                "tc-2": safe_float(get_visible_value(data_row, 14)),
                                "tc-3": safe_float(get_visible_value(data_row, 15)),
                                "tvc": safe_float(get_visible_value(data_row, 17)),
                            }
                            # S2 is only meaningful on the Mean row; Sensor (Min) on the Min row,
                            # SensorMax on the Max row — same column (19) as Sensor, different row
                            if colB == "mean":
                                cpt_structured[current_flag]["S2"] = safe_float(get_visible_value(data_row, 21))
                            elif colB == "min":
                                cpt_structured[current_flag]["Sensor"] = safe_float(get_visible_value(data_row, 19))
                            elif colB == "max":
                                cpt_structured[current_flag]["SensorMax"] = safe_float(get_visible_value(data_row, 19))

                    if cpt_structured:
                        parsed_successfully = True
                        st.success("✅ Strategy A successful.")
                except Exception as e:
                    st.write(f"Strategy A failed: {e}")

                # -------------------------------------------------------------
                # STRATEGY A2 : "Data Criteria" Anchor Parser — newer report format
                # with named Cabinet/Sensor headers (e.g. "Freezer Cabinet",
                # "Sensor FC", "Sensor PC"), found by column offset from the
                # "Data Criteria" cell rather than fixed absolute columns, so it
                # works whether or not the sheet has an extra leading column.
                #
                # Sensor columns are detected dynamically (any header cell containing
                # "sensor", not just "Sensor FC"/"Sensor PC") and only kept if that same
                # sensor name is also found in the Pulldown file — e.g. a "Defrost
                # Sensor" column with no Pulldown counterpart is dropped, since there's
                # no way to query it later. Each matched sensor's Min/Max is stored
                # dynamically in cpt_structured[flag]["sensors"][name]. The legacy
                # Sensor/SensorMax fields are also populated, from whichever matched
                # sensor appears first by column order, so today's single-sensor
                # prediction pipeline keeps working unchanged — full multi-sensor
                # prediction is a later step.
                # -------------------------------------------------------------
                if not parsed_successfully:
                    try:
                        wb_a2 = openpyxl.load_workbook(repo_cpt_file, data_only=True)
                        ws_a2 = wb_a2[wb_a2.sheetnames[0]]

                        def safe_float_a2(val):
                            if val is None:
                                return 0.0
                            try:
                                if isinstance(val, str):
                                    val = val.replace("°C", "").replace("̊C", "").strip()
                                return round(float(val), 1)
                            except (ValueError, TypeError):
                                return 0.0

                        header_row_a2 = None
                        dc_col_a2 = None
                        for r in range(1, min(ws_a2.max_row, 60) + 1):
                            for c in range(1, min(ws_a2.max_column, 30) + 1):
                                v = ws_a2.cell(r, c).value
                                if isinstance(v, str) and v.strip().lower() == "data criteria":
                                    left_v = ws_a2.cell(r, c - 1).value
                                    if isinstance(left_v, str) and left_v.strip().lower() == "testflag":
                                        header_row_a2, dc_col_a2 = r, c
                                        break
                            if header_row_a2:
                                break

                        if header_row_a2 is None:
                            raise Exception("New-format 'Data Criteria' header not found.")

                        testflag_col_a2 = dc_col_a2 - 1
                        tf_cols_a2 = [dc_col_a2 + i for i in range(1, 6)]
                        tc_cols_a2 = [dc_col_a2 + i for i in range(8, 11)]
                        tvc_cols_a2 = [dc_col_a2 + i for i in range(12, 15)]
                        s2_col_a2 = dc_col_a2 + 18

                        # Dynamic sensor column detection — any header cell (same row as
                        # "Data Criteria") containing "sensor", in column order
                        all_cpt_sensor_cols_a2 = {}
                        for c in range(1, min(ws_a2.max_column, 30) + 1):
                            v = ws_a2.cell(header_row_a2, c).value
                            if isinstance(v, str) and "sensor" in v.lower():
                                all_cpt_sensor_cols_a2[v.strip()] = c

                        # Cross-match against every "sensor"-named reading found in the
                        # already-parsed Pulldown file (sheet_data/sheet_labels, parsed
                        # earlier in this same upload) — keep only names present in both
                        pulldown_sensor_readings_a2 = find_all_sensor_readings(sheet_data, sheet_labels)
                        pulldown_sensor_norms_a2 = {normalize_sensor_name(lbl) for lbl in pulldown_sensor_readings_a2.keys()}

                        sensor_cols_a2 = {}
                        unmatched_cpt_sensors_a2 = []
                        for label, col in all_cpt_sensor_cols_a2.items():
                            if normalize_sensor_name(label) in pulldown_sensor_norms_a2:
                                sensor_cols_a2[label] = col
                            else:
                                unmatched_cpt_sensors_a2.append(label)
                        unmatched_pulldown_sensors_a2 = [
                            lbl for lbl in pulldown_sensor_readings_a2
                            if normalize_sensor_name(lbl) not in {normalize_sensor_name(l) for l in all_cpt_sensor_cols_a2}
                        ]
                        if unmatched_cpt_sensors_a2:
                            st.caption(f"ℹ️ CPT sensor column(s) with no Pulldown match, not used: {', '.join(unmatched_cpt_sensors_a2)}")
                        if unmatched_pulldown_sensors_a2:
                            st.caption(f"ℹ️ Pulldown sensor reading(s) with no CPT match, not used as a regulator: {', '.join(unmatched_pulldown_sensors_a2)}")

                        # Whichever matched sensor is leftmost drives the legacy
                        # Sensor/SensorMax fields, for backward compatibility
                        primary_sensor_name_a2 = min(sensor_cols_a2, key=sensor_cols_a2.get) if sensor_cols_a2 else None

                        # Dynamic thermocouple detection from the CPT header rows, matched
                        # against the Pulldown file's thermocouples (same cabinet rules)
                        _first_sensor_col_a2 = min(all_cpt_sensor_cols_a2.values()) if all_cpt_sensor_cols_a2 else min(ws_a2.max_column, 30) + 1
                        cpt_channel_cols_a2 = detect_cpt_channel_columns(ws_a2, header_row_a2, dc_col_a2 + 1, _first_sensor_col_a2 - 1)
                        _cab_cfg_a2 = get_cabinet_sensor_config(selected_volume, selected_arrangement)
                        _cpt_match_done_a2 = False
                        _pulldown_slots_a2 = set(k for k in p_extracted if k != "S2")

                        cpt_structured = {}
                        current_flag_a2 = None
                        for data_row in range(header_row_a2 + 2, ws_a2.max_row + 1):
                            testflag_val = ws_a2.cell(data_row, testflag_col_a2).value
                            crit_val = ws_a2.cell(data_row, dc_col_a2).value
                            crit = str(crit_val).strip().lower() if crit_val is not None else ""

                            if testflag_val is not None and str(testflag_val).strip():
                                current_flag_a2 = str(testflag_val).strip()
                                if current_flag_a2 not in cpt_structured:
                                    cpt_structured[current_flag_a2] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0, "sensors": {}}

                            if current_flag_a2 is None:
                                continue
                            if current_flag_a2 not in cpt_structured:
                                cpt_structured[current_flag_a2] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0, "sensors": {}}

                            if crit in metric_types:
                                cpt_structured[current_flag_a2][crit] = {
                                    "tf-1": safe_float_a2(ws_a2.cell(data_row, tf_cols_a2[0]).value),
                                    "tf-2": safe_float_a2(ws_a2.cell(data_row, tf_cols_a2[1]).value),
                                    "tf-3": safe_float_a2(ws_a2.cell(data_row, tf_cols_a2[2]).value),
                                    "tf-4": safe_float_a2(ws_a2.cell(data_row, tf_cols_a2[3]).value),
                                    "tf-5": safe_float_a2(ws_a2.cell(data_row, tf_cols_a2[4]).value),
                                    "tc-1": safe_float_a2(ws_a2.cell(data_row, tc_cols_a2[0]).value),
                                    "tc-2": safe_float_a2(ws_a2.cell(data_row, tc_cols_a2[1]).value),
                                    "tc-3": safe_float_a2(ws_a2.cell(data_row, tc_cols_a2[2]).value),
                                    "tvc": round(sum(safe_float_a2(ws_a2.cell(data_row, c).value) for c in tvc_cols_a2) / 3, 1),
                                }
                                # Extra thermocouples (tf Box -> tf-6, tvc-1..3, ...) that are
                                # present in BOTH files are stored next to the fixed ones
                                _row_ch, _row_notes, _row_labels = extract_cpt_row_channels(ws_a2, data_row, cpt_channel_cols_a2, _cab_cfg_a2)
                                if not _cpt_match_done_a2:
                                    _, _m_notes = match_pulldown_cpt_channels(p_extracted, _row_ch, _row_labels)
                                    cpt_match_notes.extend([n for n in _row_notes if n.startswith('ℹ️')] + _m_notes)  # per-cabinet 'not found' warnings are covered by the match notes
                                    _cpt_match_done_a2 = True
                                for _k, _v in _row_ch.items():
                                    if _k in _pulldown_slots_a2 and _k not in cpt_structured[current_flag_a2][crit]:
                                        cpt_structured[current_flag_a2][crit][_k] = _v
                                if crit == "mean":
                                    cpt_structured[current_flag_a2]["S2"] = safe_float_a2(ws_a2.cell(data_row, s2_col_a2).value)
                                elif crit in ("min", "max"):
                                    for sensor_name, sensor_col in sensor_cols_a2.items():
                                        sensor_block = cpt_structured[current_flag_a2]["sensors"].setdefault(sensor_name, {"min": 0.0, "max": 0.0})
                                        sensor_block[crit] = safe_float_a2(ws_a2.cell(data_row, sensor_col).value)
                                        if sensor_name == primary_sensor_name_a2:
                                            cpt_structured[current_flag_a2]["Sensor" if crit == "min" else "SensorMax"] = sensor_block[crit]

                        if cpt_structured:
                            parsed_successfully = True
                            st.success("✅ Strategy A2 (new report format) successful.")
                    except Exception as e:
                        st.write(f"Strategy A2 failed: {e}")

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
                                if current_flag not in cpt_structured: cpt_structured[current_flag] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0}
                                    
                                try: rt_val = float(row.get("runtime_pct", 0.0))
                                except (ValueError, TypeError): rt_val = 0.0
                                    
                                if rt_val == 100: continue

                                metric_map = {"mean": "mean", "avg": "mean", "average": "mean",
                                               "min": "min", "max": "max", "(max+min)/2": "(max+min)/2"}
                                metric_key = metric_map.get(val_crit)
                                if metric_key:
                                    cpt_structured[current_flag][metric_key] = {
                                        "tf-1": round(float(row.get("tf1", 0.0)), 1), "tf-2": round(float(row.get("tf2", 0.0)), 1),
                                        "tf-3": round(float(row.get("tf3", 0.0)), 1), "tf-4": round(float(row.get("tf4", 0.0)), 1),
                                        "tf-5": round(float(row.get("tf5", 0.0)), 1), "tc-1": round(float(row.get("tc1", 0.0)), 1),
                                        "tc-2": round(float(row.get("tc2", 0.0)), 1), "tc-3": round(float(row.get("tc3", 0.0)), 1),
                                        "tvc":  round(float(row.get("vc", 0.0)), 1),
                                    }
                                    if metric_key == "mean":
                                        cpt_structured[current_flag]["S2"] = round(float(row.get("s2", 0.0)), 1)
                                    elif metric_key == "min":
                                        cpt_structured[current_flag]["Sensor"] = round(float(row.get("sensor", 0.0)), 1)
                                    elif metric_key == "max":
                                        cpt_structured[current_flag]["SensorMax"] = round(float(row.get("sensor", 0.0)), 1)
                            if cpt_structured: parsed_successfully = True
                    except Exception: pass

                # --- STRATEGY C: Section Layout Matrix Format (Mean values only — this layout has no separate Min/Max/Avg rows) ---
                if not parsed_successfully:
                    try:
                        df_cpt_seg = pd.read_excel(repo_cpt_file, sheet_name=0, header=None)
                        current_flag = "Unknown"
                        for idx, r in df_cpt_seg.iterrows():
                            val_0 = str(r.iloc[0]).strip()
                            if pd.notna(r.iloc[0]) and ("level" in val_0.lower() or "boost" in val_0.lower()):
                                current_flag = val_0
                                if current_flag not in cpt_structured: cpt_structured[current_flag] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0, "mean": {}}
                                continue
                            if val_0.lower() in ["section", "min", "nan", ""] or pd.isna(r.iloc[0]): continue
                            clean_tag = normalize_sensor_name(val_0)
                            
                            if current_flag != "Unknown":
                                if current_flag not in cpt_structured: cpt_structured[current_flag] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0, "mean": {}}
                                if clean_tag == "sensor":
                                    try: cpt_structured[current_flag]["Sensor"] = round(float(r.iloc[5]), 1)
                                    except (ValueError, TypeError, IndexError): cpt_structured[current_flag]["Sensor"] = round(float(r.iloc[1]), 1)
                                elif clean_tag == "s2":
                                    cpt_structured[current_flag]["S2"] = round(float(r.iloc[8]), 1)
                                else:
                                    mapping_dict = {
                                        "tf1": "tf-1", "tf2": "tf-2", "tf3": "tf-3", "tf4": "tf-4", "tf5": "tf-5",
                                        "tc1": "tc-1", "tc2": "tc-2", "tc3": "tc-3"
                                    }
                                    if clean_tag in mapping_dict: cpt_structured[current_flag]["mean"][mapping_dict[clean_tag]] = round(float(r.iloc[8]), 1)
                                    elif "tvc" in clean_tag:
                                        if "tvc_vals" not in cpt_structured[current_flag]: cpt_structured[current_flag]["tvc_vals"] = []
                                        cpt_structured[current_flag]["tvc_vals"].append(float(r.iloc[8]))

                        for flg in cpt_structured:
                            if "tvc_vals" in cpt_structured[flg] and cpt_structured[flg]["tvc_vals"]:
                                cpt_structured[flg]["mean"]["tvc"] = round(sum(cpt_structured[flg]["tvc_vals"]) / len(cpt_structured[flg]["tvc_vals"]), 1)
                                del cpt_structured[flg]["tvc_vals"]
                        parsed_successfully = True
                    except Exception: pass

                # 3. SAVE DATA MATRIX AND FORCE RETENTION TO HARD DISK
                if not parsed_successfully or not cpt_structured:
                    raise ValueError("CPT processing pipeline failed. Spreadsheet structural pattern unknown.")

                # Sensor and S2 are required in BOTH files — without them the record can't
                # be used for prediction, so nothing is saved until the file is reworked.
                _missing_report = []
                if resolved_sensor is None:
                    _missing_report.append("Sensor is missing from the Pulldown file")
                if 's2' not in sheet_data:
                    _missing_report.append("S2 is missing from the Pulldown file")
                if not cpt_has_sensor_data(cpt_structured):
                    _missing_report.append("Sensor is missing from the CPT file (or no sensor name matched the Pulldown file)")
                if not cpt_has_s2_data(cpt_structured):
                    _missing_report.append("S2 is missing from the CPT file")
                if _missing_report:
                    st.error("❌ Nothing was saved. Please rework the file(s) and upload again:\n\n" + "\n".join(f"- {m}" for m in _missing_report))
                    st.stop()

                new_block = {
                    "pulldown_baseline_sensor": resolved_sensor,
                    "original_pulldown_baseline_sensor": resolved_sensor,
                    "entry_code": entry_code,
                    "test_id": test_id,
                    "pulldown_checkpoints": copy.deepcopy(pulldown_checkpoints),
                    "original_pulldown_data": copy.deepcopy(p_extracted),
                    "original_cpt_data": copy.deepcopy(cpt_structured),
                    "pulldown_data": copy.deepcopy(p_extracted),
                    "cpt_data": copy.deepcopy(cpt_structured)
                }
                
                verify_db_structure(selected_volume, selected_arrangement, p_repo_key, c_repo_key)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key].append(new_block)
                st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key] = st.session_state.db[selected_volume][selected_arrangement][p_repo_key][c_repo_key][-10:]
                
                save_memory_to_disk(st.session_state.db)
                
                st.session_state.p_file_key += 1
                st.session_state.cpt_file_key += 1
                
                st.success(f"🚀 Model Simulator Trained successfully! Hard-Backup saved to storage.")
                for _note in cabinet_notifications:
                    st.caption(_note)
                for _note in cpt_match_notes:
                    st.caption(_note)
                if pulldown_checkpoints:
                    st.info(f"📍 Also captured {len(pulldown_checkpoints)} named checkpoint(s) from the Pulldown file: {', '.join(pulldown_checkpoints.keys())}")
                st.rerun()
            except Exception:
                import traceback
                st.code(traceback.format_exc())
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
            
        # --- LOOP THROUGH AND RENDER EACH TRAINED DATASET ---
            for run_idx, record in enumerate(records):
                _entry_code_display = record.get('entry_code')
                _expander_title = f"📦 Trained Dataset Record #{run_idx + 1}"
                if _entry_code_display:
                    _expander_title += f" — {_entry_code_display}"
                with st.expander(_expander_title, expanded=(run_idx == 0)):
                    
                    # Row management buttons
                    c_btn1, c_btn2 = st.columns([4, 1])
                    with c_btn1:
                        _baseline_sensor_display = record.get('pulldown_baseline_sensor')
                        _baseline_sensor_text = "No data" if _baseline_sensor_display is None else f"{_baseline_sensor_display}°C"
                        st.markdown(f"**Baseline Sensor Target Setting:** `{_baseline_sensor_text}`")
                        _test_id_display = record.get('test_id')
                        st.caption(f"Entry Code: `{_entry_code_display or 'Not found'}`  |  Test ID: `{_test_id_display or 'Not found'}`")
                    with c_btn2:
                        if st.button("🗑️ Delete Dataset", key=f"del_ds_{p_inspect_key}_{c_inspect_key}_{run_idx}"):
                            records.pop(run_idx)
                            save_memory_to_disk(st.session_state.db)
                            st.success("Dataset purged successfully!")
                            st.rerun()

                    # Initialize a state version counter for this specific loop record if it doesn't exist
                    version_key = f"ver_{p_inspect_key}_{c_inspect_key}_{run_idx}"
                    if version_key not in st.session_state:
                        st.session_state[version_key] = 1
                    
                    current_ver = st.session_state[version_key]

                    # Section A: Display Pulldown Data Summary Table
                    st.markdown("#### 🔹 Counted Pulldown Matrix")

                    p_df = round_df(add_avg_columns(build_pulldown_df(record["pulldown_data"], record.get("pulldown_baseline_sensor"))))
                    
                    if "original_pulldown_data" not in record:
                        record["original_pulldown_data"] = record["pulldown_data"].copy()
                    if "original_pulldown_baseline_sensor" not in record:
                        record["original_pulldown_baseline_sensor"] = record.get("pulldown_baseline_sensor")

                    original_p_df = round_df(add_avg_columns(build_pulldown_df(record["original_pulldown_data"], record.get("original_pulldown_baseline_sensor"))))

                    pulldown_edit_mode_key = f"pulldown_edit_mode_{p_inspect_key}_{c_inspect_key}_{run_idx}"
                    if pulldown_edit_mode_key not in st.session_state:
                        st.session_state[pulldown_edit_mode_key] = False
                    p_undo_base_key = f"p_undo_{p_inspect_key}_{c_inspect_key}_{run_idx}_v{current_ver}"

                    if not st.session_state[pulldown_edit_mode_key]:
                        # VIEW MODE — same table style as the Original table below; Sensor is
                        # highlighted red if no reading was ever found for this record
                        st.markdown(render_simple_html_table(p_df, highlight_missing_cols=["Sensor"]), unsafe_allow_html=True)
                        if pd.isna(p_df.iloc[0]["Sensor"]):
                            st.caption(":red[⚠️ No Sensor reading on file for this record.]")
                        if st.button("✏️ Edit Pulldown Matrix", key=f"p_edit_toggle_on_{run_idx}_v{current_ver}"):
                            st.session_state[pulldown_edit_mode_key] = True
                            st.rerun()
                        # Nothing pending to save on the pulldown side while just viewing
                        edited_p_df = p_df.copy()
                    else:
                        # EDIT MODE — editable grid with Undo/Redo (whole-table snapshots, not
                        # per-cell — see undoable_data_editor). Computed average columns (e.g.
                        # tf-a, tvc-a) are shown but not directly editable. st.data_editor
                        # can't highlight individual cells, so a caption below flags a missing
                        # Sensor instead.
                        edited_p_df = undoable_data_editor(
                            p_undo_base_key,
                            p_df,
                            build_column_config(p_df, disabled_cols=avg_column_names(p_df))
                        )
                        if pd.isna(edited_p_df.iloc[0]["Sensor"]):
                            st.caption(":red[⚠️ Sensor is empty — enter a value above if available.]")

                        if st.button("❌ Cancel Editing", key=f"p_edit_cancel_{run_idx}_v{current_ver}"):
                            reset_undo_state(p_undo_base_key)
                            st.session_state[pulldown_edit_mode_key] = False
                            st.rerun()
                    
                    st.markdown("##### 📄 Original Uploaded Pulldown Matrix")
                    # Always reflects the file as first uploaded — never touched by edits
                    st.markdown(render_simple_html_table(original_p_df, highlight_missing_cols=["Sensor"]), unsafe_allow_html=True)
                    
                    # Section B: Display CPT Multivariable Flags Data Matrix
                    st.markdown("#### 🔹 Counted Positions for CPT Matrix")

                    def build_cpt_rows(cpt_data_dict):
                        rows = []
                        sensor_names_here = sensor_names_in_cpt_data(cpt_data_dict)
                        # Thermocouple columns: the 9 classic ones, plus any extra stored for this
                        # record (e.g. tf-6 = "tf Box", tvc-1..3). If individual tvc-N values exist
                        # the single "tvc" average column is replaced by them (tvc-a is computed).
                        fixed_cols = ["tf-1", "tf-2", "tf-3", "tf-4", "tf-5", "tc-1", "tc-2", "tc-3", "tvc"]
                        extra_cols = set()
                        for _blk in cpt_data_dict.values():
                            if not isinstance(_blk, dict):
                                continue
                            for _mk in metric_types:
                                for _k in (_blk.get(_mk) or {}):
                                    if re.match(r'^.+-\d+$', str(_k)) and _k not in fixed_cols:
                                        extra_cols.add(_k)
                        chan_cols = fixed_cols + sorted(extra_cols)
                        if any(k.startswith("tvc-") for k in extra_cols):
                            chan_cols.remove("tvc")
                        _rank = {"tf": 0, "tcc": 1, "tc": 2, "tvc": 3}
                        def _ch_key(k):
                            m = re.match(r'^(.+)-(\d+)$', k)
                            p, n = (m.group(1), int(m.group(2))) if m else (k, 0)
                            return (_rank.get(p, 99), p, n)
                        chan_cols.sort(key=_ch_key)
                        for flag_name, flag_block in cpt_data_dict.items():
                            for metric_key in metric_types:
                                metric_data = flag_block.get(metric_key) if isinstance(flag_block, dict) else None
                                if not metric_data:
                                    continue
                                row = {"Test Flag": flag_name, "Metric": metric_labels[metric_key]}
                                for _c in chan_cols:
                                    row[_c] = metric_data.get(_c, 0.0 if _c in fixed_cols else np.nan)
                                row["S2"] = flag_block.get("S2", 0.0) if metric_key == "mean" else np.nan
                                if sensor_names_here:
                                    # Dynamic multi-sensor record — one column per sensor name,
                                    # each showing its Min value on the Min row, Max on the Max
                                    # row, and blank elsewhere (same per-row convention as before)
                                    for sname in sensor_names_here:
                                        sdata = flag_block.get("sensors", {}).get(sname, {})
                                        if metric_key == "min":
                                            row[sname] = sdata.get("min", np.nan)
                                        elif metric_key == "max":
                                            row[sname] = sdata.get("max", np.nan)
                                        else:
                                            row[sname] = np.nan
                                else:
                                    # Legacy record with no "sensors" dict — keep the old single column
                                    row["Sensor"] = (
                                        flag_block.get("Sensor", 0.0) if metric_key == "min"
                                        else flag_block.get("SensorMax", 0.0) if metric_key == "max"
                                        else np.nan
                                    )
                                rows.append(row)
                        return rows

                    cpt_rows = build_cpt_rows(record["cpt_data"])

                    if cpt_rows:
                        cpt_df = round_df(add_avg_columns(pd.DataFrame(cpt_rows)))
                        
                        if "original_cpt_data" not in record:
                            record["original_cpt_data"] = record["cpt_data"].copy()

                        original_cpt_rows = build_cpt_rows(record["original_cpt_data"])
                        original_cpt_df = round_df(add_avg_columns(pd.DataFrame(original_cpt_rows)))
                        
                        cpt_edit_mode_key = f"cpt_edit_mode_{p_inspect_key}_{c_inspect_key}_{run_idx}"
                        if cpt_edit_mode_key not in st.session_state:
                            st.session_state[cpt_edit_mode_key] = False
                        cpt_undo_base_key = f"cpt_undo_{p_inspect_key}_{c_inspect_key}_{run_idx}_v{current_ver}"

                        if not st.session_state[cpt_edit_mode_key]:
                            # VIEW MODE — same real merged-cell (rowspan) look as the Original table below,
                            # reflecting the current (possibly already-edited) values
                            st.markdown(render_merged_cpt_table(cpt_df), unsafe_allow_html=True)
                            if st.button("✏️ Edit CPT Matrix", key=f"cpt_edit_toggle_on_{run_idx}_v{current_ver}"):
                                st.session_state[cpt_edit_mode_key] = True
                                st.rerun()
                            # Nothing pending to save on the CPT side while just viewing
                            edited_cpt_df = cpt_df.copy()
                        else:
                            # EDIT MODE — editable grid with Undo/Redo (whole-table snapshots, not
                            # per-cell — see undoable_data_editor). Average columns, Test Flag
                            # and Metric are locked.
                            edited_cpt_df = undoable_data_editor(
                                cpt_undo_base_key,
                                cpt_df,
                                build_column_config(
                                    cpt_df,
                                    disabled_cols=avg_column_names(cpt_df) + ["Test Flag", "Metric"],
                                    text_cols=["Test Flag", "Metric"]
                                )
                            )

                            if st.button("❌ Cancel Editing", key=f"cpt_edit_cancel_{run_idx}_v{current_ver}"):
                                reset_undo_state(cpt_undo_base_key)
                                st.session_state[cpt_edit_mode_key] = False
                                st.rerun()

                        st.markdown("##### 📄 Original Uploaded CPT Matrix")
                        # Always reflects the file as first uploaded — never touched by edits
                        st.markdown(render_merged_cpt_table(original_cpt_df), unsafe_allow_html=True)

                        # Detect changes
                        pulldown_changed = not edited_p_df.equals(p_df)
                        cpt_changed = not edited_cpt_df.equals(cpt_df)
                        dataset_changed = pulldown_changed or cpt_changed



                        if dataset_changed:
                            st.success("🟡 Unsaved changes detected.")
                            save_clicked = st.button(
                                "💾 Save Edited Dataset",
                                key=f"save_dataset_{run_idx}_v{current_ver}",
                                type="primary"
                            )
                        else:
                            st.info("No changes made.")
                            save_clicked = False

                        if save_clicked:
                            # Save Pulldown Matrix (average columns are computed display-only
                            # fields, never stored). Sensor lives separately as
                            # pulldown_baseline_sensor, not inside pulldown_data.
                            sensor_row_val = edited_p_df.iloc[0].get("Sensor")
                            record["pulldown_baseline_sensor"] = None if pd.isna(sensor_row_val) else float(sensor_row_val)
                            record["pulldown_data"] = edited_p_df.drop(columns=avg_column_names(edited_p_df) + ["Sensor"], errors="ignore").iloc[0].to_dict()

                            # Save CPT Matrix — reassemble the 4 rows per flag back into the nested structure
                            label_to_key = {v: k for k, v in metric_labels.items()}
                            # Sensor names this record's table was built with (empty for legacy
                            # single-sensor records, which use the plain "Sensor" column instead)
                            sensor_names_saved = sensor_names_in_cpt_data(record["cpt_data"])
                            new_cpt = {}
                            for _, row in edited_cpt_df.iterrows():
                                flag = row["Test Flag"]
                                metric_key = label_to_key.get(row["Metric"], str(row["Metric"]).lower())

                                if flag not in new_cpt:
                                    new_cpt[flag] = {"S2": 0.0, "Sensor": 0.0, "SensorMax": 0.0}
                                    if sensor_names_saved:
                                        new_cpt[flag]["sensors"] = {sn: {"min": 0.0, "max": 0.0} for sn in sensor_names_saved}

                                _skip_cols = {"Test Flag", "Metric", "S2", "Sensor"} | set(sensor_names_saved) | set(avg_column_names(edited_cpt_df.drop(columns=list(sensor_names_saved), errors="ignore")))
                                _chan_saved = {}
                                for _c in edited_cpt_df.columns:
                                    if _c in _skip_cols:
                                        continue
                                    if pd.notna(row[_c]):
                                        _chan_saved[_c] = float(row[_c])
                                # Keep the single "tvc" average in step when only tvc-1..N are shown
                                if "tvc" not in _chan_saved:
                                    _tvc_parts = [v for k, v in _chan_saved.items() if k.startswith("tvc-")]
                                    if _tvc_parts:
                                        _chan_saved["tvc"] = round(sum(_tvc_parts) / len(_tvc_parts), 1)
                                new_cpt[flag][metric_key] = _chan_saved
                                if metric_key == "mean" and pd.notna(row["S2"]):
                                    new_cpt[flag]["S2"] = float(row["S2"])
                                # The Sensor column is now a single field, populated per-row:
                                # its value on the Min row is the Sensor Min reading, and on the
                                # Max row is the Sensor Max reading — route each back accordingly.
                                if sensor_names_saved:
                                    # Dynamic multi-sensor record: each sensor column's value on the
                                    # Min row is that sensor's Min, on the Max row its Max
                                    for sn in sensor_names_saved:
                                        if sn in row.index and pd.notna(row[sn]) and metric_key in ("min", "max"):
                                            new_cpt[flag]["sensors"][sn][metric_key] = float(row[sn])
                                    # Keep the legacy Sensor/SensorMax in step with the primary
                                    # (first) sensor, which is what today's prediction reads
                                    primary_block = new_cpt[flag]["sensors"][sensor_names_saved[0]]
                                    new_cpt[flag]["Sensor"] = primary_block["min"]
                                    new_cpt[flag]["SensorMax"] = primary_block["max"]
                                else:
                                    if metric_key == "min" and "Sensor" in row.index and pd.notna(row["Sensor"]):
                                        new_cpt[flag]["Sensor"] = float(row["Sensor"])
                                    if metric_key == "max" and "Sensor" in row.index and pd.notna(row["Sensor"]):
                                        new_cpt[flag]["SensorMax"] = float(row["Sensor"])
                            record["cpt_data"] = new_cpt

                            # Commit file data structure changes to the physical disk 
                            save_memory_to_disk(st.session_state.db)
                            
                            # Return the Pulldown and CPT matrices to their read-only view, now showing the saved values
                            st.session_state[pulldown_edit_mode_key] = False
                            st.session_state[cpt_edit_mode_key] = False
                            reset_undo_state(p_undo_base_key)
                            reset_undo_state(cpt_undo_base_key)

                            # INCREMENT VERSION: Wipes out the stale data cache instantly on rerun
                            st.session_state[version_key] += 1
                            
                            st.success("✅ Dataset updated successfully.")
                            st.rerun()
                    else:
                        st.warning("No CPT entries found inside this specific record block.")
