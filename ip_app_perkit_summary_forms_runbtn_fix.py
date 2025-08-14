
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import math
import altair as alt

# --- Scenario Save / Load helpers ---
import json, numpy as _np
from datetime import date as _date, datetime as _dt, timedelta as _td

SCENARIO_VERSION = "1.0"

def _to_jsonable(x):
    """Convert pandas/NumPy/datetime values to plain JSON types."""
    import pandas as pd
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        if isinstance(x, float):
            try:
                if _np.isnan(x) or _np.isinf(x):
                    return None
            except Exception:
                pass
        return x
    if isinstance(x, (_np.integer,)):
        return int(x)
    if isinstance(x, (_np.floating,)):
        try:
            return None if _np.isnan(x) or _np.isinf(x) else float(x)
        except Exception:
            return float(x)
    if isinstance(x, (_np.bool_,)):
        return bool(x)
    try:
        import pandas as pd
        if isinstance(x, (pd.Timestamp, _dt, _date)):
            return pd.to_datetime(x).strftime("%Y-%m-%d")
        if isinstance(x, (pd.Timedelta, _td)):
            return str(x)
        if x is pd.NaT:
            return None
    except Exception:
        pass
    return str(x)

def _df_to_records(df):
    """Return list of dict rows with JSON-safe values."""
    import pandas as pd
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    out = []
    for _, row in df.iterrows():
        rec = {c: _to_jsonable(v) for c, v in row.items()}
        out.append(rec)
    return out

def _records_to_df(records):
    import pandas as pd
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    for col in df.columns:
        if "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                pass
    return df

def serialize_config():
    """Collect current UI/tables into a JSON-safe dict."""
    payload = {
        "_version": SCENARIO_VERSION,
        "start_date": _to_jsonable(st.session_state.get("start_date")),
        "end_date":   _to_jsonable(st.session_state.get("end_date")),
        # try pull both fraction and percent if available
        "buffer_fraction": globals().get("buffer_percent", None),
        "buffer_percent_pct": None if globals().get("buffer_percent", None) is None else int(round(float(globals().get("buffer_percent"))*100)),
        "allow_borrow": globals().get("allow_borrow", True),
        "randomization": globals().get("randomization", {}),
        "target_enrollment_global": globals().get("target_enrollment_global", 0),
        "countries_df": _df_to_records(globals().get("countries_df")),
        "visit_df": _df_to_records(globals().get("visit_df")),
        "sku_df": _df_to_records(globals().get("sku_df")),
        "lots_df": _df_to_records(globals().get("lots_df")),
        "sites_active_df": _df_to_records(globals().get("sites_active_df")),
    }
    return payload

def restore_config(payload: dict):
    """Apply a JSON scenario payload back into session_state and globals (where safe)."""
    import pandas as pd
    # dates
    try:
        st.session_state["start_date"] = pd.to_datetime(payload.get("start_date")).date()
    except Exception:
        pass
    try:
        st.session_state["end_date"] = pd.to_datetime(payload.get("end_date")).date()
    except Exception:
        pass
    # buffer
    frac = payload.get("buffer_fraction", None)
    if frac is None and "buffer_percent_pct" in payload:
        try:
            frac = float(payload["buffer_percent_pct"]) / 100.0
        except Exception:
            frac = None
    if isinstance(frac, (int, float)):
        globals()["buffer_percent"] = max(0.0, min(1.0, float(frac)))
    # allow_borrow
    if "allow_borrow" in payload:
        globals()["allow_borrow"] = bool(payload["allow_borrow"])
    # randomization
    if "randomization" in payload and isinstance(payload["randomization"], dict):
        globals()["randomization"] = payload["randomization"]
    # target
    if "target_enrollment_global" in payload:
        globals()["target_enrollment_global"] = int(payload["target_enrollment_global"])
    # tables back into session_state so editors can pick them up in your code if needed
    for k in ["countries_df","visit_df","sku_df","lots_df","sites_active_df"]:
        if k in payload:
            st.session_state[k] = _records_to_df(payload.get(k, []))


st.set_page_config(page_title="IP Forecasting Tool v8.10h", layout="wide")
st.title("Investigational Product Forecasting Tool v8.10h")

# ---------- Sidebar: Trial configuration ----------
st.sidebar.header("Trial Configuration")
if "start_date" not in st.session_state:
    st.session_state["start_date"] = datetime(2026, 1, 1).date()
if "end_date" not in st.session_state:
    st.session_state["end_date"] = datetime(2027, 12, 31).date()
start_date = st.sidebar.date_input("Study Start Date", st.session_state["start_date"], key="start_date")
end_date = st.sidebar.date_input("Study End Date", st.session_state["end_date"], key="end_date")

# ---------- Dynamic Countries Setup (start offsets, custom ramps, country dropout, label group) ----------
st.sidebar.header("Countries")
st.sidebar.caption("""
Add countries and set:
- Sites_Total
- Start_Offset_Months (0 = start of study; 2 = starts in 3rd month)
- Sites_Activate_per_Month (used if Custom_Ramp blank)
- Custom_Ramp (comma-separated monthly activations; overrides linear ramp)
- Recruit_per_site_per_month (patients/site/month)
- Dropout_Monthly_% (country-specific)
- Target_enrollment
- Label_Group (e.g., US, ROW; borrowing is ROW → US only when enabled)
""")

default_countries = pd.DataFrame({
    "Country": ["USA", "UK", "DE"],
    "Sites_Total": [24, 8, 10],
    "Start_Offset_Months": [0, 1, 2],
    "Sites_Activate_per_Month": [6, 3, 2],
    "Custom_Ramp": ["", "", ""],
    "Recruit_per_site_per_month": [1.0, 0.6, 0.5],
    "Dropout_Monthly_%": [3.0, 3.0, 3.0],
    "Target_enrollment": [150, 90, 60],
    "Label_Group": ["US", "ROW", "ROW"],
})

if "countries_df" not in st.session_state:
    st.session_state["countries_df"] = default_countries.copy()

countries_df = st.sidebar.data_editor(
    st.session_state["countries_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="countries_editor",
)

# Clean/validate
countries_df["Country"] = countries_df["Country"].astype(str).str.strip()
for col in ["Sites_Total", "Start_Offset_Months", "Sites_Activate_per_Month",
            "Recruit_per_site_per_month", "Dropout_Monthly_%", "Target_enrollment"]:
    countries_df[col] = pd.to_numeric(countries_df[col], errors="coerce").fillna(0.0)
countries_df["Custom_Ramp"] = countries_df["Custom_Ramp"].astype(str).fillna("")
countries_df["Label_Group"] = countries_df.get("Label_Group", "UNSET").astype(str).str.strip().replace({"": "UNSET"})
countries_df = countries_df[countries_df["Country"] != ""]
st.session_state["countries_df"] = countries_df.copy()

country_list = countries_df["Country"].tolist()

# Global enrollment cap (optional), default to sum of country targets, hard cap 1000
sum_country_targets = int(countries_df["Target_enrollment"].sum())
target_enrollment_global = st.sidebar.number_input(
    "Target total enrollment (global)",
    min_value=0, max_value=1000,
    value=min(sum_country_targets, 1000),
    step=1
)

# ---------- Study Arms (ratios only) ----------
st.sidebar.header("Study Arms & Randomization")
randomization = {
    'Active': st.sidebar.number_input("Randomization Ratio - Active", min_value=0.0, value=1.0, step=0.05, format="%.4f"),
    'Placebo': st.sidebar.number_input("Randomization Ratio - Placebo", min_value=0.0, value=1.0, step=0.05, format="%.4f"),
}
ratio_sum = sum(randomization.values())
if ratio_sum <= 0:
    st.sidebar.error("Randomization ratios must sum to more than 0. Increase at least one ratio.")
    st.stop()

buffer_percent = st.sidebar.slider("Buffer (%)", 0, 50, 15) / 100.0

# ---------- Editable Visit Schedule (UNIFIED: Kit_Type) ----------
st.header("Visit Schedule")
st.caption("""
Define per-visit timing and kit plan. **Kit_Type** is the single naming convention
(previously 'IP_Kit_Type' and 'Drug_Types'). If you need multiple kits in a visit,
enter them comma-separated in **Kit_Type** (e.g., "Bottle 50mg, Vial 100mg").
Day 0 = enrollment.
""")

# Choose how visits bucket into months (28-day = 4-week cycles; 30-day = calendar-ish)
bucket_days = st.sidebar.radio(
    'Visit month bucketing', [28, 30], index=0,
    format_func=lambda x: f"{x}-day month"
)

default_visits = pd.DataFrame({
    "Visit_Number": [1, 2, 3, 4],
    "Day_of_Visit": [0, 28, 56, 84],
    "Kit_Type": ["Bottle 50mg", "Bottle 50mg", "Bottle 50mg", "Bottle 50mg"],
    "Bottles_Active": [1.0, 1.0, 1.0, 1.0],
    "Bottles_Placebo": [1.0, 1.0, 1.0, 1.0],
})

def migrate_visit_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Kit_Type" in df.columns:
        if ("IP_Kit_Type" in df.columns) or ("Drug_Types" in df.columns):
            legacy = []
            if "IP_Kit_Type" in df.columns:
                legacy.append(df["IP_Kit_Type"].astype(str).fillna(""))
            if "Drug_Types" in df.columns:
                legacy.append(df["Drug_Types"].astype(str).fillna(""))
            merged = []
            for i in range(len(df)):
                parts = []
                for s in legacy:
                    val = s.iloc[i].strip()
                    if val and val.lower() != "nan":
                        parts.append(val)
                if df["Kit_Type"].iloc[i] and str(df["Kit_Type"].iloc[i]).lower() != "nan":
                    parts.append(str(df["Kit_Type"].iloc[i]).strip())
                merged.append(", ".join([p for p in parts if p]))
            df["Kit_Type"] = pd.Series(merged).replace("^\s*,\s*|\s*,\s*$", "", regex=True)
        return df

    kits = []
    if "IP_Kit_Type" in df.columns or "Drug_Types" in df.columns:
        n = len(df)
        for i in range(n):
            parts = []
            if "IP_Kit_Type" in df.columns:
                val = str(df["IP_Kit_Type"].iloc[i]) if i < len(df["IP_Kit_Type"]) else ""
                if val and val.lower() != "nan":
                    parts.append(val.strip())
            if "Drug_Types" in df.columns:
                val = str(df["Drug_Types"].iloc[i]) if i < len(df["Drug_Types"]) else ""
                if val and val.lower() != "nan":
                    parts.append(val.strip())
            kits.append(", ".join([p for p in parts if p]))
        df["Kit_Type"] = pd.Series(kits).replace("^\s*,\s*|\s*,\s*$", "", regex=True)
    else:
        df["Kit_Type"] = ""

    keep = ["Visit_Number", "Day_of_Visit", "Kit_Type", "Bottles_Active", "Bottles_Placebo"]
    for col in ["Visit_Number", "Day_of_Visit", "Bottles_Active", "Bottles_Placebo"]:
        if col not in df.columns:
            df[col] = default_visits[col]
    df = df[keep]
    return df

if "visit_df" not in st.session_state:
    st.session_state["visit_df"] = default_visits.copy()
else:
    st.session_state["visit_df"] = migrate_visit_df(st.session_state["visit_df"])

# ensure Month_Offset exists before display
st.session_state['visit_df']['Month_Offset'] = (st.session_state['visit_df']['Day_of_Visit'] // bucket_days).astype(int)
with st.form('visit_form', clear_on_submit=False):
    visit_df = st.data_editor(
        st.session_state['visit_df'],
        num_rows='dynamic',
        use_container_width=True,
        key='visit_editor',
        column_config={
            'Visit_Number': st.column_config.NumberColumn('Visit_Number', step=1, min_value=1),
            'Day_of_Visit': st.column_config.NumberColumn('Day_of_Visit', step=1, min_value=0),
            'Kit_Type': st.column_config.TextColumn('Kit_Type', help='Comma-separated if multiple kits per visit'),
            'Bottles_Active': st.column_config.NumberColumn('Bottles_Active', step=1, min_value=0),
            'Bottles_Placebo': st.column_config.NumberColumn('Bottles_Placebo', step=1, min_value=0),
            'Month_Offset': st.column_config.NumberColumn('Month_Offset', disabled=True, help=f'Auto from Day_of_Visit using {bucket_days}-day months'),
        },
    )
    visit_save = st.form_submit_button('Save Visit Schedule')

if 'visit_save' in locals() and visit_save:
    # Clean/validate visit table and recompute Month_Offset
    visit_df['Visit_Number'] = pd.to_numeric(visit_df['Visit_Number'], errors='coerce').fillna(0).astype(int)
    visit_df['Day_of_Visit'] = pd.to_numeric(visit_df['Day_of_Visit'], errors='coerce').fillna(0).astype(int)
    for col in ['Bottles_Active', 'Bottles_Placebo']:
        visit_df[col] = pd.to_numeric(visit_df[col], errors='coerce').fillna(0.0)
    visit_df['Kit_Type'] = visit_df['Kit_Type'].astype(str).fillna('')
    visit_df['Month_Offset'] = (visit_df['Day_of_Visit'] // bucket_days).astype(int)
    visit_df = visit_df.sort_values(by='Day_of_Visit').reset_index(drop=True)
    st.session_state['visit_df'] = visit_df.copy()
    st.session_state['run_ok'] = False
    st.success('Visit schedule saved. Click **Run Scenario / Refresh** to recompute.')

# ---------- SKU Catalog (UNIFIED: Kit_Type) ----------
st.header("SKU Catalog (Packaging & Dispensing)")
st.caption("""
Single naming convention: **Kit_Type** (Visit Schedule must reference only types found here).
""")

default_skus = pd.DataFrame({
    "Kit_Type": ["Bottle 50mg", "Bottle 100mg"],
    "Unit_Strength_mg": [50, 100],
    "Units_per_Bottle": [30, 30],
    "Form": ["Tablet", "Tablet"],
    "Notes": ["", ""],
})

def migrate_sku_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Kit_Type" not in df.columns and "SKU_Name" in df.columns:
        df = df.rename(columns={"SKU_Name": "Kit_Type"})
    for col in ["Unit_Strength_mg", "Units_per_Bottle"]:
        if col not in df.columns:
            df[col] = default_skus[col]
    for col in ["Form", "Notes"]:
        if col not in df.columns:
            df[col] = ""
    cols = ["Kit_Type", "Unit_Strength_mg", "Units_per_Bottle", "Form", "Notes"]
    for c in cols:
        if c not in df.columns:
            df[c] = default_skus[c] if c in default_skus.columns else ""
    return df[cols]

if "sku_df" not in st.session_state:
    st.session_state["sku_df"] = default_skus.copy()
else:
    st.session_state["sku_df"] = migrate_sku_df(st.session_state["sku_df"])

with st.form('sku_form', clear_on_submit=False):
    sku_df = st.data_editor(

    st.session_state["sku_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="sku_editor",

    )
    sku_form_save = st.form_submit_button('Save SKU Catalog')

if 'sku_form_save' in locals() and sku_form_save:
    st.session_state['sku_df'] = sku_df.copy()
    st.session_state['run_ok'] = False
    st.success('Save SKU Catalogd. Click **Run Scenario / Refresh** to recompute.')


# Cleanup
for col in ["Unit_Strength_mg", "Units_per_Bottle"]:
    sku_df[col] = pd.to_numeric(sku_df[col], errors="coerce").fillna(0.0)
sku_df["Kit_Type"] = sku_df["Kit_Type"].astype(str).str.strip()
sku_df["Form"] = sku_df["Form"].astype(str)
sku_df["Notes"] = sku_df["Notes"].astype(str)
st.session_state["sku_df"] = sku_df.copy()

# ---------- Supply Lots (Batches) WITH RELEASE DATE ----------
st.header("Supply Lots / Batches")
st.caption("Enter batches by **Kit_Type** with **Lot_ID**, **Qty_Bottles**, **Release_Date** (available-to-ship), **Expiry_Date** (last usable date), and **Label_Group**. Lots are available starting on Release_Date and expire at the end of their Expiry month. Borrowing rule: ROW → US only when enabled.")

default_lots = pd.DataFrame({
    "Lot_ID": ["L-001", "L-002", "L-003"],
    "Kit_Type": ["Bottle 50mg", "Bottle 100mg", "Bottle 50mg"],
    "Qty_Bottles": [2000, 1500, 1000],
    "Release_Date": [pd.to_datetime("2026-02-15"), pd.to_datetime("2026-05-01"), pd.to_datetime("2027-01-10")],
    "Expiry_Date": [pd.to_datetime("2027-06-30"), pd.to_datetime("2027-12-31"), pd.to_datetime("2027-09-30")],
    "Label_Group": ["US", "ROW", "ROW"],
    "Notes": ["", "", ""],
})

if "lots_df" not in st.session_state:
    st.session_state["lots_df"] = default_lots.copy()

with st.form('lots_form', clear_on_submit=False):
    lots_df = st.data_editor(

    st.session_state["lots_df"],
    num_rows="dynamic",
    use_container_width=True,
    key="lots_editor",

    )
    lots_form_save = st.form_submit_button('Save Lots / Batches')

if 'lots_form_save' in locals() and lots_form_save:
    st.session_state['lots_df'] = lots_df.copy()
    st.session_state['run_ok'] = False
    st.success('Save Lots / Batchesd. Click **Run Scenario / Refresh** to recompute.')


# Clean lots
lots_df["Lot_ID"] = lots_df["Lot_ID"].astype(str).str.strip()
lots_df["Kit_Type"] = lots_df["Kit_Type"].astype(str).str.strip()
lots_df["Qty_Bottles"] = pd.to_numeric(lots_df["Qty_Bottles"], errors="coerce").fillna(0.0)
lots_df["Release_Date"] = pd.to_datetime(lots_df["Release_Date"], errors="coerce")
lots_df["Expiry_Date"] = pd.to_datetime(lots_df["Expiry_Date"], errors="coerce")
lots_df["Label_Group"] = lots_df.get("Label_Group", "").astype(str).str.strip()
lots_df = lots_df.dropna(subset=["Kit_Type"])
st.session_state["lots_df"] = lots_df.copy()

# ---------- HARD GATE: Visit Schedule Kit_Types must exist in SKU Catalog ----------
def explode_visit_kits(kit_series: pd.Series) -> set[str]:
    kits = set()
    for raw in kit_series.fillna("").astype(str).tolist():
        parts = [p.strip() for p in raw.split(",") if p.strip() != ""]
        kits.update(parts)
    return kits

visit_kits = explode_visit_kits(visit_df.get("Kit_Type", pd.Series(dtype=str)))
sku_kits = set(sku_df["Kit_Type"].dropna().astype(str).str.strip().tolist())

missing = sorted(list(visit_kits - sku_kits))
if missing:
    st.error("⚠️ The following Kit_Type values are used in the Visit Schedule but not defined in the SKU Catalog:\n\n- " +
             "\n- ".join(missing) +
             "\n\nAdd them to the SKU Catalog (or update the Visit Schedule), then rerun.", icon="🚫")
    st.stop()

# ---------- Time Index ----------
study_months = pd.date_range(start_date, end_date, freq='MS')
n_months = len(study_months)
month_labels = study_months.strftime("%Y-%m")

# ---------- Countries & Activation ----------
def parse_custom_ramp(text):
    text = (text or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in text.split(",")]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except ValueError:
            vals.append(0.0)
    return vals

def build_sites_from_ramp(months, countries_df):
    df = pd.DataFrame({"Month": months.strftime("%Y-%m")})
    m = len(months)
    for _, row in countries_df.iterrows():
        c = row["Country"]
        total = float(row["Sites_Total"])
        start_off = int(row["Start_Offset_Months"])
        ramp = float(row["Sites_Activate_per_Month"])
        seq = parse_custom_ramp(row.get("Custom_Ramp", ""))
        active = [0.0] * m
        running = 0.0
        for i in range(m):
            if i < start_off:
                active[i] = 0.0
            else:
                j = i - start_off
                inc = seq[j] if j < len(seq) and len(seq) > 0 else ramp
                running = min(total, running + inc)
                active[i] = running
        df[f"{c}_Active_Sites"] = active
    return df

def ensure_sites_active_table(existing_df: pd.DataFrame, months: pd.Index, countries: list[str]) -> pd.DataFrame:
    base = pd.DataFrame({"Month": months.strftime("%Y-%m")})
    for c in countries:
        base[f"{c}_Active_Sites"] = 0.0
    if existing_df is None or existing_df.empty:
        return base
    df = existing_df.copy()
    if "Month" not in df.columns:
        df["Month"] = ""
    df["Month"] = df["Month"].astype(str)
    df = df[df["Month"].isin(base["Month"])]
    merged = base.merge(df, on="Month", how="left", suffixes=("", "_user"))
    for c in countries:
        col = f"{c}_Active_Sites"
        user_col = f"{col}_user"
        if user_col in merged.columns:
            merged[col] = pd.to_numeric(merged[user_col], errors="coerce").fillna(merged[col])
            merged.drop(columns=[user_col], inplace=True, errors="ignore")
        else:
            merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)
    keep_cols = ["Month"] + [f"{c}_Active_Sites" for c in countries]
    merged = merged[keep_cols]
    return merged

st.header("Active Sites by Month")
st.caption("Auto-generate from country ramps (start offset + custom or linear), or uncheck to edit manually.")

auto_generate = st.checkbox("Auto-generate from per-country ramp / custom sequence", value=True)

if auto_generate:
    sites_active_df = build_sites_from_ramp(study_months, countries_df)
else:
    existing = st.session_state.get("sites_active_df", None)
    sites_active_df = ensure_sites_active_table(existing, study_months, country_list)

try:
    from streamlit import column_config
    col_cfg = {"Month": column_config.Column(disabled=True)}
except Exception:
    col_cfg = None

with st.form('sites_form', clear_on_submit=False):
    sites_active_df = st.data_editor(

    sites_active_df,
    num_rows="dynamic",
    use_container_width=True,
    key="sites_active_editor",
    column_config=col_cfg,
    disabled=auto_generate

    )
    sites_form_save = st.form_submit_button('Save Active Sites by Month')

if 'sites_form_save' in locals() and sites_form_save:
    st.session_state['sites_active_df'] = sites_active_df.copy()
    st.session_state['run_ok'] = False
    st.success('Save Active Sites by Monthd. Click **Run Scenario / Refresh** to recompute.')


if not auto_generate:
    sites_active_df = ensure_sites_active_table(sites_active_df, study_months, country_list)
    for c in country_list:
        col = f"{c}_Active_Sites"
        sites_active_df[col] = pd.to_numeric(sites_active_df[col], errors="coerce").fillna(0.0)
    st.session_state["sites_active_df"] = sites_active_df.copy()


# ===== Run Scenario gate =====
if 'run_ok' not in st.session_state:
    st.session_state['run_ok'] = False
run_clicked = st.button('▶ Run Scenario / Refresh', type='primary')
if run_clicked:
    st.session_state['run_ok'] = True
if not st.session_state['run_ok']:
    st.info('Edits saved. Click **Run Scenario / Refresh** when you are ready to compute the forecast.')
    st.stop()

# ---------- Recruitment (raw) ----------
recruitment_raw = pd.DataFrame(index=study_months, columns=country_list, dtype=float)
recruitment_raw[:] = 0.0

for i, month_lbl in enumerate(month_labels):
    for c in country_list:
        col = f"{c}_Active_Sites"
        row = sites_active_df.loc[sites_active_df["Month"] == month_lbl]
        active_sites = float(row[col].values[0]) if (len(row) and col in sites_active_df.columns) else 0.0
        rate_row = countries_df.loc[countries_df["Country"] == c, "Recruit_per_site_per_month"]
        rate = float(rate_row.values[0]) if not rate_row.empty else 0.0
        recruitment_raw.loc[pd.to_datetime(month_lbl), c] = active_sites * rate

# ---------- Apply caps ----------
recruitment_capped = pd.DataFrame(index=study_months, columns=country_list, dtype=float)
recruitment_capped[:] = 0.0

remaining_country = {row["Country"]: float(row["Target_enrollment"]) for _, row in countries_df.iterrows()}
remaining_global = float(min(1000, target_enrollment_global))

for month in study_months:
    month_vals = {c: 0.0 for c in country_list}
    for c in country_list:
        proposed = float(recruitment_raw.loc[month, c])
        used = min(proposed, remaining_country[c])
        month_vals[c] = used
    month_sum = sum(month_vals.values())

    if remaining_global <= 0:
        for c in country_list:
            recruitment_capped.loc[month, c] = 0.0
        continue

    if month_sum > remaining_global and month_sum > 0:
        scale = remaining_global / month_sum
        for c in country_list:
            month_vals[c] *= scale
        month_sum = remaining_global

    for c in country_list:
        recruitment_capped.loc[month, c] = month_vals[c]
        remaining_country[c] = max(0.0, remaining_country[c] - month_vals[c])
    remaining_global = max(0.0, remaining_global - month_sum)

recruitment_capped["Total"] = recruitment_capped[country_list].sum(axis=1)

# Ensure numeric dtype
for col in recruitment_capped.columns:
    recruitment_capped[col] = pd.to_numeric(recruitment_capped[col], errors='coerce').fillna(0.0)

# ---------- New enrollments per country ----------
new_enroll_by_country = {c: recruitment_capped[c].values for c in country_list}

# ---------- Arm splits per country ----------
arm_ratio_sum = sum(randomization.values())
arm_props = {arm: (randomization[arm] / arm_ratio_sum) if arm_ratio_sum > 0 else 0.0 for arm in randomization}
new_enroll_arm_country = {
    c: {arm: new_enroll_by_country[c] * arm_props[arm] for arm in randomization}
    for c in country_list
}

# ---------- Country-specific dropout survival ----------
def survival_vector(dropout_percent, max_offset):
    p = max(0.0, min(100.0, float(dropout_percent))) / 100.0
    return np.array([(1.0 - p) ** k for k in range(max(1, max_offset + 1))], dtype=float)

visit_df["Month_Offset"] = (visit_df["Day_of_Visit"] // bucket_days).astype(int)
max_offset = int(visit_df["Month_Offset"].max()) if len(visit_df) else 0

survival_by_country = {
    c: survival_vector(countries_df.loc[countries_df["Country"] == c, "Dropout_Monthly_%"].values[0] if not countries_df.loc[countries_df["Country"] == c, "Dropout_Monthly_%"].empty else 0.0,
                       max_offset)
    for c in country_list
}

# Per-offset bottles per subject per arm
offset_bottles_arr = {
    arm: np.array([
        float(visit_df.loc[visit_df["Month_Offset"] == k, f"Bottles_{arm}"].sum())
        for k in range(max_offset + 1)
    ], dtype=float)
    for arm in randomization
}

# ---------- Bottles/vials demand via convolution (sum over countries) ----------
bottles_by_arm = {arm: np.zeros(len(study_months)) for arm in randomization}
# New: per-country breakdown
bottles_by_arm_country = {c: {arm: np.zeros(len(study_months)) for arm in randomization} for c in country_list}

for arm in randomization:
    for m, month_lbl in enumerate(month_labels):
        total_bottles = 0.0
        for c in country_list:
            surv_vec = survival_by_country[c]
            for k in range(max_offset + 1):
                t = m - k
                if t < 0:
                    continue
                cohort = new_enroll_arm_country[c][arm][t]
                alive = surv_vec[k] if k < len(surv_vec) else surv_vec[-1]
                bottles_ps = offset_bottles_arr[arm][k] if k < len(offset_bottles_arr[arm]) else 0.0
                total_bottles += cohort * alive * bottles_ps
                bottles_by_arm_country[c][arm][m] += cohort * alive * bottles_ps
        bottles_by_arm[arm][m] = math.ceil(total_bottles * (1.0 + buffer_percent)) if total_bottles > 0 else 0.0

# ---------- Outputs ----------
study_months_str = month_labels
bottles_df = pd.DataFrame({arm: bottles_by_arm[arm] for arm in randomization}, index=pd.to_datetime(study_months_str))

# Per-country arm bottles (rounded with buffer)
bottles_by_arm_country_df = {}
for c in country_list:
    tmp = {arm: [math.ceil(x * (1.0 + buffer_percent)) if x > 0 else 0.0 for x in bottles_by_arm_country[c][arm]] for arm in randomization}
    bottles_by_arm_country_df[c] = pd.DataFrame(tmp, index=pd.to_datetime(study_months_str))


# ---------- Demand by Kit_Type (allocate Bottles_* across Kit_Type entries) ----------
def parse_kits(s: str) -> list:
    if not isinstance(s, str):
        return []
    return [p.strip() for p in s.split(",") if p.strip() != ""]

def is_placebo_kit(name: str) -> bool:
    if not isinstance(name, str):
        return False
    u = name.upper()
    return ('PBO' in u) or ('PLACEBO' in u) or (u.startswith('PLA') and 'ACTIVE' not in u)

# Build per-offset bottles per subject per arm PER KIT
max_offset_local = int(visit_df["Month_Offset"].max()) if len(visit_df) else 0
offset_bottles_by_kit = {arm: {} for arm in randomization}
for arm in randomization:
    for k in range(max_offset_local + 1):
        rows = visit_df.loc[visit_df["Month_Offset"] == k]
        # Build kit counts only for the kits relevant to this arm
        kit_counts = {}
        total_kits = 0
        for _, r in rows.iterrows():
            kits_all = parse_kits(r.get("Kit_Type", ""))
            if arm.lower() == 'placebo':
                kits = [kt for kt in kits_all if is_placebo_kit(kt)]
            else:
                kits = [kt for kt in kits_all if not is_placebo_kit(kt)]
            if len(kits) == 0:
                continue
            total_kits += len(kits)
            for kt in kits:
                kit_counts[kt] = kit_counts.get(kt, 0) + 1
        bottles_ps = float(rows[f"Bottles_{arm}"].sum()) if len(rows) else 0.0
        if bottles_ps > 0 and total_kits > 0:
            for kt, cnt in kit_counts.items():
                share = bottles_ps * (cnt / total_kits)
                offset_bottles_by_kit[arm][(k, kt)] = offset_bottles_by_kit[arm].get((k, kt), 0.0) + share

kit_types_all = sorted(set([kt for _, kt in offset_bottles_by_kit['Active'].keys()]) | set([kt for _, kt in offset_bottles_by_kit['Placebo'].keys()]))
demand_kit_monthly = pd.DataFrame(0.0, index=pd.to_datetime(study_months_str), columns=kit_types_all)

# Demand by (Label_Group, Kit_Type)
label_map = dict(zip(countries_df["Country"], countries_df["Label_Group"]))
label_groups_sorted = sorted(countries_df["Label_Group"].unique().tolist())
demand_label_kit = pd.DataFrame(
    0.0,
    index=pd.to_datetime(study_months_str),
    columns=pd.MultiIndex.from_product([label_groups_sorted, kit_types_all], names=["Label_Group","Kit_Type"])
)

for arm in randomization:
    for m, month_lbl in enumerate(study_months_str):
        month_total_by_kit = {}
        for c in country_list:
            surv_vec = survival_by_country[c]
            for k in range(max_offset_local + 1):
                t = m - k
                if t < 0:
                    continue
                cohort = new_enroll_arm_country[c][arm][t]
                alive = surv_vec[k] if k < len(surv_vec) else surv_vec[-1]
                for (kk, kt), bottles_ps in offset_bottles_by_kit[arm].items():
                    if kk != k:
                        continue
                    month_total_by_kit[kt] = month_total_by_kit.get(kt, 0.0) + cohort * alive * bottles_ps
                    lg = label_map.get(c, 'UNSET')
                    prev = demand_label_kit.loc[pd.to_datetime(month_lbl), (lg, kt)]
                    demand_label_kit.loc[pd.to_datetime(month_lbl), (lg, kt)] = prev + cohort * alive * bottles_ps
        for kt, val in month_total_by_kit.items():
            demand_kit_monthly.loc[pd.to_datetime(month_lbl), kt] += math.ceil(val * (1.0 + buffer_percent))
        row_idx = pd.to_datetime(month_lbl)
        for lg in label_groups_sorted:
            for kt2 in kit_types_all:
                raw_val = demand_label_kit.loc[row_idx, (lg, kt2)]
                demand_label_kit.loc[row_idx, (lg, kt2)] = math.ceil(raw_val * (1.0 + buffer_percent)) if raw_val > 0 else 0.0

summary_cols = {"Month": study_months_str}
for c in country_list:
    summary_cols[f"{c} Enrolled (cum)"] = recruitment_capped[c].cumsum().round(0)
summary_cols["Total Enrolled (cum)"] = recruitment_capped["Total"].cumsum().round(0)
# Replace arm-based totals with per-Kit_Type demand columns
for kt in demand_kit_monthly.columns:
    summary_cols[str(kt)] = demand_kit_monthly[kt].round(0)

summary_df = pd.DataFrame(summary_cols)

st.subheader("Forecast Summary")
st.dataframe(summary_df, use_container_width=True)

# ---------- Supply Allocation (FIFO by release then expiry) ----------
allow_borrow = st.checkbox("Allow cross-label borrowing (ROW → US only)", value=True)
months_idx = pd.to_datetime(study_months_str)
month_end_idx = months_idx + pd.offsets.MonthEnd(0)

lots_clean = lots_df.dropna(subset=["Kit_Type", "Qty_Bottles", "Expiry_Date", "Release_Date"]).copy()
lots_clean["Expiry_MonthEnd"] = pd.to_datetime(lots_clean["Expiry_Date"]) + pd.offsets.MonthEnd(0)
lots_clean["Release_Date"] = pd.to_datetime(lots_clean["Release_Date"])
# sort by label, kit, Release first (earlier releases first), then Expiry, then Lot_ID
lots_clean = lots_clean.sort_values(["Label_Group","Kit_Type","Release_Date","Expiry_MonthEnd","Lot_ID"])

# Allocation results by (Label_Group, Kit_Type)
supply_onhand_labelkit = pd.DataFrame(0.0, index=months_idx, columns=demand_label_kit.columns)
shortfall_labelkit = pd.DataFrame(0.0, index=months_idx, columns=demand_label_kit.columns)

# Also aggregate across labels back to kit-only (for existing displays)
supply_onhand_monthly = pd.DataFrame(0.0, index=months_idx, columns=demand_kit_monthly.columns)
shortfall_monthly = pd.DataFrame(0.0, index=months_idx, columns=demand_kit_monthly.columns)

def norm_label(x):
    return str(x).strip().upper()

for (lg, kt) in demand_label_kit.columns:
    lot_rows_match = lots_clean[(lots_clean["Label_Group"] == lg) & (lots_clean["Kit_Type"] == kt)].copy()
    remaining_match = list(lot_rows_match["Qty_Bottles"].tolist())
    expiries_match = list(lot_rows_match["Expiry_MonthEnd"].tolist())
    releases_match = list(lot_rows_match["Release_Date"].tolist())

    # One-way borrowing: only ROW -> US when enabled
    if allow_borrow and norm_label(lg) == "US":
        lot_rows_borrow = lots_clean[(lots_clean["Label_Group"].str.upper() == "ROW") & (lots_clean["Kit_Type"] == kt)].copy()
        remaining_borrow = list(lot_rows_borrow["Qty_Bottles"].tolist())
        expiries_borrow = list(lot_rows_borrow["Expiry_MonthEnd"].tolist())
        releases_borrow = list(lot_rows_borrow["Release_Date"].tolist())
    else:
        lot_rows_borrow = pd.DataFrame(columns=lots_clean.columns)
        remaining_borrow, expiries_borrow, releases_borrow = [], [], []

    for mi, mdate in enumerate(month_end_idx):
        demand = float(demand_label_kit.iloc[mi].loc[(lg, kt)])
        supplied = 0.0
        # 1) consume from matching label (only lots released by now and not expired)
        for i in range(len(remaining_match)):
            if remaining_match[i] <= 0:
                continue
            if (pd.isna(expiries_match[i]) or mdate <= expiries_match[i]) and (pd.isna(releases_match[i]) or releases_match[i] <= mdate):
                take = min(remaining_match[i], demand - supplied)
                if take > 0:
                    remaining_match[i] -= take
                    supplied += take
            if supplied >= demand:
                break
        # 2) borrow from ROW to US if needed
        if allow_borrow and norm_label(lg) == "US" and supplied < demand:
            for i in range(len(remaining_borrow)):
                if remaining_borrow[i] <= 0:
                    continue
                if (pd.isna(expiries_borrow[i]) or mdate <= expiries_borrow[i]) and (pd.isna(releases_borrow[i]) or releases_borrow[i] <= mdate):
                    take = min(remaining_borrow[i], demand - supplied)
                    if take > 0:
                        remaining_borrow[i] -= take
                        supplied += take
                if supplied >= demand:
                    break

        supply_onhand_labelkit.iloc[mi].loc[(lg, kt)] = supplied
        shortfall_labelkit.iloc[mi].loc[(lg, kt)] = max(0.0, demand - supplied)

# Aggregate to kit level for legacy displays
supply_onhand_monthly = supply_onhand_labelkit.groupby(level=1, axis=1).sum()
shortfall_monthly = shortfall_labelkit.groupby(level=1, axis=1).sum()

# Convenience totals
supply_total = supply_onhand_monthly.sum(axis=1).rename("Total_Supply_Bottles")
demand_total = demand_kit_monthly.sum(axis=1).rename("Total_Demand_Bottles")
short_total = shortfall_monthly.sum(axis=1).rename("Total_Shortfall_Bottles")

# ---------- Supply vs Demand Charts (with legend) ----------
st.subheader("Supply vs Demand — Total Bottles")
_total_sd = pd.concat([
    demand_total.rename("Demand"),
    supply_total.rename("Supply")
], axis=1).reset_index().rename(columns={"index":"Month"})
_total_sd["Month"] = pd.to_datetime(_total_sd["Month"])

_total_long = _total_sd.melt(id_vars=["Month"], var_name="Series", value_name="Bottles")

total_chart = alt.Chart(_total_long).mark_line().encode(
    x=alt.X("Month:T", title="Month"),
    y=alt.Y("Bottles:Q", title="Bottles"),
    color=alt.Color("Series:N", title=""),
    tooltip=["Month:T","Series:N","Bottles:Q"]
).properties(title="Total Supply vs Total Demand (Bottles)")

st.altair_chart(total_chart, use_container_width=True)

# ---------- Per Label_Group: Supply vs Demand (one chart per group) ----------
st.subheader("Supply vs Demand by Label Group")
# Sum across kit types for each Label_Group
demand_by_label = demand_label_kit.groupby(level=0, axis=1).sum()
supply_by_label = supply_onhand_labelkit.groupby(level=0, axis=1).sum()
short_by_label = shortfall_labelkit.groupby(level=0, axis=1).sum()

for _lg in label_groups_sorted:
    if _lg not in demand_by_label.columns or _lg not in supply_by_label.columns:
        continue
    _df = pd.concat([
        demand_by_label[_lg].rename("Demand"),
        supply_by_label[_lg].rename("Supply")
    ], axis=1).reset_index().rename(columns={"index":"Month"})
    _df["Month"] = pd.to_datetime(_df["Month"])
    _long = _df.melt(id_vars=["Month"], var_name="Series", value_name="Bottles")

    _chart = alt.Chart(_long).mark_line().encode(
        x=alt.X("Month:T", title="Month"),
        y=alt.Y("Bottles:Q", title="Bottles"),
        color=alt.Color("Series:N", title=""),
        tooltip=["Month:T","Series:N","Bottles:Q"]
    ).properties(title=f"{_lg}: Supply vs Demand (Bottles)")

    st.altair_chart(_chart, use_container_width=True)


# ---------- Supply vs Demand Charts ----------
st.subheader("Supply vs Demand — Total Bottles")
_supply_demand_df = pd.concat([demand_total, supply_total, short_total], axis=1).reset_index()
_supply_demand_df.rename(columns={"index":"Month"}, inplace=True)
_supply_demand_df["Month"] = pd.to_datetime(_supply_demand_df["Month"])

line_base = alt.Chart(_suppy_base := _supply_demand_df).encode(
    x=alt.X("Month:T", title="Month")
)

supply_line = line_base.mark_line().encode(
    y=alt.Y("Total_Supply_Bottles:Q", title="Bottles"),
    color=alt.value("#1f77b4"),
    tooltip=["Month:T","Total_Supply_Bottles:Q"]
)

demand_line = line_base.mark_line().encode(
    y=alt.Y("Total_Demand_Bottles:Q"),
    color=alt.value("#ff7f0e"),
    tooltip=["Month:T","Total_Demand_Bottles:Q"]
)

short_bar = alt.Chart(_suppy_base).mark_bar(opacity=0.4).encode(
    x="Month:T",
    y=alt.Y("Total_Shortfall_Bottles:Q", title="Shortfall Bottles"),
    color=alt.value("#d62728"),
    tooltip=["Month:T","Total_Shortfall_Bottles:Q"]
)

chart = alt.layer(supply_line, demand_line).resolve_scale(y='shared').properties(
    title="Total Supply vs Total Demand (Bottles)"
)

st.altair_chart(chart, use_container_width=True)

st.subheader("Shortfall by Month — Bottles")
st.altair_chart(short_bar.properties(title="Shortfall Bottles per Month"), use_container_width=True)


# UI: per-label per-kit shortage alerts with correct months-until
study_start_month = pd.to_datetime(study_months_str[0]).to_period('M')
any_short = False
for (lg, kt) in shortfall_labelkit.columns:
    s = shortfall_labelkit[(lg, kt)]
    if (s > 0).any():
        any_short = True
        first = s[s > 0].index[0]
        months_until = (first.to_period('M') - study_start_month).n
        st.error(f"{lg} / {kt}: shortfall begins in {months_until} month(s) (earliest {first.strftime('%Y-%m')}; short {int(s.loc[first])} bottles).")
if not any_short:
    st.success("Supply covers demand for all months in the horizon (all label groups).")

# Enrollment line chart (cumulative total enrollment over time)
st.subheader("Cumulative Enrollment Over Time")
enrollment_cum = recruitment_capped["Total"].cumsum()
st.line_chart(enrollment_cum.rename_axis("Month"))

# ---------- Shortfall Details DataFrame (for UI + Excel) ----------
short_table = shortfall_labelkit.copy()
short_table.index.name = "Month"
shortfall_details_df = short_table.stack(level=[0,1]).reset_index()
shortfall_details_df.columns = ["Month", "Label_Group", "Kit_Type", "Shortfall_Bottles"]
shortfall_details_df = shortfall_details_df[shortfall_details_df["Shortfall_Bottles"] > 0].sort_values(["Month", "Label_Group", "Kit_Type"])

# Tabs
tabs = st.tabs([
    "Countries Config",
    "Visit Schedule",
    "SKU Catalog",
    "Active Sites by Month",
    "Recruitment (raw/month)",
    "Recruitment (capped/month)",
    "Bottles (rounded) by arm",
    "Demand by Kit_Type (monthly)",
    "Demand by Label & Kit (monthly)",
    "Supply vs Demand (bottles)",
    "Lots",
    "Shortfall Details"
])
with tabs[0]:
    st.dataframe(countries_df, use_container_width=True)
with tabs[1]:
    st.dataframe(visit_df, use_container_width=True)
with tabs[2]:
    st.dataframe(sku_df, use_container_width=True)
with tabs[3]:
    st.dataframe(sites_active_df, use_container_width=True)
with tabs[4]:
    st.dataframe(recruitment_raw, use_container_width=True)
with tabs[5]:
    st.dataframe(recruitment_capped, use_container_width=True)
with tabs[6]:
    st.dataframe(bottles_df, use_container_width=True)
with tabs[7]:
    st.dataframe(demand_kit_monthly, use_container_width=True)
with tabs[8]:
    st.dataframe(demand_label_kit, use_container_width=True)
with tabs[9]:
    st.write("Supply (allocated) vs Demand — bottles per month")
    st.dataframe(pd.concat([demand_total, supply_total, short_total], axis=1), use_container_width=True)
with tabs[10]:
    lots_df = lots_df.copy()
    if 'Release_Date' in lots_df.columns:
        lots_df['Release_Date'] = pd.to_datetime(lots_df['Release_Date']).dt.strftime('%Y-%m-%d')
    if 'Expiry_Date' in lots_df.columns:
        lots_df['Expiry_Date'] = pd.to_datetime(lots_df['Expiry_Date']).dt.strftime('%Y-%m-%d')
    st.dataframe(lots_df, use_container_width=True)

with tabs[-1]:
    st.subheader('Monthly Kits by Arm per Country')
    for c in country_list:
        st.markdown(f'**{c}**')
        dfc = bottles_by_arm_country_df.get(c)
        if dfc is not None:
            st.dataframe(dfc.astype(int), use_container_width=True)
        else:
            st.info('No data for this country.')
with tabs[11]:
    st.dataframe(shortfall_details_df, use_container_width=True)

# Export
@st.cache_data
def convert_to_excel(dfs: dict):
    from io import BytesIO
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for name, df in dfs.items():
            df.to_excel(writer, sheet_name=name, index=True)
    return output.getvalue()

excel_bytes = convert_to_excel({
    "Lots": lots_df,
    "Demand_by_Kit": demand_kit_monthly,
    "Demand_by_Label_Kit": demand_label_kit,
    "Supply_Allocated": supply_onhand_monthly,
    "Shortfall": shortfall_monthly,
    "Shortfall_Details": shortfall_details_df,
    "Summary": summary_df,
    "Countries_Config": countries_df,
    "Visit_Schedule": visit_df,
    "SKU_Catalog": sku_df,
    "Sites_Active_by_Month": sites_active_df,
    "Recruitment_raw": recruitment_raw,
    "Recruitment_capped": recruitment_capped,
    "Bottles_by_arm": bottles_df,
})

st.download_button(
    label="📥 Download Forecast (Excel)",
    data=excel_bytes,
    file_name="ip_forecast.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)



# ================= Scenario Save/Load (JSON) =================
st.sidebar.markdown("---")
st.sidebar.subheader("Scenario")

_uploaded = st.sidebar.file_uploader("Load scenario (.json)", type=["json"])
if _uploaded is not None:
    try:
        _payload = json.load(_uploaded)
        restore_config(_payload)
        st.sidebar.success("Scenario loaded. Click again if the UI hasn't refreshed.")
        st.rerun() if hasattr(st, 'rerun') else (st.experimental_rerun() if hasattr(st, 'experimental_rerun') else None)
    except Exception as _e:
        st.sidebar.error(f"Could not load scenario: {_e}")

_scn_name = st.sidebar.text_input("Scenario name", value="my_scenario")
if st.sidebar.button("💾 Save scenario to JSON"):
    try:
        _blob = json.dumps(serialize_config(), indent=2).encode("utf-8")
        st.sidebar.download_button(
            "Download scenario", _blob, file_name=f"{_scn_name}.json", mime="application/json", use_container_width=True
        )
    except Exception as _e:
        st.sidebar.error(f"Could not serialize scenario: {_e}")