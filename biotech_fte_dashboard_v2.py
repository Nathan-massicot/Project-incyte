


import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -------------------------
# Page setup
# -------------------------
st.set_page_config(
    page_title="PMO Planner — v2",

    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("PMO Planner — v2")
st.caption("Weekly aggregation • Constant FTE across tasks • Capacity at Department, Role, and Department→Role • Optional CPM ")

# -------------------------
# Data & parameters (no sidebar, no uploads)
# -------------------------
tasks_df = pd.read_csv("data/DayDataMaa_dashboard_ready.csv")
depts_df = pd.read_csv("data/Regulatory_Departments_and_Teams.csv")
hr_df    = pd.read_csv("data/HR_Capacity.csv")
rates_df = pd.read_csv("data/Rates.csv")

# Project anchor & aggregation (hardcoded or set here)
j0_date = pd.to_datetime("2026-06-30")
agg_freq = "Weekly"
time_freq = "W-MON" if agg_freq == "Weekly" else "MS"
# Per-person cap (applied when headcount is provided)
per_person_cap = 0.8

# Gantt chart filter section, now in an expander above the Gantt chart
with st.expander("🔎 Gantt Filters", expanded=False):
    available_departments = sorted(tasks_df["Department"].dropna().unique())
    sel_departments = st.multiselect("Filter by Department", available_departments, default=available_departments)
    available_roles = sorted(tasks_df["Role"].dropna().unique())
    sel_roles = st.multiselect("Filter by Role", available_roles, default=available_roles)
    # Market and Language filters if present
    available_markets = sorted(tasks_df["Market"].dropna().unique()) if "Market" in tasks_df.columns else []
    if available_markets:
        sel_markets = st.multiselect("Filter by Market", available_markets, default=available_markets)
    else:
        sel_markets = []
    available_langs = sorted(tasks_df["Language"].dropna().unique()) if "Language" in tasks_df.columns else []
    if available_langs:
        sel_langs = st.multiselect("Filter by Language", available_langs, default=available_langs)
    else:
        sel_langs = []

# -------------------------
# Helpers
# -------------------------
def apply_filters(df, departments=None, roles=None, markets=None, langs=None):
    if df is None or df.empty:
        return df
    out = df.copy()
    if departments:
        out = out[out["Department"].isin(departments)]
    if roles:
        out = out[out["Role"].isin(roles)]
    if markets and "Market" in out.columns:
        out = out[out["Market"].isin(markets)]
    if langs and "Language" in out.columns:
        out = out[out["Language"].isin(langs)]
    return out

def compute_dates(df: pd.DataFrame, j0: pd.Timestamp) -> pd.DataFrame:
    """Compute Start_Date & Finish_Date from J0, Prep_Days, Finish_Days_Before_J0, then adjust with predecessors & lag if provided."""
    d = df.copy()

    # Base dates from anchor
    d["Finish_Date"] = pd.to_datetime(j0) - pd.to_timedelta(d["Finish_Days_Before_J0"], unit="D")
    d["Start_Date"] = d["Finish_Date"] - pd.to_timedelta(d["Prep_Days"], unit="D")

    # Normalize FTE columns
    if "Allocated_FTE" not in d.columns and "Allocated_FTE_Days" in d.columns:
        d["Allocated_FTE"] = d["Allocated_FTE_Days"] / d["Prep_Days"].replace(0, np.nan)
    if "Allocated_FTE_Days" not in d.columns and "Allocated_FTE" in d.columns:
        d["Allocated_FTE_Days"] = d["Allocated_FTE"] * d["Prep_Days"]

    # Predecessor handling (optional)
    if "Predecessor_Step" in d.columns:
        for _ in range(3):
            for idx, row in d.iterrows():
                pred = row.get("Predecessor_Step")
                if pd.isna(pred) or str(pred).strip() == "":
                    continue
                try:
                    preds = [int(x) for x in str(pred).replace(";",",").split(",") if str(x).strip().isdigit()]
                except Exception:
                    preds = []
                if not preds:
                    continue

                lag_raw = row.get("Lag_Days", 0)
                if isinstance(lag_raw, (int, float, np.integer, np.floating)):
                    lags = [int(lag_raw)] * len(preds)
                else:
                    try:
                        lags = [int(x) for x in str(lag_raw).replace(";",",").split(",")]
                    except Exception:
                        lags = [0] * len(preds)
                if len(lags) < len(preds):
                    lags += [0] * (len(preds) - len(lags))
                lags = lags[:len(preds)]

                pred_finishes = []
                for p, lg in zip(preds, lags):
                    m = d["Step_No"] == p
                    if not m.any():
                        continue
                    fin = pd.to_datetime(d.loc[m, "Finish_Date"]).max() + pd.to_timedelta(lg, unit="D")
                    pred_finishes.append(fin)
                if pred_finishes:
                    earliest = max(pred_finishes)
                    if pd.to_datetime(row["Start_Date"]) < earliest:
                        d.loc[idx, "Start_Date"] = earliest
                        d.loc[idx, "Finish_Date"] = earliest + pd.to_timedelta(int(row["Prep_Days"]), unit="D")

    return d

def explode_daily(dated_df: pd.DataFrame) -> pd.DataFrame:
    """Row-per-day with constant FTE across duration."""
    rows = []
    for _, r in dated_df.iterrows():
        start = pd.to_datetime(r["Start_Date"]) if not pd.isna(r.get("Start_Date")) else None
        finish = pd.to_datetime(r["Finish_Date"]) if not pd.isna(r.get("Finish_Date")) else None
        if start is None or finish is None or finish < start:
            continue
        duration = (finish - start).days or 1
        fte = float(r.get("Allocated_FTE", 0.0))
        for off in range(duration):
            day = (start + pd.Timedelta(days=off)).normalize()
            rows.append({
                "Date": day,
                "Department": r.get("Department"),
                "Role": r.get("Role"),
                "Document": r.get("Document"),
                "Step_No": r.get("Step_No"),
                "Market": r.get("Market", np.nan),
                "Language": r.get("Language", np.nan),
                "FTE": fte,
            })
    return pd.DataFrame(rows, columns=["Date","Department","Role","Document","Step_No","Market","Language","FTE"])

def aggregate_time(daily: pd.DataFrame, time_freq: str, level: str):
    if daily is None or daily.empty:
        return pd.DataFrame()
    df = daily.copy()
    df["PeriodStart"] = pd.to_datetime(df["Date"]).dt.to_period(time_freq).dt.start_time
    if level == "Department":
        grp = ["PeriodStart","Department"]
    elif level == "Role":
        grp = ["PeriodStart","Role"]
    else:  # Dept→Role
        grp = ["PeriodStart","Department","Role"]
    agg = df.groupby(grp, dropna=True, as_index=False)["FTE"].sum()
    return agg

def make_capacity_frames(tasks):
    """Build blank capacity tables for Department, Role, and Dept→Role with Headcount + Available_FTE columns."""
    if tasks is None or tasks.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    dept = pd.DataFrame({"Department": sorted(tasks["Department"].dropna().unique())})
    dept["Headcount"] = 0.0
    dept["Available_FTE"] = 0.0
    role = pd.DataFrame({"Role": sorted(tasks["Role"].dropna().unique())})
    role["Headcount"] = 0.0
    role["Available_FTE"] = 0.0
    pair = tasks[["Department","Role"]].dropna().drop_duplicates().sort_values(["Department","Role"]).reset_index(drop=True)
    pair["Headcount"] = 0.0
    pair["Available_FTE"] = 0.0
    return dept, role, pair

def apply_hr_capacity(cap_df, dept_cap, role_cap, pair_cap, per_person_cap):
    """Merge HR capacity CSV into editors; compute Available_FTE from Headcount if not set (Headcount × cap)."""
    if cap_df is None or cap_df.empty:
        return dept_cap, role_cap, pair_cap
    df = cap_df.copy()
    cols = [c.lower() for c in df.columns]
    df.columns = cols

    def col(df, name):
        for c in df.columns:
            if c.lower() == name.lower():
                return c
        return None

    # Department level
    if col(df,"department") and not col(df,"role"):
        d_rows = df[df[col(df,"department")].notna()]
        tmp = d_rows.rename(columns={col(d_rows,"department"):"Department"}).copy()
        if col(tmp,"available_fte"):
            tmp["Available_FTE"] = pd.to_numeric(tmp[col(tmp,"available_fte")], errors="coerce")
        if col(tmp,"headcount"):
            tmp["Headcount"] = pd.to_numeric(tmp[col(tmp,"headcount")], errors="coerce")
        # compute missing
        if "Available_FTE" in tmp.columns and "Headcount" in tmp.columns:
            tmp["Available_FTE"] = tmp["Available_FTE"].fillna(tmp["Headcount"] * per_person_cap)
        elif "Headcount" in tmp.columns:
            tmp["Available_FTE"] = tmp["Headcount"] * per_person_cap
        elif "Available_FTE" in tmp.columns:
            tmp["Headcount"] = (tmp["Available_FTE"] / per_person_cap).round(2)
        dept_cap = dept_cap.drop(columns=["Headcount","Available_FTE"]).merge(tmp[["Department","Headcount","Available_FTE"]], on="Department", how="left").fillna({"Headcount":0.0,"Available_FTE":0.0})

    # Role level
    if col(df,"role") and not col(df,"department"):
        r_rows = df[df[col(df,"role")].notna()]
        tmp = r_rows.rename(columns={col(r_rows,"role"):"Role"}).copy()
        if col(tmp,"available_fte"):
            tmp["Available_FTE"] = pd.to_numeric(tmp[col(tmp,"available_fte")], errors="coerce")
        if col(tmp,"headcount"):
            tmp["Headcount"] = pd.to_numeric(tmp[col(tmp,"headcount")], errors="coerce")
        if "Available_FTE" in tmp.columns and "Headcount" in tmp.columns:
            tmp["Available_FTE"] = tmp["Available_FTE"].fillna(tmp["Headcount"] * per_person_cap)
        elif "Headcount" in tmp.columns:
            tmp["Available_FTE"] = tmp["Headcount"] * per_person_cap
        elif "Available_FTE" in tmp.columns:
            tmp["Headcount"] = (tmp["Available_FTE"] / per_person_cap).round(2)
        role_cap = role_cap.drop(columns=["Headcount","Available_FTE"]).merge(tmp[["Role","Headcount","Available_FTE"]], on="Role", how="left").fillna({"Headcount":0.0,"Available_FTE":0.0})

    # Pair level
    if col(df,"department") and col(df,"role"):
        p_rows = df[df[col(df,"department")].notna() & df[col(df,"role")].notna()]
        tmp = p_rows.rename(columns={col(p_rows,"department"):"Department", col(p_rows,"role"):"Role"}).copy()
        if col(tmp,"available_fte"):
            tmp["Available_FTE"] = pd.to_numeric(tmp[col(tmp,"available_fte")], errors="coerce")
        if col(tmp,"headcount"):
            tmp["Headcount"] = pd.to_numeric(tmp[col(tmp,"headcount")], errors="coerce")
        if "Available_FTE" in tmp.columns and "Headcount" in tmp.columns:
            tmp["Available_FTE"] = tmp["Available_FTE"].fillna(tmp["Headcount"] * per_person_cap)
        elif "Headcount" in tmp.columns:
            tmp["Available_FTE"] = tmp["Headcount"] * per_person_cap
        elif "Available_FTE" in tmp.columns:
            tmp["Headcount"] = (tmp["Available_FTE"] / per_person_cap).round(2)
        pair_cap = pair_cap.drop(columns=["Headcount","Available_FTE"]).merge(tmp[["Department","Role","Headcount","Available_FTE"]], on=["Department","Role"], how="left").fillna({"Headcount":0.0,"Available_FTE":0.0})

    return dept_cap, role_cap, pair_cap

def merge_capacity(agg_df, level, dept_cap, role_cap, pair_cap, per_person_cap):
    if agg_df is None or agg_df.empty:
        return pd.DataFrame()
    df = agg_df.copy()

    if level == "Department":
        cap = dept_cap.copy()
        cap["key"] = cap["Department"]
        df["key"] = df["Department"]
    elif level == "Role":
        cap = role_cap.copy()
        cap["key"] = cap["Role"]
        df["key"] = df["Role"]
    else:
        cap = pair_cap.copy()
        cap["key"] = cap["Department"].astype(str) + " | " + cap["Role"].astype(str)
        df["key"] = df["Department"].astype(str) + " | " + df["Role"].astype(str)

    cap["Available_FTE"] = cap["Available_FTE"].fillna(0.0)
    mask = (cap["Available_FTE"] <= 0) & (cap["Headcount"] > 0)
    cap.loc[mask, "Available_FTE"] = cap.loc[mask, "Headcount"] * per_person_cap

    merged = df.merge(cap[["key","Available_FTE"]], on="key", how="left")
    merged["Available_FTE"] = merged["Available_FTE"].fillna(0.0)

    def period_days(start, label):
        start = pd.to_datetime(start)
        if label.startswith("W"):
            end = start + pd.offsets.Week(weekday=0)  # next Monday
        else:
            end = start + pd.offsets.MonthBegin(1)
        return max((end - start).days, 1)

    merged["Period_Days"] = merged["PeriodStart"].apply(lambda d: period_days(d, time_freq))
    merged["Capacity_FTE_Period"] = merged["Available_FTE"] * merged["Period_Days"]
    return merged

def compute_cpm(tasks_df: pd.DataFrame):
    if tasks_df is None or tasks_df.empty or "Prep_Days" not in tasks_df.columns:
        return pd.DataFrame()

    df = tasks_df.copy()
    df["Duration"] = pd.to_numeric(df["Prep_Days"], errors="coerce").fillna(0).astype(int)

    preds_map = {}
    for _, r in df.iterrows():
        pred_raw = r.get("Predecessor_Step")
        preds = []
        if pd.notna(pred_raw) and str(pred_raw).strip() != "":
            try:
                preds = [int(x) for x in str(pred_raw).replace(";",",").split(",") if str(x).strip().isdigit()]
            except Exception:
                preds = []
        preds_map[int(r["Step_No"])] = preds

    ES, EF = {}, {}
    steps = list(df["Step_No"].astype(int))
    for _ in range(len(steps)):
        progress = False
        for s in steps:
            if s in ES:
                continue
            preds = preds_map.get(s, [])
            if all((p in EF) for p in preds):
                es = max([EF[p] for p in preds], default=0)
                dur = int(df.loc[df["Step_No"]==s, "Duration"].iloc[0])
                ES[s] = es
                EF[s] = es + dur
                progress = True
        if not progress:
            break

    project_duration = max(EF.values()) if EF else 0

    LS, LF = {}, {}
    for _ in range(len(steps)):
        progress = False
        for s in reversed(steps):
            if s in LS:
                continue
            succs = [k for k,v in preds_map.items() if s in v]
            if not succs:
                LF[s] = project_duration
                LS[s] = LF[s] - int(df.loc[df["Step_No"]==s, "Duration"].iloc[0])
                progress = True
            elif all((ss in LS) for ss in succs):
                lf = min([LS[ss] for ss in succs])
                dur = int(df.loc[df["Step_No"]==s, "Duration"].iloc[0])
                LF[s] = lf
                LS[s] = lf - dur
                progress = True
        if not progress:
            break

    out = df[["Step_No","Document","Department","Role","Duration"]].copy()
    out["ES"] = out["Step_No"].map(ES).fillna(0).astype(int)
    out["EF"] = out["Step_No"].map(EF).fillna(out["ES"] + out["Duration"]).astype(int)
    out["LS"] = out["Step_No"].map(LS).fillna(out["ES"]).astype(int)
    out["LF"] = out["Step_No"].map(LF).fillna(out["EF"]).astype(int)
    out["Slack"] = (out["LS"] - out["ES"]).astype(int)
    out["Critical"] = out["Slack"] == 0
    return out

def attach_rates(df, level, rates_df):
    if df is None or df.empty:
        return df
    if rates_df is None or rates_df.empty:
        df["Rate_per_FTE_Day"] = 0.0
        df["Budget"] = 0.0
        return df

    rates = rates_df.copy()

    if level == "Department" and "Department" in rates.columns:
        merged = df.merge(rates[["Department","Rate_per_FTE_Day"]], on="Department", how="left")
    elif level == "Role" and "Role" in rates.columns:
        merged = df.merge(rates[["Role","Rate_per_FTE_Day"]], on="Role", how="left")
    elif level == "Department → Role" and {"Department","Role"}.issubset(set(rates.columns)):
        merged = df.merge(rates[["Department","Role","Rate_per_FTE_Day"]], on=["Department","Role"], how="left")
    else:
        merged = df.copy()
        merged["Rate_per_FTE_Day"] = np.nan

    merged["Rate_per_FTE_Day"] = pd.to_numeric(merged["Rate_per_FTE_Day"], errors="coerce").fillna(0.0)
    merged["Budget"] = merged["FTE"] * merged.get("Period_Days", 1) * merged["Rate_per_FTE_Day"]
    return merged

# -------------------------
# Data checks
# -------------------------
if tasks_df is None:
    st.error("❌ Tasks CSV not found. Place it in ./data as 'DayDataMaa_dashboard_ready.csv' or upload it from the sidebar.")
    st.stop()

required_cols = ["Step_No","Document","Prep_Days","Finish_Days_Before_J0","Department","Role"]
miss = [c for c in required_cols if c not in tasks_df.columns]
if miss:
    st.warning(f"Some required columns are missing: {miss}. The app will try to proceed if possible.")

# Apply filters
tasks_filtered = apply_filters(tasks_df, sel_departments, sel_roles, sel_markets, sel_langs)

# Compute dates & daily
dated = compute_dates(tasks_filtered, pd.to_datetime(j0_date))
daily = explode_daily(dated)

# Capacity editors (three levels)
def make_capacity_frames(tasks):
    if tasks is None or tasks.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    dept = pd.DataFrame({"Department": sorted(tasks["Department"].dropna().unique())})
    dept["Headcount"] = 0.0
    dept["Available_FTE"] = 0.0
    role = pd.DataFrame({"Role": sorted(tasks["Role"].dropna().unique())})
    role["Headcount"] = 0.0
    role["Available_FTE"] = 0.0
    pair = tasks[["Department","Role"]].dropna().drop_duplicates().sort_values(["Department","Role"]).reset_index(drop=True)
    pair["Headcount"] = 0.0
    pair["Available_FTE"] = 0.0
    return dept, role, pair

dept_cap, role_cap, pair_cap = make_capacity_frames(tasks_filtered)
dept_cap, role_cap, pair_cap = apply_hr_capacity(hr_df, dept_cap, role_cap, pair_cap, per_person_cap)

st.divider()
st.subheader("👥 Capacity (edit or auto-load from ./data/HR_Capacity.csv)")

# Capacity-specific filters (same as Gantt filters)
with st.expander("⚙️ Options — Capacity Filters", expanded=False):
    available_departments_cap = sorted(tasks_df["Department"].dropna().unique())
    sel_departments = st.multiselect(
        "Filter by Department",
        available_departments_cap,
        default=available_departments_cap,
        key="cap_depts"
    )

    available_roles_cap = sorted(tasks_df["Role"].dropna().unique())
    sel_roles = st.multiselect(
        "Filter by Role",
        available_roles_cap,
        default=available_roles_cap,
        key="cap_roles"
    )

    # Market and Language filters if present
    available_markets_cap = sorted(tasks_df["Market"].dropna().unique()) if "Market" in tasks_df.columns else []
    if available_markets_cap:
        sel_markets = st.multiselect(
            "Filter by Market",
            available_markets_cap,
            default=available_markets_cap,
            key="cap_markets"
        )
    else:
        sel_markets = []

    available_langs_cap = sorted(tasks_df["Language"].dropna().unique()) if "Language" in tasks_df.columns else []
    if available_langs_cap:
        sel_langs = st.multiselect(
            "Filter by Language",
            available_langs_cap,
            default=available_langs_cap,
            key="cap_langs"
        )
    else:
        sel_langs = []

# Apply the capacity filters to tasks
tasks_filtered = apply_filters(tasks_df, sel_departments, sel_roles, sel_markets, sel_langs)

# Use filtered tasks for capacity calculations
dept_cap, role_cap, pair_cap = make_capacity_frames(tasks_filtered)
dept_cap, role_cap, pair_cap = apply_hr_capacity(hr_df, dept_cap, role_cap, pair_cap, per_person_cap)

tab_d, tab_r, tab_p = st.tabs(["Department","Role","Department → Role"])
with tab_d:
    dept_cap = st.data_editor(dept_cap, use_container_width=True, num_rows="dynamic", key="dept_cap_editor_v2_fix")
    st.caption("If Headcount is provided, Available_FTE will be computed as Headcount × per-person cap unless you set it explicitly.")
with tab_r:
    role_cap = st.data_editor(role_cap, use_container_width=True, num_rows="dynamic", key="role_cap_editor_v2_fix")
with tab_p:
    pair_cap = st.data_editor(pair_cap, use_container_width=True, num_rows="dynamic", key="pair_cap_editor_v2_fix")

# KPIs
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    total_fte_days = float(dated["Allocated_FTE_Days"].sum()) if "Allocated_FTE_Days" in dated.columns else float((dated["Allocated_FTE"]*dated["Prep_Days"]).sum() if not dated.empty and "Allocated_FTE" in dated.columns else 0.0)
    st.metric("Total FTE-Days (scope)", f"{total_fte_days:,.1f}")
with col2:
    est_start = pd.to_datetime(dated["Start_Date"]).min().date() if not dated.empty else "—"
    est_finish = pd.to_datetime(dated["Finish_Date"]).max().date() if not dated.empty else "—"
    st.metric("Timeline", f"{est_start} → {est_finish}")
with col3:
    st.metric("Tasks", int(dated.shape[0]))



# Gantt chart logic
st.divider()
st.subheader("🗂️ Task Timeline (Gantt)")

# Use the same filtered data as main dashboard
gantt_tasks = apply_filters(tasks_df, sel_departments, sel_roles, sel_markets, sel_langs)
gantt_dated = compute_dates(gantt_tasks, pd.to_datetime(j0_date))

if gantt_dated.empty:
    st.info("No tasks to display.")
else:
    # Gantt chart logic (adapted from Process Methodology – Gantt, but using dashboard style)
    gantt_df = gantt_dated.copy()
    cpm = compute_cpm(gantt_dated)
    if not cpm.empty:
        gantt_df = gantt_df.merge(cpm[["Step_No","Critical","Slack"]], on="Step_No", how="left")
        gantt_df["Critical"] = gantt_df["Critical"].fillna(False)

    fig_gantt = go.Figure()
    colors = px.colors.qualitative.Plotly
    dept_colors = {d: colors[i % len(colors)] for i, d in enumerate(sorted(gantt_df["Department"].dropna().unique()))}
    for idx, row in gantt_df.iterrows():
        color = dept_colors.get(row["Department"], "#888")
        line_width = 6
        is_critical = bool(row.get("Critical", False))
        fig_gantt.add_trace(go.Scatter(
            x=[row["Start_Date"], row["Finish_Date"]],
            y=[row["Document"], row["Document"]],
            mode="lines",
            line=dict(color=color, width=line_width if not is_critical else line_width+2, dash="solid" if not is_critical else "dash"),
            name=row["Department"],
            showlegend=False,
            hovertemplate=(
                f"Document: {row['Document']}<br>"
                f"Department: {row['Department']}<br>"
                f"Role: {row['Role']}<br>"
                f"Start: {row['Start_Date']}<br>"
                f"Finish: {row['Finish_Date']}<br>"
                f"Step: {row['Step_No']}<br>"
                f"Slack: {row.get('Slack','')}<br>"
                f"{'Critical' if is_critical else ''}"
            ),
            marker=dict(symbol="line-ns-open"),
        ))
    # Add extra horizontal space to ensure the full timeline is visible
    x_min = pd.to_datetime(gantt_df["Start_Date"]).min()
    x_max = pd.to_datetime(gantt_df["Finish_Date"]).max()
    if not pd.isna(x_min) and not pd.isna(x_max):
        x_range = (x_min - pd.Timedelta(days=7), x_max + pd.Timedelta(days=7))
        fig_gantt.update_xaxes(range=x_range, tickfont=dict(size=8))
    else:
        fig_gantt.update_xaxes(tickfont=dict(size=8))
    fig_gantt.update_yaxes(autorange="reversed", tickfont=dict(size=8))
    fig_gantt.update_layout(
        height=650,
        margin=dict(l=20, r=20, t=40, b=20),
        title="Task Plan (dates adjusted with predecessors & lag where provided)",
        xaxis_title="Date",
        yaxis_title="Document",
    )
    st.plotly_chart(fig_gantt, use_container_width=True)


# Aggregation & Charts
st.divider()
st.subheader("📈 Demand vs Capacity")

# Use default breakdown level and chart type
level_choice = "Department"
chart_type = "Stacked Area"

agg = aggregate_time(daily, "W-MON" if agg_freq=="Weekly" else "MS", level_choice)
merged_dept = merge_capacity(agg, level_choice, dept_cap, role_cap, pair_cap, per_person_cap)

if merged_dept.empty:
    st.info("No data to plot (check your filters or CSV structure).")
else:
    if level_choice == "Department":
        series = "Department"
    elif level_choice == "Role":
        series = "Role"
    else:
        merged_dept["Pair"] = merged_dept["Department"].astype(str) + " | " + merged_dept["Role"].astype(str)
        series = "Pair"

    pivot = merged_dept.pivot_table(index="PeriodStart", columns=series, values="FTE", aggfunc="sum").fillna(0.0)
    pivot = pivot.sort_index()
    cap_tot = merged_dept.groupby("PeriodStart", as_index=False)["Capacity_FTE_Period"].sum().set_index("PeriodStart").sort_index()

    if chart_type == "Stacked Area":
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Scatter(x=pivot.index, y=pivot[col], mode="lines", stackgroup="one", name=str(col)))
        if not cap_tot.empty:
            fig.add_trace(go.Scatter(x=cap_tot.index, y=cap_tot["Capacity_FTE_Period"], mode="lines", name="Capacity (period)", line=dict(width=3, dash="dash")))
        fig.update_layout(height=420, margin=dict(l=10,r=10,t=40,b=10), title=f"{agg_freq} FTE Demand — by {level_choice}", xaxis_title="Period Start", yaxis_title="FTE (per period)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        fig = go.Figure()
        for col in pivot.columns:
            fig.add_trace(go.Bar(x=pivot.index, y=pivot[col], name=str(col)))
        if not cap_tot.empty:
            fig.add_trace(go.Scatter(x=cap_tot.index, y=cap_tot["Capacity_FTE_Period"], mode="lines", name="Capacity (period)", line=dict(width=3, dash="dash")))
        fig.update_layout(barmode="stack", height=420, margin=dict(l=10,r=10,t=40,b=10), title=f"{agg_freq} FTE Demand — by {level_choice}", xaxis_title="Period Start", yaxis_title="FTE (per period)")
        st.plotly_chart(fig, use_container_width=True)

# Critical Path section (now after capacity, wrapped in expander)
with st.expander("🔥 Show Critical Path Tasks", expanded=False):
    st.subheader("🧠 Critical Path & Slack (CPM)")
    cpm_gantt = compute_cpm(gantt_dated)
    if cpm_gantt.empty:
        st.info("No predecessor data found — cannot compute CPM. Add a 'Predecessor_Step' column to enable this.")
    else:
        st.dataframe(cpm_gantt.sort_values("ES").reset_index(drop=True), use_container_width=True)
        st.caption("Tasks with **Slack = 0** are critical. ES/EF/LS/LF are in *days* (relative model units).")


# Exports
st.divider()
st.subheader("📥 Export Data")

colA, colB, colC = st.columns(3)
with colA:
    if not daily.empty:
        st.download_button(
            "Download Daily Demand (CSV)",
            data=daily.to_csv(index=False).encode("utf-8"),
            file_name="daily_demand.csv",
            mime="text/csv"
        )
with colB:
    agg_for_export = aggregate_time(daily, "W-MON" if agg_freq=="Weekly" else "MS", level_choice)
    if not agg_for_export.empty:
        st.download_button(
            f"Download {agg_freq} Demand by {level_choice} (CSV)",
            data=agg_for_export.to_csv(index=False).encode("utf-8"),
            file_name=f"demand_{agg_freq.lower()}_{level_choice.replace(' ','_')}.csv",
            mime="text/csv"
        )
with colC:
    if not dated.empty:
        st.download_button(
            "Download Task Plan with Dates (CSV)",
            data=dated.sort_values(["Start_Date","Finish_Date"]).to_csv(index=False).encode("utf-8"),
            file_name="task_plan_with_dates.csv",
            mime="text/csv"
        )

with st.expander("ℹ️ Notes"):
    st.markdown(
        """
        - Place your CSVs in a local **./data** folder to auto-load them without uploads.
        - **Capacity** per period = `Available_FTE (daily)` × `number of days in period`.
        - When **Headcount** is provided but `Available_FTE` is blank, capacity = `Headcount × per-person cap`.
        - Add **Predecessor_Step** and **Lag_Days** to enable CPM and date adjustment.
        - Include **Market** / **Language** columns to enable those filters.
        - Provide **Rates.csv** to compute **Budget**; or add `Rate_per_FTE_Day` to tasks.
        """
    )
