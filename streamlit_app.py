import os
import re
from datetime import datetime, timedelta

import io

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# Mapping "process label" -> CSV file
# ------------------------------------------------------------------
PROCESS_CSV = {
    "MAA – new active substance (centralised)": "DayDataMaa.csv",
    "Variation Type IA / IB (centralised)": "DayDataVariationTypeIAIB.csv",
    "Variation Type II (quality or new indication)": "DayDataVariationType2.csv",
    "FDA (new form / strength)": "DayDataFDAMaa.csv",
}

@st.cache_data
def load_data(process_label: str) -> pd.DataFrame:
    path = PROCESS_CSV[process_label]
    if not os.path.exists(path):
        st.error(f"Fichier {path} introuvable.")
        st.stop()

    df = pd.read_csv(path)
    if "Effort_Days" not in df.columns:
        df["Effort_Days"] = df["Prep_Days"] * 0.8
    df.columns = [c.strip() for c in df.columns]
    required = {"Step_No", "Document", "Prep_Days", "Predecessor", "Concurrency_Group"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"Colonnes manquantes dans {path} : {', '.join(sorted(missing))}.")
        st.stop()
    df["Step_No"] = pd.to_numeric(df["Step_No"], errors="coerce").astype("Int64")
    df["Concurrency_Group"] = pd.to_numeric(df["Concurrency_Group"], errors="coerce").astype("Int64")
    df["Prep_Days"] = pd.to_numeric(df["Prep_Days"], errors="coerce").fillna(0).clip(lower=0).astype(int)
    return df

GROUP_COLOR_MAP = {
    "1": "#2ecc40", "2": "#0074d9", "3": "#ff851b", "4": "#b10dc9", "5": "#ff4136",
    "6": "#7fdbff", "7": "#3d9970", "8": "#f012be", "9": "#85144b", "10": "#aaaaaa",
}

def compute_schedule(df: pd.DataFrame, j0_datetime: datetime) -> pd.DataFrame:
    gantt_df = df.copy(deep=True)
    gantt_df = gantt_df[gantt_df["Concurrency_Group"].notna()].copy()
    group_order = sorted(gantt_df["Concurrency_Group"].dropna().astype(int).unique())
    project_start = j0_datetime - timedelta(days=365)
    prev_group_max_finish = {}
    gantt_df["Start"] = pd.NaT
    gantt_df["Finish"] = pd.NaT
    for idx, group in enumerate(group_order):
        group_start = project_start if idx == 0 else prev_group_max_finish[group_order[idx - 1]]
        mask = gantt_df["Concurrency_Group"].astype(int) == group
        gantt_df.loc[mask, "Start"] = pd.to_datetime(group_start)
        gantt_df.loc[mask, "Finish"] = gantt_df.loc[mask, "Start"] + pd.to_timedelta(gantt_df.loc[mask, "Prep_Days"], unit="D")
        prev_group_max_finish[group] = gantt_df.loc[mask, "Finish"].max()
    gantt_df["Start"] = pd.to_datetime(gantt_df["Start"])
    gantt_df["Finish"] = pd.to_datetime(gantt_df["Finish"])
    gantt_df = gantt_df.sort_values(by=["Concurrency_Group", "Prep_Days", "Document"], ascending=[True, True, True]).reset_index(drop=True)
    return gantt_df

def compute_weekly_load(df: pd.DataFrame, fte_available: float) -> pd.DataFrame:
    expanded = []
    for _, row in df.iterrows():
        start = row["Start"].date()
        end = row["Finish"].date()
        days = (end - start).days or 1
        daily_effort = row["Effort_Days"] / days
        for i in range(days):
            current_day = start + timedelta(days=i)
            week = current_day.isocalendar()[1]
            year = current_day.isocalendar()[0]
            expanded.append({"Week": f"{year}-W{week:02d}", "Date": current_day, "Effort_Days": daily_effort})
    df_expanded = pd.DataFrame(expanded)
    weekly = df_expanded.groupby("Week").agg(Total_Effort_Days=("Effort_Days", "sum")).reset_index()
    weekly["Load_%"] = (weekly["Total_Effort_Days"] / (5 * fte_available)) * 100
    return weekly

def main() -> None:
    st.set_page_config(page_title="Gantt Chart – Process Tracking", layout="wide")
    st.title("Regulatory Process Tracking & Gantt Chart")
    with st.expander("Options", expanded=True):
        process = st.selectbox("Choose process type", list(PROCESS_CSV.keys()))
        df = load_data(process)
        planning_mode = st.radio("Planning anchor point", ["Use J0 as submission date (classic)", "Use custom project start date"], index=0)
        anchor_date = st.date_input("Select anchor date", value=datetime.today())
        anchor_datetime = datetime.combine(anchor_date, datetime.min.time())
        reverse_from_j0 = planning_mode.startswith("Use J0")
        gantt_df = compute_schedule(df, anchor_datetime if reverse_from_j0 else anchor_datetime + timedelta(days=365))
        do_filter = st.checkbox("Enable Gantt filters", value=False)
        if do_filter:
            available_groups = sorted([int(g) for g in gantt_df["Concurrency_Group"].dropna().unique()])
            selected_groups = st.multiselect("Groups to display", options=available_groups, default=available_groups)
            available_tasks = gantt_df[gantt_df["Concurrency_Group"].isin(selected_groups)]["Document"].astype(str).tolist()
            selected_tasks = st.multiselect("Tasks to display (optional)", options=available_tasks, default=available_tasks)
            gantt_df = gantt_df[gantt_df["Concurrency_Group"].isin(selected_groups) & gantt_df["Document"].isin(selected_tasks)]
    gantt_df["Concurrency_Group"] = gantt_df["Concurrency_Group"].astype(str)
    color_map = {g: GROUP_COLOR_MAP.get(g, "#111111") for g in gantt_df["Concurrency_Group"].unique()}
    fig = px.timeline(gantt_df, x_start="Start", x_end="Finish", y="Document", color="Concurrency_Group", color_discrete_map=color_map, hover_data=["Step_No", "Prep_Days", "Start", "Finish", "Concurrency_Group", "Predecessor"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=600, xaxis_title="Date", yaxis_title="Document", margin=dict(l=0, r=20, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    min_start = gantt_df["Start"].min()
    max_finish = gantt_df["Finish"].max()
    st.markdown(f"**Project Duration:** {(max_finish - min_start).days} days (from `{min_start.date()}` to `{max_finish.date()}`)")
    safe_process = re.sub(r"[^\w]+", "_", process)
    anchor_label = "J0" if reverse_from_j0 else "Start"
    date_str = anchor_date.strftime("%Y-%m-%d")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button("Download **Gantt** schedule (CSV)", data=gantt_df.to_csv(index=False).encode("utf-8"), file_name=f"Gantt_{safe_process}_{anchor_label}_{date_str}.csv", mime="text/csv")
    with col2:
        fte = st.number_input("FTE available", min_value=0.1, value=1.0, step=0.1)
        weekly_load = compute_weekly_load(gantt_df, fte)
        st.download_button("Download weekly **FTE** load (CSV)", data=weekly_load.to_csv(index=False).encode("utf-8"), file_name=f"FTE_Weekly_Load_{safe_process}_{date_str}.csv", mime="text/csv")
    with st.expander("🔥 Show Critical Path Tasks"):
        gantt_df["Slack_Days"] = gantt_df.groupby("Concurrency_Group")["Prep_Days"].transform("max") - gantt_df["Prep_Days"]
        gantt_df["Critical"] = gantt_df["Slack_Days"] == 0
        gantt_df["Week_ISO"] = gantt_df["Start"].dt.isocalendar().week.astype(str).str.zfill(2)
        gantt_df["Year_ISO"] = gantt_df["Start"].dt.isocalendar().year.astype(str)
        gantt_df["Week_Label"] = gantt_df["Year_ISO"] + "-W" + gantt_df["Week_ISO"]
        weekly_load = compute_weekly_load(gantt_df, fte)
        gantt_df = gantt_df.merge(weekly_load[["Week", "Load_%"]], left_on="Week_Label", right_on="Week", how="left")
        def compute_urgency(row):
            score = 50 if row["Critical"] else 0
            overload = max(0, row.get("Load_%", 0) - 100)
            return score + overload
        gantt_df["Urgency_Score"] = gantt_df.apply(compute_urgency, axis=1)
        critical_tasks = gantt_df[gantt_df["Critical"]].copy()
        display_df = critical_tasks[["Step_No", "Document", "Prep_Days", "Effort_Days", "Start", "Finish", "Concurrency_Group", "Urgency_Score"]].rename(columns={"Step_No": "Step", "Prep_Days": "Duration (Days)", "Effort_Days": "Effort (FTE Days)", "Concurrency_Group": "Group"})
        st.markdown("Les tâches critiques sont celles qui n'ont **aucune marge de retard (Slack = 0)**.")
        st.dataframe(display_df, use_container_width=True)

if __name__ == "__main__":
    main()